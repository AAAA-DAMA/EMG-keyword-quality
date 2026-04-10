#!/usr/bin/env python3
# coding: utf-8
"""
总控脚本（基于你原来的 run_batch 思路）：
1) 直接调用原始 finallv2.py 训练
2) 先跑 EMG_BERT（bert-base-chinese），再跑 EMG_RoBERTa（hfl/chinese-roberta-wwm-ext）
3) 每个模型跑 5 个随机种子
4) 训练完成后，读取各 run 的 best_model.pt
5) 再单独对噪声测试集进行评测并汇总

注意：
- 训练阶段不改 finallv2.py 的主流程
- 噪声评测阶段单独加载 best_model.pt，并使用“不增强版 eval collate”
- 默认参数按 24G 显存做了更稳的设置：batch_size=8, grad_acc=4, max_len=384
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import re
import csv
import gc
import json
import time
import argparse
import logging
import subprocess
import importlib.util
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger("run_finallv2_train_then_noise_eval")


SEEDS = [42, 52, 85, 261, 112233]

MODEL_SPECS = [
    {
        "model_key": "emg_v2_bert_base",
        "model_name": "bert-base-chinese",
    },
    {
        "model_key": "emg_v2_roberta_wwm_ext",
        "model_name": "hfl/chinese-roberta-wwm-ext",
    },
]

# 比你旧脚本更稳的默认参数
DEFAULT_HYPER = {
    "--model_version": "v2",
    "--lr": "2e-5",
    "--batch_size": "16",
    "--gradient_accumulation_steps": "4",
    "--epochs": "6",
    "--max_len": "384",
}

TEST_FILES = [
    "test_clean.json",
    "test_noise_1.json",
    "test_noise_2.json",
    "test_noise_3.json",
    "test_noise_all.json",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std_str(vals: List[float], digits: int = 4) -> str:
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=0):.{digits}f}"


def load_module_from_path(module_name: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到脚本文件: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(command: List[str], log_file_path: str):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        text=True,
        bufsize=1,
        env=os.environ.copy()
    )

    full_output = []
    logger.info(f"日志保存在: {log_file_path}")
    with open(log_file_path, "w", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            full_output.append(line)

    process.wait()
    return process.returncode, "".join(full_output)


def extract_train_metrics(log_text: str):
    dev_pattern = r"训练完成。最佳验证 F1=(\d+\.\d+)，出现于 epoch (\d+)"
    dev_matches = re.findall(dev_pattern, log_text)
    best_dev_f1 = None
    best_dev_epoch = None
    if dev_matches:
        best_dev_f1 = float(dev_matches[-1][0])
        best_dev_epoch = int(dev_matches[-1][1])

    test_pattern = r"测试集 - Acc: (\d+\.\d+), F1: (\d+\.\d+)"
    test_matches = re.findall(test_pattern, log_text)
    test_acc = None
    test_f1 = None
    if test_matches:
        test_acc = float(test_matches[-1][0])
        test_f1 = float(test_matches[-1][1])

    return {
        "best_dev_f1": best_dev_f1,
        "best_dev_epoch": best_dev_epoch,
        "internal_test_acc": test_acc,
        "internal_test_f1": test_f1,
    }


def save_predictions_csv(outfile, ids, labels, preds, probs):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "真实标签", "预测标签", "概率_0", "概率_1"])
        for i, t, p, pr in zip(ids, labels, preds, probs):
            writer.writerow([i, t, p, round(pr[0], 6), round(pr[1], 6)])


def make_eval_collate_fn_no_aug(mod, tokenizer, max_length: int = 512):
    """
    复刻 finallv2.py 的 term_mask 构造，但关闭训练增强：
    - 不做正样本随机删词
    - 不做关键词乱序
    """
    try:
        import jieba
        have_jieba = True
    except Exception:
        have_jieba = False

    def collate(batch: List[Dict]):
        texts_a = [it['abst'] for it in batch]
        texts_b = []
        ids = [it['id'] for it in batch]
        labels = [(-1 if it['label'] is None else int(it['label'])) for it in batch]

        for it in batch:
            kw_list = it['keywords']
            texts_b.append('；'.join(kw_list) if kw_list else '')

        enc = tokenizer(
            text=texts_a,
            text_pair=texts_b,
            padding='longest',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = enc['input_ids']
        _, seq_len = input_ids.shape
        term_mask = torch.zeros_like(input_ids)

        for i, it in enumerate(batch):
            kws = it['keywords']
            if not kws:
                continue
            ids_list = input_ids[i].tolist()

            for kw in kws:
                kw = str(kw).strip()
                if not kw:
                    continue
                kw_tokens = tokenizer.tokenize(kw)
                if len(kw_tokens) == 0:
                    continue

                try:
                    kw_ids = tokenizer.convert_tokens_to_ids(kw_tokens)
                except Exception:
                    kw_ids = []

                matched = False
                if kw_ids:
                    for j in range(len(ids_list) - len(kw_ids) + 1):
                        if ids_list[j:j + len(kw_ids)] == kw_ids:
                            term_mask[i, j:j + len(kw_ids)] = 1
                            matched = True
                    if matched:
                        continue

                try:
                    decoded = tokenizer.decode(ids_list, clean_up_tokenization_spaces=True)
                except Exception:
                    decoded = None

                if decoded and kw in decoded:
                    try:
                        sub_enc = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)
                        off_map = sub_enc['offset_mapping']
                        start_char = decoded.find(kw)
                        if start_char >= 0:
                            end_char = start_char + len(kw)
                            for t_idx, (s, e) in enumerate(off_map):
                                if s < end_char and e > start_char and t_idx < seq_len:
                                    term_mask[i, t_idx] = 1
                                    matched = True
                    except Exception:
                        pass

                if (not matched) and have_jieba:
                    import jieba as jb
                    pieces = list(jb.cut(kw))
                    for piece in pieces:
                        piece_tokens = tokenizer.tokenize(piece)
                        if not piece_tokens:
                            continue
                        piece_ids = tokenizer.convert_tokens_to_ids(piece_tokens)
                        for j in range(len(ids_list) - len(piece_ids) + 1):
                            if ids_list[j:j + len(piece_ids)] == piece_ids:
                                term_mask[i, j:j + len(piece_ids)] = 1

        token_type_ids = enc.get('token_type_ids', torch.zeros_like(input_ids))
        return {
            'ids': ids,
            'input_ids': input_ids,
            'attention_mask': enc['attention_mask'],
            'token_type_ids': token_type_ids,
            'term_mask': term_mask.long(),
            'labels': torch.tensor(labels, dtype=torch.long),
            'raw_texts': [it['abst'] for it in batch],
        }

    return collate


def evaluate_noise_sets(
    script_path: str,
    model_name: str,
    checkpoint_path: str,
    noise_test_paths: Dict[str, str],
    output_dir: str,
    batch_size: int = 8,
    max_len: int = 384,
):
    mod = load_module_from_path("finallv2_eval_mod", script_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = mod.AutoTokenizer.from_pretrained(model_name)

    model = mod.TermSentParaBERT_V2_Fast(pretrained_name=model_name, dropout_prob=0.1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    collate_fn = make_eval_collate_fn_no_aug(mod, tokenizer, max_length=max_len)

    results = {}
    for test_name, test_path in noise_test_paths.items():
        ds = mod.AbstKeywordDataset(test_path)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        acc, f1, ids, labels, preds, probs = mod.evaluate(model, loader, device, tokenizer)
        results[test_name] = {"acc": float(acc), "f1": float(f1)}
        logger.info(f"[NoiseEval][{model_name}] {test_name}: acc={acc:.4f}, f1={f1:.4f}")
        save_predictions_csv(
            os.path.join(output_dir, f"predictions_{test_name}.csv"),
            ids, labels, preds, probs
        )

    save_json(os.path.join(output_dir, "noise_metrics.json"), results)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def write_train_summary_csv(path: str, rows: List[Dict]):
    header = [
        "model_key", "seed", "model_name",
        "best_dev_f1", "best_dev_epoch",
        "internal_test_acc", "internal_test_f1",
        "run_dir"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_noise_detailed_csv(path: str, rows: List[Dict], test_names: List[str]):
    header = ["model_key", "seed", "model_name"] + [f"{t}_acc" for t in test_names] + [f"{t}_f1" for t in test_names]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            line = [row["model_key"], row["seed"], row["model_name"]]
            for t in test_names:
                line.append(row["noise_results"][t]["acc"])
            for t in test_names:
                line.append(row["noise_results"][t]["f1"])
            writer.writerow(line)


def aggregate_noise_results(rows: List[Dict], test_names: List[str]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_key"]].append(row)

    agg_rows = []
    for model_key, items in grouped.items():
        model_name = items[0]["model_name"]
        for t in test_names:
            accs = [x["noise_results"][t]["acc"] for x in items]
            f1s = [x["noise_results"][t]["f1"] for x in items]
            agg_rows.append({
                "model_key": model_key,
                "model_name": model_name,
                "test_name": t,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=0)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=0)),
                "n_runs": len(items),
            })
    return agg_rows


def write_agg_csv(path: str, agg_rows: List[Dict]):
    header = ["model_key", "model_name", "test_name", "acc_mean", "acc_std", "f1_mean", "f1_std", "n_runs"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)


def write_paper_tables(output_dir: str, rows: List[Dict], test_names: List[str]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_key"]].append(row)

    model_order = ["emg_v2_bert_base", "emg_v2_roberta_wwm_ext"]

    f1_csv = os.path.join(output_dir, "paper_table_macro_f1.csv")
    with open(f1_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + test_names)
        for mk in model_order:
            items = grouped.get(mk, [])
            if not items:
                continue
            row = [mk]
            for t in test_names:
                vals = [x["noise_results"][t]["f1"] for x in items]
                row.append(mean_std_str(vals, digits=4))
            writer.writerow(row)

    acc_csv = os.path.join(output_dir, "paper_table_accuracy.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + test_names)
        for mk in model_order:
            items = grouped.get(mk, [])
            if not items:
                continue
            row = [mk]
            for t in test_names:
                vals = [x["noise_results"][t]["acc"] for x in items]
                row.append(mean_std_str(vals, digits=4))
            writer.writerow(row)

    drop_csv = os.path.join(output_dir, "paper_table_f1_drop_vs_clean.csv")
    with open(drop_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "test_noise_1", "test_noise_2", "test_noise_3", "test_noise_all"])
        for mk in model_order:
            items = grouped.get(mk, [])
            if not items:
                continue
            clean_vals = np.array([x["noise_results"]["test_clean"]["f1"] for x in items], dtype=float)
            row = [mk]
            for t in ["test_noise_1", "test_noise_2", "test_noise_3", "test_noise_all"]:
                cur_vals = np.array([x["noise_results"][t]["f1"] for x in items], dtype=float)
                drops = clean_vals - cur_vals
                row.append(mean_std_str(drops.tolist(), digits=4))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True, help="原始 finallv2.py 路径")
    parser.add_argument("--original_train", type=str, required=True)
    parser.add_argument("--original_dev", type=str, required=True)
    parser.add_argument("--noise_test_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,52,85,261,112233")
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.script_path):
        raise FileNotFoundError(f"找不到训练脚本: {args.script_path}")

    ensure_dir(args.output_root)
    save_json(os.path.join(args.output_root, "master_config.json"), vars(args))

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    noise_test_paths = {Path(p).stem: p for p in [os.path.join(args.noise_test_dir, fn) for fn in TEST_FILES]}
    for p in noise_test_paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到噪声测试集: {p}")

    train_summary_rows = []
    noise_rows = []

    logger.info("=== 开始 finallv2 原始训练 + 独立噪声评测 ===")
    logger.info(f"Seeds: {seeds}")

    for spec in MODEL_SPECS:
        for seed in seeds:
            run_dir = os.path.join(args.output_root, spec["model_key"], f"seed_{seed}")
            ensure_dir(run_dir)
            train_log = os.path.join(run_dir, "train_log.txt")
            train_metrics_json = os.path.join(run_dir, "train_metrics.json")
            ckpt_path = os.path.join(run_dir, "best_model.pt")
            noise_metrics_json = os.path.join(run_dir, "noise_metrics.json")

            logger.info("=" * 100)
            logger.info(f"开始运行: model={spec['model_key']} | seed={seed}")

            if args.skip_existing and os.path.exists(ckpt_path) and os.path.exists(train_metrics_json):
                logger.info(f"[SKIP-TRAIN] 已存在训练结果: {run_dir}")
                train_metrics = load_json(train_metrics_json)
            else:
                cmd = [
                    sys.executable, args.script_path,
                    "--seed", str(seed),
                    "--output_dir", run_dir,
                    "--original_train", args.original_train,
                    "--original_dev", args.original_dev,
                ]
                for k, v in DEFAULT_HYPER.items():
                    cmd.extend([k, v])
                cmd.extend(["--model_name", spec["model_name"]])

                retcode, log_text = run_command(cmd, train_log)
                if retcode != 0:
                    logger.error(f"训练失败: model={spec['model_key']} | seed={seed} | returncode={retcode}")
                    continue

                train_metrics = extract_train_metrics(log_text)
                train_metrics["model_key"] = spec["model_key"]
                train_metrics["model_name"] = spec["model_name"]
                train_metrics["seed"] = seed
                train_metrics["run_dir"] = run_dir
                save_json(train_metrics_json, train_metrics)

            train_summary_rows.append({
                "model_key": spec["model_key"],
                "seed": seed,
                "model_name": spec["model_name"],
                "best_dev_f1": train_metrics.get("best_dev_f1"),
                "best_dev_epoch": train_metrics.get("best_dev_epoch"),
                "internal_test_acc": train_metrics.get("internal_test_acc"),
                "internal_test_f1": train_metrics.get("internal_test_f1"),
                "run_dir": run_dir,
            })

            if not os.path.exists(ckpt_path):
                logger.error(f"缺少 best_model.pt，跳过噪声评测: {ckpt_path}")
                continue

            if args.skip_existing and os.path.exists(noise_metrics_json):
                logger.info(f"[SKIP-EVAL] 已存在噪声评测结果: {noise_metrics_json}")
                noise_results = load_json(noise_metrics_json)
            else:
                noise_results = evaluate_noise_sets(
                    script_path=args.script_path,
                    model_name=spec["model_name"],
                    checkpoint_path=ckpt_path,
                    noise_test_paths=noise_test_paths,
                    output_dir=run_dir,
                    batch_size=8,
                    max_len=384,
                )

            noise_rows.append({
                "model_key": spec["model_key"],
                "seed": seed,
                "model_name": spec["model_name"],
                "noise_results": noise_results,
            })

    test_names = list(noise_test_paths.keys())

    write_train_summary_csv(os.path.join(args.output_root, "train_summary.csv"), train_summary_rows)
    write_noise_detailed_csv(os.path.join(args.output_root, "noise_detailed.csv"), noise_rows, test_names)

    agg_rows = aggregate_noise_results(noise_rows, test_names)
    write_agg_csv(os.path.join(args.output_root, "noise_aggregated.csv"), agg_rows)
    write_paper_tables(args.output_root, noise_rows, test_names)

    save_json(os.path.join(args.output_root, "all_noise_results.json"), {
        "train_summary": train_summary_rows,
        "noise_results": noise_rows,
        "aggregated": agg_rows,
    })

    logger.info("=" * 100)
    logger.info("全部完成")
    logger.info(f"训练汇总: {os.path.join(args.output_root, 'train_summary.csv')}")
    logger.info(f"噪声详细结果: {os.path.join(args.output_root, 'noise_detailed.csv')}")
    logger.info(f"噪声汇总长表: {os.path.join(args.output_root, 'noise_aggregated.csv')}")
    logger.info(f"论文主表(Macro-F1): {os.path.join(args.output_root, 'paper_table_macro_f1.csv')}")


if __name__ == "__main__":
    main()
