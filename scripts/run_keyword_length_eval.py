#!/usr/bin/env python3
# coding: utf-8
"""
不同关键词长度分布实验总控脚本（同时兼容 baseline 与 EMG）
---------------------------------------------------------
用途：
1. baseline 模型来自一个脚本（如 xinjizuo.py）
2. EMG 模型来自另一个脚本（如 finallv2.py）
3. 自动读取已经训练好的 best_model.pt
4. 在一个或多个测试集上按“关键词长度分桶”评测
5. 汇总成详细表、聚合表和论文可用表

默认 4 个模型：
- baseline_bert_base
- emg_v2_bert_base
- baseline_roberta_wwm_ext
- emg_v2_roberta_wwm_ext

默认长度桶（按“候选关键词集合平均关键词字数”）：
- len_le_2
- len_eq_3
- len_eq_4
- len_ge_5

示例：
python run_keyword_length_distribution_master.py \
  --baseline_script /root/autodl-tmp/MVPproject/xinjizuo.py \
  --emg_script /root/autodl-tmp/MVPproject/finallv2.py \
  --baseline_ckpt_root /root/autodl-tmp/MVPproject/output_noise_master \
  --emg_ckpt_root /root/autodl-tmp/MVPproject/final_results_emg_retrain \
  --test_paths /root/autodl-tmp/MVPproject/output_attention_optimized_cijuduan/split_data/noise_testsets/test_clean.json \
  --output_dir /root/autodl-tmp/MVPproject/output_length_master
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import csv
import json
import gc
import argparse
import logging
import importlib.util
from collections import defaultdict
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("keyword_length_master")


MODEL_SPECS = [
    {
        "model_key": "baseline_bert_base",
        "family": "baseline",
        "model_name": "bert-base-chinese",
    },
    {
        "model_key": "emg_v2_bert_base",
        "family": "emg",
        "model_name": "bert-base-chinese",
    },
    {
        "model_key": "baseline_roberta_wwm_ext",
        "family": "baseline",
        "model_name": "hfl/chinese-roberta-wwm-ext",
    },
    {
        "model_key": "emg_v2_roberta_wwm_ext",
        "family": "emg",
        "model_name": "hfl/chinese-roberta-wwm-ext",
    },
]

DEFAULT_SEEDS = [42, 52, 85, 261, 112233]


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


def normalize_kw(kw: str) -> str:
    return str(kw).strip()


def keyword_char_len(kw: str) -> int:
    return len(normalize_kw(kw))


def avg_keyword_len(sample: Dict) -> float:
    kws = sample.get("keywords", []) or []
    kws = [normalize_kw(k) for k in kws if normalize_kw(k)]
    if not kws:
        return 0.0
    return float(sum(keyword_char_len(k) for k in kws)) / len(kws)


def bucket_name_from_avg_len(avg_len: float) -> str:
    if avg_len <= 2:
        return "len_le_2"
    elif avg_len <= 3:
        return "len_eq_3"
    elif avg_len <= 4:
        return "len_eq_4"
    else:
        return "len_ge_5"


def build_bucket_indices(items: List[Dict]):
    bucket_indices = {
        "len_le_2": [],
        "len_eq_3": [],
        "len_eq_4": [],
        "len_ge_5": [],
    }
    for idx, item in enumerate(items):
        name = bucket_name_from_avg_len(avg_keyword_len(item))
        bucket_indices[name].append(idx)
    return bucket_indices


def make_baseline_eval_collate(mod, tokenizer, max_length: int = 512):
    # baseline 原脚本本身没有训练增强，直接复用
    return mod.make_collate_fn(tokenizer, max_length=max_length)


def make_emg_eval_collate(mod, tokenizer, max_length: int = 512):
    """
    EMG 评测专用：关闭训练增强，但保留完整 term_mask 构造逻辑
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


def evaluate_subset_baseline(mod, model, dataloader, device):
    acc, f1, ids, labels, preds, probs = mod.evaluate(model, dataloader, device)
    return {"acc": float(acc), "f1": float(f1), "n": len(labels)}


def evaluate_subset_emg(mod, model, dataloader, device, tokenizer):
    acc, f1, ids, labels, preds, probs = mod.evaluate(model, dataloader, device, tokenizer)
    return {"acc": float(acc), "f1": float(f1), "n": len(labels)}


def get_checkpoint_path(root: str, model_key: str, seed: int) -> str:
    return os.path.join(root, model_key, f"seed_{seed}", "best_model.pt")


def load_model_and_tokenizer(family: str, mod, model_name: str, checkpoint_path: str, device):
    if family == "baseline":
        tokenizer = mod.AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = mod.TransformerForBinaryClassification(model_name).to(device)
    else:
        tokenizer = mod.AutoTokenizer.from_pretrained(model_name)
        model = mod.TermSentParaBERT_V2_Fast(pretrained_name=model_name, dropout_prob=0.1).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, tokenizer


def evaluate_one_run(
    baseline_mod,
    emg_mod,
    spec: Dict,
    checkpoint_path: str,
    test_path: str,
    batch_size: int,
    max_len: int,
    device,
):
    dataset_name = Path(test_path).stem
    mod = baseline_mod if spec["family"] == "baseline" else emg_mod
    ds = mod.AbstKeywordDataset(test_path)

    bucket_indices = build_bucket_indices(ds.items)

    model, tokenizer = load_model_and_tokenizer(
        family=spec["family"],
        mod=mod,
        model_name=spec["model_name"],
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if spec["family"] == "baseline":
        collate_fn = make_baseline_eval_collate(mod, tokenizer, max_length=max_len)
    else:
        collate_fn = make_emg_eval_collate(mod, tokenizer, max_length=max_len)

    rows = []
    for bucket_name, indices in bucket_indices.items():
        if len(indices) == 0:
            rows.append({
                "dataset": dataset_name,
                "bucket": bucket_name,
                "n": 0,
                "acc": "",
                "f1": "",
            })
            continue

        subset = Subset(ds, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        if spec["family"] == "baseline":
            metrics = evaluate_subset_baseline(mod, model, loader, device)
        else:
            metrics = evaluate_subset_emg(mod, model, loader, device, tokenizer)

        rows.append({
            "dataset": dataset_name,
            "bucket": bucket_name,
            "n": metrics["n"],
            "acc": metrics["acc"],
            "f1": metrics["f1"],
        })

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def write_detailed_csv(path: str, rows: List[Dict]):
    header = ["model_key", "family", "seed", "model_name", "dataset", "bucket", "n", "acc", "f1"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: List[Dict]):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model_key"], row["family"], row["model_name"], row["dataset"], row["bucket"])
        grouped[key].append(row)

    agg_rows = []
    for (model_key, family, model_name, dataset, bucket), items in grouped.items():
        valid_items = [x for x in items if isinstance(x["acc"], (float, int)) and isinstance(x["f1"], (float, int))]
        if len(valid_items) == 0:
            agg_rows.append({
                "model_key": model_key,
                "family": family,
                "model_name": model_name,
                "dataset": dataset,
                "bucket": bucket,
                "n_mean": 0,
                "acc_mean": "",
                "acc_std": "",
                "f1_mean": "",
                "f1_std": "",
                "n_runs": 0,
            })
            continue

        accs = np.array([x["acc"] for x in valid_items], dtype=float)
        f1s = np.array([x["f1"] for x in valid_items], dtype=float)
        ns = np.array([x["n"] for x in valid_items], dtype=float)

        agg_rows.append({
            "model_key": model_key,
            "family": family,
            "model_name": model_name,
            "dataset": dataset,
            "bucket": bucket,
            "n_mean": float(ns.mean()),
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std(ddof=0)),
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(ddof=0)),
            "n_runs": len(valid_items),
        })

    return agg_rows


def write_agg_csv(path: str, rows: List[Dict]):
    header = ["model_key", "family", "model_name", "dataset", "bucket", "n_mean", "acc_mean", "acc_std", "f1_mean", "f1_std", "n_runs"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_paper_tables(output_dir: str, agg_rows: List[Dict]):
    by_dataset = defaultdict(list)
    for row in agg_rows:
        by_dataset[row["dataset"]].append(row)

    model_order = [
        "baseline_bert_base",
        "emg_v2_bert_base",
        "baseline_roberta_wwm_ext",
        "emg_v2_roberta_wwm_ext",
    ]
    bucket_order = ["len_le_2", "len_eq_3", "len_eq_4", "len_ge_5"]

    for dataset_name, rows in by_dataset.items():
        row_map = {(r["model_key"], r["bucket"]): r for r in rows}

        f1_path = os.path.join(output_dir, f"paper_table_length_f1_{dataset_name}.csv")
        with open(f1_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + bucket_order)
            for mk in model_order:
                row = [mk]
                for b in bucket_order:
                    r = row_map.get((mk, b))
                    if not r or r["n_runs"] == 0 or r["f1_mean"] == "":
                        row.append("")
                    else:
                        row.append(f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
                writer.writerow(row)

        acc_path = os.path.join(output_dir, f"paper_table_length_acc_{dataset_name}.csv")
        with open(acc_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + bucket_order)
            for mk in model_order:
                row = [mk]
                for b in bucket_order:
                    r = row_map.get((mk, b))
                    if not r or r["n_runs"] == 0 or r["acc_mean"] == "":
                        row.append("")
                    else:
                        row.append(f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_script", type=str, required=True)
    parser.add_argument("--emg_script", type=str, required=True)
    parser.add_argument("--baseline_ckpt_root", type=str, required=True)
    parser.add_argument("--emg_ckpt_root", type=str, required=True)
    parser.add_argument("--test_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="42,52,85,261,112233")
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    save_json(os.path.join(args.output_dir, "master_config.json"), vars(args))

    baseline_mod = load_module_from_path("baseline_mod", args.baseline_script)
    emg_mod = load_module_from_path("emg_mod", args.emg_script)

    for p in args.test_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到测试集: {p}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    logger.info(f"Seeds: {seeds}")

    detailed_rows = []

    for spec in MODEL_SPECS:
        ckpt_root = args.baseline_ckpt_root if spec["family"] == "baseline" else args.emg_ckpt_root
        for seed in seeds:
            checkpoint_path = get_checkpoint_path(ckpt_root, spec["model_key"], seed)
            if not os.path.exists(checkpoint_path):
                logger.warning(f"缺少 checkpoint，跳过: {checkpoint_path}")
                continue

            logger.info("=" * 100)
            logger.info(f"开始评测: {spec['model_key']} | seed={seed}")

            for test_path in args.test_paths:
                dataset_name = Path(test_path).stem
                try:
                    rows = evaluate_one_run(
                        baseline_mod=baseline_mod,
                        emg_mod=emg_mod,
                        spec=spec,
                        checkpoint_path=checkpoint_path,
                        test_path=test_path,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        device=device,
                    )
                    for row in rows:
                        row["model_key"] = spec["model_key"]
                        row["family"] = spec["family"]
                        row["seed"] = seed
                        row["model_name"] = spec["model_name"]
                        detailed_rows.append(row)
                        if row["acc"] != "" and row["f1"] != "":
                            logger.info(
                                f"[{spec['model_key']}][seed={seed}][{row['dataset']}][{row['bucket']}] "
                                f"n={row['n']} acc={row['acc']:.4f} f1={row['f1']:.4f}"
                            )
                except Exception as e:
                    logger.exception(f"评测失败: model={spec['model_key']} seed={seed} dataset={dataset_name} error={e}")
                    continue

    detailed_csv = os.path.join(args.output_dir, "length_bucket_detailed.csv")
    write_detailed_csv(detailed_csv, detailed_rows)

    agg_rows = aggregate_rows(detailed_rows)
    agg_csv = os.path.join(args.output_dir, "length_bucket_aggregated.csv")
    write_agg_csv(agg_csv, agg_rows)

    write_paper_tables(args.output_dir, agg_rows)
    save_json(os.path.join(args.output_dir, "all_length_bucket_results.json"), {
        "detailed": detailed_rows,
        "aggregated": agg_rows,
    })

    logger.info("=" * 100)
    logger.info("全部完成")
    logger.info(f"详细结果: {detailed_csv}")
    logger.info(f"聚合结果: {agg_csv}")

if __name__ == "__main__":
    main()
