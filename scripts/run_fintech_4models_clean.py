#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
四模型总控脚本（clean 数据集直传版）

当前约定：
- train_csl_style_clean.json  -> 训练脚本的 --original_train
- test_csl_style_clean.json   -> 训练脚本的 --original_dev

注意：
- xinjizuo.py / finallv2.py 内部会自行从 original_train 再切分出内部 dev；
- original_dev 会被它们当作最终 test 使用；
- 因此这里不再支持 --train_path / --dev_path / --test_path 三文件合并模式。
"""

import os
import re
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path
from statistics import mean, pstdev


DEFAULT_SEEDS = [42, 52, 85, 261, 112233]

MODEL_CONFIGS = [
    {
        "model_key": "baseline_bert_base",
        "family": "baseline",
        "model_name": "bert-base-chinese",
        "script_type": "baseline",
    },
    {
        "model_key": "emg_v2_bert_base",
        "family": "emg",
        "model_name": "bert-base-chinese",
        "script_type": "emg",
        "model_version": "v2",
    },
    {
        "model_key": "baseline_roberta_wwm_ext",
        "family": "baseline",
        "model_name": "hfl/chinese-roberta-wwm-ext",
        "script_type": "baseline",
    },
    {
        "model_key": "emg_v2_roberta_wwm_ext",
        "family": "emg",
        "model_name": "hfl/chinese-roberta-wwm-ext",
        "script_type": "emg",
        "model_version": "v2",
    },
]


def _match_last(patterns, text, cast=float):
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                return tuple(cast(x) for x in last)
            return cast(last)
    return None


def extract_metrics(log_text: str):
    best_dev = _match_last([
        r"最佳验证 F1=(\d+\.\d+)，出现于 epoch (\d+)",
        r"最佳验证 F1=(\d+\.\d+),\s*出现于 epoch (\d+)",
        r"更新最佳模型[:：]\s*epoch=(\d+),\s*dev_f1=(\d+\.\d+)",
    ], log_text, cast=str)

    best_dev_f1 = None
    best_dev_epoch = None
    if best_dev is not None:
        if len(best_dev) == 2 and best_dev[0].isdigit() and "." in best_dev[1]:
            best_dev_epoch = int(best_dev[0])
            best_dev_f1 = float(best_dev[1])
        else:
            best_dev_f1 = float(best_dev[0])
            best_dev_epoch = int(best_dev[1])

    test_pair = _match_last([
        r"测试集\s*-\s*Acc:\s*(\d+\.\d+),\s*F1:\s*(\d+\.\d+)",
        r"test_acc=(\d+\.\d+)\s+test_f1=(\d+\.\d+)",
        r"internal_test_acc=(\d+\.\d+)\s+internal_test_f1=(\d+\.\d+)",
    ], log_text)

    internal_test_acc = None
    internal_test_f1 = None
    if test_pair is not None:
        internal_test_acc = float(test_pair[0])
        internal_test_f1 = float(test_pair[1])

    return {
        "best_dev_f1": best_dev_f1,
        "best_dev_epoch": best_dev_epoch,
        "internal_test_acc": internal_test_acc,
        "internal_test_f1": internal_test_f1,
    }


def run_command(cmd, log_file: Path, env=None):
    print(f"\n[RUN] {' '.join(cmd)}")
    print(f"[LOG] {log_file}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines = []
    with open(log_file, "w", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)

    ret = process.wait()
    return ret, "".join(lines)


def build_command(cfg, args, run_output_dir: Path, seed: int):
    script_path = args.baseline_script if cfg["script_type"] == "baseline" else args.emg_script

    cmd = [
        "python", script_path,
        "--original_train", str(args.original_train),
        "--original_dev", str(args.original_dev),
        "--output_dir", str(run_output_dir),
        "--seed", str(seed),
        "--model_name", cfg["model_name"],
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]

    if args.max_len is not None:
        cmd.extend(["--max_len", str(args.max_len)])

    if cfg["script_type"] == "emg":
        cmd.extend(["--model_version", cfg.get("model_version", "v2")])

    return cmd


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) <= 1:
        return 0.0 if xs else None
    return pstdev(xs)


def fmt_mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "NA"
    return f"{safe_mean(xs):.4f} ± {safe_std(xs):.4f}"


def save_csv(path: Path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_script", type=str, required=True)
    parser.add_argument("--emg_script", type=str, required=True)

    parser.add_argument("--original_train", type=str, required=True,
                        help="清洗后的训练集，例如 train_csl_style_clean.json")
    parser.add_argument("--original_dev", type=str, required=True,
                        help="清洗后的测试集，例如 test_csl_style_clean.json")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    args.original_train = str(Path(args.original_train).resolve())
    args.original_dev = str(Path(args.original_dev).resolve())

    if not os.path.exists(args.original_train):
        raise FileNotFoundError(f"original_train 不存在: {args.original_train}")
    if not os.path.exists(args.original_dev):
        raise FileNotFoundError(f"original_dev 不存在: {args.original_dev}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "data_prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "direct_clean_paths",
            "original_train": args.original_train,
            "original_dev": args.original_dev,
            "note": "training scripts will split original_train internally and use original_dev as test"
        }, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["OMP_NUM_THREADS"] = "1"

    start_time = time.time()
    all_rows = []

    print("=" * 80)
    print("开始运行外部领域 4 模型实验（clean 数据集直传版）")
    print(f"Seeds: {args.seeds}")
    print(f"original_train: {args.original_train}")
    print(f"original_dev(test): {args.original_dev}")
    print("=" * 80)

    for cfg in MODEL_CONFIGS:
        print("\n" + "=" * 80)
        print(f"[MODEL] {cfg['model_key']} | {cfg['model_name']}")
        print("=" * 80)

        for seed in args.seeds:
            run_output_dir = output_dir / cfg["model_key"] / f"seed_{seed}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            log_file = run_output_dir / "train_log.txt"
            result_json = run_output_dir / "run_result.json"

            if args.skip_existing and result_json.exists():
                print(f"[SKIP] 已存在结果，跳过: {result_json}")
                with open(result_json, "r", encoding="utf-8") as f:
                    row = json.load(f)
                all_rows.append(row)
                continue

            cmd = build_command(
                cfg=cfg,
                args=args,
                run_output_dir=run_output_dir,
                seed=seed,
            )

            ret, log_text = run_command(cmd, log_file=log_file, env=env)
            metrics = extract_metrics(log_text)

            row = {
                "model_key": cfg["model_key"],
                "family": cfg["family"],
                "model_name": cfg["model_name"],
                "seed": seed,
                "return_code": ret,
                "best_dev_f1": metrics["best_dev_f1"],
                "best_dev_epoch": metrics["best_dev_epoch"],
                "internal_test_acc": metrics["internal_test_acc"],
                "internal_test_f1": metrics["internal_test_f1"],
                "run_dir": str(run_output_dir),
            }

            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

            all_rows.append(row)

            if ret != 0:
                print(f"[WARN] 运行退出码非 0: {ret}")
            else:
                print(f"[OK] 完成: {cfg['model_key']} | seed={seed}")

    all_run_csv = output_dir / "all_run_results.csv"
    save_csv(
        all_run_csv,
        all_rows,
        fieldnames=[
            "model_key", "family", "model_name", "seed", "return_code",
            "best_dev_f1", "best_dev_epoch", "internal_test_acc", "internal_test_f1", "run_dir"
        ],
    )

    summary_rows = []
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["model_key"], []).append(row)

    for model_key, rows in grouped.items():
        test_f1s = [r["internal_test_f1"] for r in rows]
        test_accs = [r["internal_test_acc"] for r in rows]
        dev_f1s = [r["best_dev_f1"] for r in rows]

        summary_rows.append({
            "model_key": model_key,
            "n_runs": len(rows),
            "test_f1_mean": safe_mean(test_f1s),
            "test_f1_std": safe_std(test_f1s),
            "test_acc_mean": safe_mean(test_accs),
            "test_acc_std": safe_std(test_accs),
            "best_dev_f1_mean": safe_mean(dev_f1s),
            "best_dev_f1_std": safe_std(dev_f1s),
            "paper_test_f1": fmt_mean_std(test_f1s),
            "paper_test_acc": fmt_mean_std(test_accs),
            "paper_best_dev_f1": fmt_mean_std(dev_f1s),
        })

    summary_rows.sort(key=lambda x: x["model_key"])

    summary_csv = output_dir / "summary_by_model.csv"
    save_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "model_key", "n_runs",
            "test_f1_mean", "test_f1_std",
            "test_acc_mean", "test_acc_std",
            "best_dev_f1_mean", "best_dev_f1_std",
            "paper_test_f1", "paper_test_acc", "paper_best_dev_f1",
        ],
    )

    paper_f1_rows = []
    paper_acc_rows = []
    for row in summary_rows:
        paper_f1_rows.append({
            "Model": row["model_key"],
            "External_Test_F1 (Mean±Std)": row["paper_test_f1"],
        })
        paper_acc_rows.append({
            "Model": row["model_key"],
            "External_Test_Acc (Mean±Std)": row["paper_test_acc"],
        })

    paper_f1_csv = output_dir / "paper_table_test_f1.csv"
    paper_acc_csv = output_dir / "paper_table_test_acc.csv"

    save_csv(paper_f1_csv, paper_f1_rows, ["Model", "External_Test_F1 (Mean±Std)"])
    save_csv(paper_acc_csv, paper_acc_rows, ["Model", "External_Test_Acc (Mean±Std)"])

    final_obj = {
        "config": vars(args),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "all_runs_file": str(all_run_csv),
        "summary_file": str(summary_csv),
        "paper_table_test_f1": str(paper_f1_csv),
        "paper_table_test_acc": str(paper_acc_csv),
        "summary_rows": summary_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] 全部实验完成")
    print(f"[SAVE] {all_run_csv}")
    print(f"[SAVE] {summary_csv}")
    print(f"[SAVE] {paper_f1_csv}")
    print(f"[SAVE] {paper_acc_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
