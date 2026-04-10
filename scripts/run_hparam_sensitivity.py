#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
关键超参数敏感性实验总控脚本

默认用于 EMG 模型，建议优先跑：
1) Pdel 敏感性
2) 非术语抑制常数 delta(non_term_penalty) 敏感性

示例：
python run_hparam_sensitivity.py \
  --script /mnt/data/finallv2_sensitivity.py \
  --original_train /root/autodl-tmp/MVPproject/train.json \
  --original_dev /root/autodl-tmp/MVPproject/dev.json \
  --output_dir /root/autodl-tmp/MVPproject/output_sensitivity_pdel \
  --param_name p_del \
  --param_values 0 0.1 0.3 0.5 0.7 \
  --seeds 42 52 85 \
  --model_name bert-base-chinese \
  --model_version v2
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, pstdev


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
        r"最佳模型在第\s*(\d+)\s*轮，验证集F1[:：]\s*(\d+\.\d+)",
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
        r"测试集准确率[:：]\s*(\d+\.\d+)[，,]\s*F1分数[:：]\s*(\d+\.\d+)",
    ], log_text)

    test_acc, test_f1 = None, None
    if test_pair is not None:
        test_acc = float(test_pair[0])
        test_f1 = float(test_pair[1])

    return {
        "best_dev_f1": best_dev_f1,
        "best_dev_epoch": best_dev_epoch,
        "test_acc": test_acc,
        "test_f1": test_f1,
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


def save_csv(path: Path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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




def parse_bool_like(x: str):
    s = str(x).strip().lower()
    if s in {"1", "true", "on", "yes", "y", "enable", "enabled"}:
        return True
    if s in {"0", "false", "off", "no", "n", "disable", "disabled"}:
        return False
    raise ValueError(f"无法解析布尔型参数取值: {x}")


def normalize_param_value(param_name: str, raw_value: str):
    if param_name in {"p_del", "non_term_penalty", "term_bias_scale"}:
        return float(raw_value)
    if param_name == "progressive_schedule":
        return parse_bool_like(raw_value)
    raise ValueError(f"未知参数名: {param_name}")


def param_value_to_dirname(param_name: str, value):
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value)
    return text.replace("/", "_")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True, help="finallv2_sensitivity.py 路径")
    parser.add_argument("--original_train", type=str, required=True)
    parser.add_argument("--original_dev", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--model_version", type=str, default="v2", choices=["base", "v1", "v2"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--param_name", type=str, required=True,
                        choices=["p_del", "non_term_penalty", "term_bias_scale", "progressive_schedule"])
    parser.add_argument("--param_values", type=str, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 85])
    parser.add_argument("--disable_keyword_shuffle", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    normalized_param_values = [normalize_param_value(args.param_name, x) for x in args.param_values]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["OMP_NUM_THREADS"] = "1"

    all_rows = []
    start_time = time.time()

    print("=" * 80)
    print("开始运行关键超参数敏感性实验")
    print(f"param_name: {args.param_name}")
    print(f"param_values(raw): {args.param_values}")
    print(f"param_values(norm): {normalized_param_values}")
    print(f"seeds: {args.seeds}")
    print("=" * 80)

    for param_value in normalized_param_values:
        print("\n" + "=" * 80)
        print(f"[{args.param_name}] = {param_value}")
        print("=" * 80)

        for seed in args.seeds:
            run_dir = output_dir / f"{args.param_name}_{param_value_to_dirname(args.param_name, param_value)}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = run_dir / "train_log.txt"
            result_json = run_dir / "run_result.json"

            if args.skip_existing and result_json.exists():
                with open(result_json, "r", encoding="utf-8") as f:
                    row = json.load(f)
                all_rows.append(row)
                print(f"[SKIP] {result_json}")
                continue

            cmd = [
                "python", args.script,
                "--original_train", args.original_train,
                "--original_dev", args.original_dev,
                "--output_dir", str(run_dir),
                "--seed", str(seed),
                "--model_name", args.model_name,
                "--model_version", args.model_version,
                "--batch_size", str(args.batch_size),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--max_len", str(args.max_len),
            ]
            if args.param_name == "progressive_schedule":
                if not param_value:
                    cmd.append("--disable_progressive_schedule")
            else:
                cmd.extend([f"--{args.param_name}", str(param_value)])

            if args.disable_keyword_shuffle:
                cmd.append("--disable_keyword_shuffle")

            ret, log_text = run_command(cmd, log_file, env=env)
            metrics = extract_metrics(log_text)

            row = {
                "param_name": args.param_name,
                "param_value": param_value,
                "seed": seed,
                "return_code": ret,
                "best_dev_f1": metrics["best_dev_f1"],
                "best_dev_epoch": metrics["best_dev_epoch"],
                "test_acc": metrics["test_acc"],
                "test_f1": metrics["test_f1"],
                "run_dir": str(run_dir),
            }

            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            all_rows.append(row)

    all_run_csv = output_dir / "all_run_results.csv"
    save_csv(
        all_run_csv,
        all_rows,
        ["param_name", "param_value", "seed", "return_code", "best_dev_f1", "best_dev_epoch", "test_acc", "test_f1", "run_dir"]
    )

    summary_rows = []
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["param_value"], []).append(row)

    def _sort_key(item):
        key = item[0]
        if isinstance(key, bool):
            return (0, int(key))
        try:
            return (1, float(key))
        except Exception:
            return (2, str(key))

    for param_value, rows in sorted(grouped.items(), key=_sort_key):
        f1s = [r["test_f1"] for r in rows]
        accs = [r["test_acc"] for r in rows]
        devs = [r["best_dev_f1"] for r in rows]
        summary_rows.append({
            "param_name": args.param_name,
            "param_value": param_value,
            "n_runs": len(rows),
            "test_f1_mean": safe_mean(f1s),
            "test_f1_std": safe_std(f1s),
            "test_acc_mean": safe_mean(accs),
            "test_acc_std": safe_std(accs),
            "best_dev_f1_mean": safe_mean(devs),
            "best_dev_f1_std": safe_std(devs),
            "paper_test_f1": fmt_mean_std(f1s),
            "paper_test_acc": fmt_mean_std(accs),
            "paper_best_dev_f1": fmt_mean_std(devs),
        })

    summary_csv = output_dir / "summary_by_value.csv"
    save_csv(
        summary_csv,
        summary_rows,
        [
            "param_name", "param_value", "n_runs",
            "test_f1_mean", "test_f1_std",
            "test_acc_mean", "test_acc_std",
            "best_dev_f1_mean", "best_dev_f1_std",
            "paper_test_f1", "paper_test_acc", "paper_best_dev_f1",
        ]
    )

    final_obj = {
        "config": vars(args),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "all_runs_file": str(all_run_csv),
        "summary_file": str(summary_csv),
        "summary_rows": summary_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] 参数敏感性实验完成")
    print(f"[SAVE] {all_run_csv}")
    print(f"[SAVE] {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
