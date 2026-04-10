#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import argparse
from pathlib import Path


def _read_text(path: Path) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _find_last_float(patterns, text):
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.MULTILINE)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                last = last[-1]
            try:
                return float(last)
            except Exception:
                pass
    return None


def _find_last_pair(patterns, text):
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.MULTILINE)
        if matches:
            a, b = matches[-1]
            try:
                return float(a), float(b)
            except Exception:
                pass
    return None, None


def parse_log(log_text: str):
    metrics = {}

    # params
    val = _find_last_float([
        r"模型参数量:\s*([\d\.]+)\s*M",
    ], log_text)
    if val is None:
        total_params = _find_last_float([
            r"模型总参数:\s*([\d,]+)"
        ], log_text)
        if total_params is not None:
            val = total_params / 1e6
    metrics['params_m'] = val

    # timing
    total_train_time_sec = _find_last_float([
        r"总训练耗时:\s*([\d\.]+)\s*s",
    ], log_text)
    avg_epoch_time_sec = _find_last_float([
        r"平均每轮耗时:\s*([\d\.]+)\s*s",
    ], log_text)
    if avg_epoch_time_sec is None:
        epoch_times = re.findall(r"Epoch\s+\d+\s+耗时:\s*([\d\.]+)\s*s", log_text)
        if epoch_times:
            vals = [float(x) for x in epoch_times]
            avg_epoch_time_sec = sum(vals) / len(vals)
    metrics['total_train_time_sec'] = total_train_time_sec
    metrics['avg_epoch_time_sec'] = avg_epoch_time_sec

    # gpu mem
    metrics['peak_gpu_memory_gb'] = _find_last_float([
        r"峰值GPU内存使用:\s*([\d\.]+)\s*GB",
        r"Peak GPU memory:\s*([\d\.]+)\s*GB",
    ], log_text)

    # inference
    metrics['inference_ms_per_sample'] = _find_last_float([
        r"推理时延:\s*([\d\.]+)\s*ms/sample",
        r"Inference latency:\s*([\d\.]+)\s*ms/sample",
    ], log_text)
    metrics['inference_samples_per_sec'] = _find_last_float([
        r"推理吞吐:\s*([\d\.]+)\s*samples/s",
        r"Inference throughput:\s*([\d\.]+)\s*samples/s",
    ], log_text)

    # test metrics
    acc, f1 = _find_last_pair([
        r"测试集\s*-\s*Acc:\s*([\d\.]+),\s*F1:\s*([\d\.]+)",
        r"测试集准确率[:：]\s*([\d\.]+)[，,]\s*F1分数[:：]\s*([\d\.]+)",
        r"test_acc=([\d\.]+)\s+test_f1=([\d\.]+)",
        r"internal_test_acc=([\d\.]+)\s+internal_test_f1=([\d\.]+)",
    ], log_text)
    metrics['test_acc'] = acc
    metrics['test_f1'] = f1

    # best dev f1
    best_dev = _find_last_float([
        r"最佳模型在第\s*\d+\s*轮[，,]\s*验证集F1[:：]\s*([\d\.]+)",
        r"最佳验证\s*F1=([\d\.]+)[，,]\s*出现于\s*epoch\s*\d+",
        r"更新最佳模型[:：]\s*epoch=\d+,\s*dev_f1=([\d\.]+)",
    ], log_text)
    metrics['best_dev_f1'] = best_dev

    return metrics


def should_write_json(existing_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    return not existing_path.exists()


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def scan_runs(root_dir: Path):
    rows = []
    for log_path in root_dir.rglob('train_log.txt'):
        run_dir = log_path.parent
        rel = run_dir.relative_to(root_dir)
        parts = rel.parts
        model_key = parts[0] if len(parts) >= 1 else ''
        seed = None
        if len(parts) >= 2 and parts[1].startswith('seed_'):
            try:
                seed = int(parts[1].split('_', 1)[1])
            except Exception:
                seed = parts[1]

        text = _read_text(log_path)
        metrics = parse_log(text)
        row = {
            'model_key': model_key,
            'seed': seed,
            'run_dir': str(run_dir),
            'log_file': str(log_path),
            **metrics,
        }
        rows.append(row)
    rows.sort(key=lambda x: (str(x['model_key']), str(x['seed'])))
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        'model_key', 'seed', 'run_dir', 'log_file',
        'params_m', 'total_train_time_sec', 'avg_epoch_time_sec',
        'peak_gpu_memory_gb', 'inference_ms_per_sample', 'inference_samples_per_sec',
        'test_acc', 'test_f1', 'best_dev_f1'
    ]
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Recover efficiency metrics from train_log.txt files.')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--write_json', action='store_true', help='Write efficiency_metrics.json into each run dir')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing efficiency_metrics.json if present')
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f'root_dir 不存在: {root_dir}')

    rows = scan_runs(root_dir)
    if not rows:
        print(f'[WARN] 在 {root_dir} 下没有找到 train_log.txt')
        return

    if args.write_json:
        for row in rows:
            out_path = Path(row['run_dir']) / 'efficiency_metrics.json'
            if should_write_json(out_path, args.overwrite):
                obj = {
                    'model_key': row['model_key'],
                    'seed': row['seed'],
                    'params_m': row['params_m'],
                    'total_train_time_sec': row['total_train_time_sec'],
                    'avg_epoch_time_sec': row['avg_epoch_time_sec'],
                    'peak_gpu_memory_gb': row['peak_gpu_memory_gb'],
                    'inference_ms_per_sample': row['inference_ms_per_sample'],
                    'inference_samples_per_sec': row['inference_samples_per_sec'],
                    'test_acc': row['test_acc'],
                    'test_f1': row['test_f1'],
                    'best_dev_f1': row['best_dev_f1'],
                }
                save_json(out_path, obj)

    csv_path = root_dir / 'recovered_efficiency_summary.csv'
    write_csv(csv_path, rows)
    print(f'[DONE] 共恢复 {len(rows)} 个 run')
    print(f'[SAVE] {csv_path}')
    if args.write_json:
        print('[SAVE] 已为各 run 目录补写 efficiency_metrics.json')


if __name__ == '__main__':
    main()
