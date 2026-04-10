# llm_zero_shot_qwen.py
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import csv
import argparse
import logging
import sys
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# =========================
# Data loading
# =========================
def load_jsonl_dataset(json_path: str) -> List[Dict[str, Any]]:
    assert os.path.exists(json_path), f"数据集不存在: {json_path}"
    data = []

    with open(json_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning(f"第 {lineno} 行 JSON 解析失败，跳过。错误: {e}")
                continue

            if "abst" not in obj or "keyword" not in obj:
                logger.warning(f"第 {lineno} 行缺少 abst/keyword 字段，跳过。")
                continue

            abst = str(obj["abst"]).strip()
            if not abst:
                logger.warning(f"第 {lineno} 行 abst 为空，跳过。")
                continue

            kw = obj["keyword"]
            if isinstance(kw, list):
                kw_list = [str(x).strip() for x in kw if str(x).strip()]
                kw_text = "；".join(kw_list)
            else:
                kw_text = str(kw).strip()

            label = obj.get("label", None)
            if label is not None:
                try:
                    if isinstance(label, (int, float)):
                        label = int(label)
                    else:
                        s = str(label).strip()
                        if s.isdigit():
                            label = int(s)
                        elif s.lower() in ["true", "yes", "1"]:
                            label = 1
                        else:
                            label = 0
                    label = 1 if label > 0 else 0
                except Exception:
                    label = None

            data.append({
                "id": obj.get("id", f"sample_{lineno}"),
                "abst": abst,
                "keyword": kw_text,
                "label": label
            })

    if len(data) == 0:
        raise RuntimeError(f"{json_path} 中没有有效样本。")

    logger.info(f"成功加载 {len(data)} 条样本: {json_path}")
    return data


# =========================
# Prompt
# =========================
def build_zero_shot_prompt(abst: str, keywords: str) -> str:
    return f"""你是一个科技文献关键词质量评估助手。

任务：根据给定的论文摘要，判断候选关键词集合中的关键词是否全部真实、准确地反映了摘要内容。

判定规则：
1. 如果候选关键词集合中的所有关键词都与摘要语义一致，并且可以视为真实关键词，则输出“是”。
2. 只要其中存在一个不准确、不相关，或明显不应作为该摘要关键词的词，则输出“否”。
3. 只输出一个字：“是”或“否”，不要输出解释。

摘要：
{abst}

候选关键词：
{keywords}

答案："""


# =========================
# Output parsing
# =========================
def parse_prediction(output_text: str):
    """
    返回:
        pred: 1 表示“是”, 0 表示“否”
        valid: 是否成功解析
    """
    text = output_text.strip()

    # 优先看开头
    if text.startswith("是"):
        return 1, True
    if text.startswith("否"):
        return 0, True

    # 再做兜底解析
    if ("是" in text) and ("否" not in text):
        return 1, True
    if "否" in text:
        return 0, True

    return 0, False


# =========================
# Inference
# =========================
@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_input_length: int = 4096,
    max_new_tokens: int = 4
) -> str:
    messages = [
        {"role": "system", "content": "你是一个严谨的科技文献关键词质量评估助手。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


# =========================
# Save predictions
# =========================
def save_predictions_csv(save_path: str, rows: List[Dict[str, Any]]):
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "label", "pred", "valid_parse",
                "raw_output", "keyword", "abst"
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"预测结果已保存到: {save_path}")
    
def save_result_json(save_path: str, result_dict: dict):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"结果指标已保存到: {save_path}")

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="/root/autodl-tmp/MVPproject/output_attention_optimized_cijuduan/split_data/test.json",
        help="测试集路径。你也可以直接填 original_dev 的路径。"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./llm_zero_shot_predictions.csv"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
        help="prompt 最大输入 token 长度"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="./llm_zero_shot_result.json"
    )
    args = parser.parse_args()

    logger.info(f"加载测试集: {args.test_path}")
    test_data = load_jsonl_dataset(args.test_path)

    logger.info(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    device = model.device
    logger.info(f"模型设备: {device}")

    y_true, y_pred = [], []
    save_rows = []
    invalid_count = 0

    for idx, sample in enumerate(test_data, 1):
        prompt = build_zero_shot_prompt(sample["abst"], sample["keyword"])
        raw_output = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens
        )

        pred, valid = parse_prediction(raw_output)
        if not valid:
            invalid_count += 1

        label = sample["label"]

        if label is not None:
            y_true.append(label)
            y_pred.append(pred)

        save_rows.append({
            "id": sample["id"],
            "label": label,
            "pred": pred,
            "valid_parse": int(valid),
            "raw_output": raw_output,
            "keyword": sample["keyword"],
            "abst": sample["abst"]
        })

        if idx % 50 == 0:
            logger.info(f"已完成 {idx}/{len(test_data)} 条")

    save_predictions_csv(args.output_csv, save_rows)

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        cls_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    
        logger.info("========== Zero-shot 结果 ==========")
        logger.info(f"Accuracy  : {acc:.4f}")
        logger.info(f"Macro-F1  : {macro_f1:.4f}")
        logger.info(f"无法规范解析的输出条数: {invalid_count}")
        logger.info("分类报告:")
        logger.info("\n" + classification_report(y_true, y_pred, digits=4))
    
        result_dict = {
            "setting": "zero-shot",
            "model_name": args.model_name,
            "test_path": args.test_path,
            "num_samples": len(y_true),
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6),
            "invalid_parse_count": int(invalid_count),
            "classification_report": cls_report
        }
    
        save_result_json(args.result_json, result_dict)
    else:
        logger.warning("测试集没有可用 label，无法计算指标。")


if __name__ == "__main__":
    main()
