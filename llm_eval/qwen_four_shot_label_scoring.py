# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import csv
import argparse
import logging
import sys
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, classification_report


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def save_json(save_path: str, obj: dict):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 JSON: {save_path}")


def save_predictions_csv(save_path: str, rows: List[Dict[str, Any]]):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "label", "pred",
                "score_yes", "score_no",
                "keyword", "abst"
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"预测结果已保存到: {save_path}")


def normalize_label(label):
    if label is None:
        return None
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
        return 1 if label > 0 else 0
    except Exception:
        return None


def normalize_keywords(kw) -> str:
    if isinstance(kw, list):
        return "；".join([str(x).strip() for x in kw if str(x).strip()])
    return str(kw).strip()


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

            data.append({
                "id": obj.get("id", f"sample_{lineno}"),
                "abst": abst,
                "keyword": normalize_keywords(obj["keyword"]),
                "label": normalize_label(obj.get("label", None))
            })

    if len(data) == 0:
        raise RuntimeError(f"{json_path} 中没有有效样本。")

    logger.info(f"成功加载 {len(data)} 条样本: {json_path}")
    return data


def example_score(item: Dict[str, Any], target_abst_len: int = 180, target_kw_count: int = 4) -> float:
    abst_len = len(item["abst"])
    kw_count = len([x for x in item["keyword"].split("；") if x.strip()])
    return abs(abst_len - target_abst_len) + 10.0 * abs(kw_count - target_kw_count)


def select_four_shot_examples(train_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    固定选 4 个示例：
    2 个正样本 + 2 个负样本
    选中等长度、关键词数量适中的样本，避免 prompt 过长
    """
    pos = [x for x in train_data if x["label"] == 1]
    neg = [x for x in train_data if x["label"] == 0]

    if len(pos) < 2 or len(neg) < 2:
        raise RuntimeError("训练集中正负样本不足，无法构造 4-shot 示例。")

    pos = sorted(pos, key=example_score)
    neg = sorted(neg, key=example_score)

    selected = [pos[0], neg[0], pos[1], neg[1]]

    logger.info("已固定选取 4-shot 示例：2 个正样本 + 2 个负样本")
    for i, ex in enumerate(selected, 1):
        logger.info(
            f"[示例{i}] label={ex['label']} | id={ex['id']} | "
            f"abst_len={len(ex['abst'])} | keywords={ex['keyword']}"
        )

    return selected


def build_four_shot_prompt(examples: List[Dict[str, Any]], abst: str, keywords: str) -> str:
    prefix = """你是一个科技文献关键词质量评估助手。

任务：根据给定的论文摘要，判断候选关键词集合中的关键词是否全部真实、准确地反映了摘要内容。

判定规则：
1. 如果候选关键词集合中的所有关键词都与摘要语义一致，并且可以视为真实关键词，则答案为“是”。
2. 只要其中存在一个不准确、不相关，或明显不应作为该摘要关键词的词，则答案为“否”。
3. 只需判断“是”或“否”。

"""

    demos = []
    for i, ex in enumerate(examples, 1):
        ans = "是" if ex["label"] == 1 else "否"
        demos.append(
            f"示例{i}：\n"
            f"摘要：{ex['abst']}\n"
            f"候选关键词：{ex['keyword']}\n"
            f"答案：{ans}\n"
        )

    suffix = f"""
现在请判断下面样本：

摘要：
{abst}

候选关键词：
{keywords}

答案："""
    return prefix + "\n".join(demos) + suffix


@torch.no_grad()
def score_label(
    model,
    tokenizer,
    prompt: str,
    label_text: str,
    device,
    max_input_length: int = 4096
) -> float:
    """
    计算条件分数:
        score(label | prompt) = sum log p(label_tokens | prompt)
    """
    messages = [
        {"role": "system", "content": "你是一个严谨的科技文献关键词质量评估助手。"},
        {"role": "user", "content": prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    if len(label_ids) == 0:
        raise ValueError(f"标签 `{label_text}` 被 tokenizer 编码为空。")

    # 给 label 预留长度，避免截断后越界
    reserve = len(label_ids) + 1
    prompt_max_len = max(8, max_input_length - reserve)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_max_len
    )["input_ids"]

    input_ids = torch.tensor([prompt_ids + label_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    total_score = 0.0

    for i, token_id in enumerate(label_ids):
        pos = prompt_len + i - 1
        token_logprob = log_probs[0, pos, token_id].item()
        total_score += token_logprob

    return total_score


def predict_one(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    abst: str,
    keywords: str,
    device,
    max_input_length: int = 4096
) -> Tuple[int, float, float]:
    prompt = build_four_shot_prompt(examples, abst, keywords)

    score_yes = score_label(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        label_text="是",
        device=device,
        max_input_length=max_input_length
    )
    score_no = score_label(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        label_text="否",
        device=device,
        max_input_length=max_input_length
    )

    pred = 1 if score_yes > score_no else 0
    return pred, score_yes, score_no


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="/root/autodl-tmp/MVPproject/output_attention_optimized_cijuduan/split_data/train.json",
        help="4-shot 示例来源训练集路径"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/root/autodl-tmp/MVPproject/output_attention_optimized_cijuduan/split_data/test.json",
        help="测试集路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./llm_four_shot_label_scoring_predictions.csv"
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="./llm_four_shot_label_scoring_result.json"
    )
    parser.add_argument(
        "--examples_json",
        type=str,
        default="./llm_four_shot_examples.json",
        help="保存选中的 4-shot 示例"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096
    )
    args = parser.parse_args()

    logger.info(f"加载训练集: {args.train_path}")
    train_data = load_jsonl_dataset(args.train_path)

    logger.info(f"加载测试集: {args.test_path}")
    test_data = load_jsonl_dataset(args.test_path)

    few_shot_examples = select_four_shot_examples(train_data)
    save_json(args.examples_json, {"few_shot_examples": few_shot_examples})

    logger.info(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = model.device
    logger.info(f"模型设备: {device}")

    y_true, y_pred = [], []
    save_rows = []

    for idx, sample in enumerate(test_data, 1):
        pred, score_yes, score_no = predict_one(
            model=model,
            tokenizer=tokenizer,
            examples=few_shot_examples,
            abst=sample["abst"],
            keywords=sample["keyword"],
            device=device,
            max_input_length=args.max_input_length
        )

        if sample["label"] is not None:
            y_true.append(sample["label"])
            y_pred.append(pred)

        save_rows.append({
            "id": sample["id"],
            "label": sample["label"],
            "pred": pred,
            "score_yes": round(score_yes, 6),
            "score_no": round(score_no, 6),
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

        logger.info("========== 4-shot Label Scoring 结果 ==========")
        logger.info(f"Accuracy  : {acc:.4f}")
        logger.info(f"Macro-F1  : {macro_f1:.4f}")
        logger.info("分类报告:")
        logger.info("\n" + classification_report(y_true, y_pred, digits=4))

        result_dict = {
            "setting": "four-shot-label-scoring",
            "model_name": args.model_name,
            "train_path": args.train_path,
            "test_path": args.test_path,
            "num_samples": len(y_true),
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6),
            "classification_report": cls_report
        }
        save_json(args.result_json, result_dict)
    else:
        logger.warning("测试集没有可用 label，无法计算指标。")


if __name__ == "__main__":
    main()
