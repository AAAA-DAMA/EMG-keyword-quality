#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fintech-Key-Phrase -> 二分类判别数据集（v3 小修清洗增强版）

在 v2 基础上新增：
1. 更严格的 gold 过滤：
   - 删除更多泛词/上位词
   - 删除明显半截词（如结尾异常、被更长术语覆盖）
   - 删除过短且信息量弱的术语
2. 更严格的 negative 过滤：
   - 增加黑名单，避免明显离谱的伪术语进入负样本
   - 优先选择“同后缀/同形态/近邻文本”术语
3. 最终输出前再做一轮样本级清洗：
   - 保证 keyword 去重
   - 删除空 keyword 样本
   - 删除负样本里与正样本完全相同的情况
"""

import os
import re
import json
import random
import argparse
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"解析 JSONL 失败: {path}, line={line_no}, error={e}")
    return items


def write_jsonl(path: str, items: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


PURE_NUM_RE = re.compile(r"^[\d\.\-%+/]+$")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
ALNUM_RE = re.compile(r"^[A-Za-z0-9\-\+_/\.]+$")
SPLIT_PUNCT_RE = re.compile(r"[，。；、：:（）()《》“”\"'【】\[\],;]")

# gold/neg 通用黑名单：明显过泛、对任务帮助很弱
GENERIC_BAD_PHRASES = {
    "国际水平", "产业链", "公司", "产品", "业务", "工作", "规划", "部署", "概述",
    "发展", "应用", "系统", "技术", "结构", "分析", "平台", "服务", "建设",
    "水平", "能力", "行业", "市场", "数据", "方案", "项目", "规划和部署",
    "产品研发", "生产制造", "国内外", "战略投资", "合作", "物流运输",
    "金融", "支付", "材料", "能源", "环保", "设备", "电子", "机械",
    "标准", "控制", "开关", "变压器", "芯片", "药", "汽车", "工业",
    "建筑", "医疗", "办公", "终端", "发射设备", "电力", "虚拟", "增强现实",
    "数字时代", "装备", "民航", "政府", "交通", "生化", "血浆", "药品",
}

# 负样本黑名单：你刚才样本里明显跑偏或风格太跳的伪词
NEG_BAD_PHRASES = {
    "国土", "航空", "珠宝首饰", "人参", "主动笔", "百士欣", "配送药品",
    "快递", "农残", "半夏", "国土", "钛白粉", "VQFN", "Notebook",
    "软件平台", "自动化", "智能", "冷藏", "办公", "民航", "政府信息化",
    "机械设备", "仪器一体", "仿真", "航空", "国土", "通信", "驱动",
}

# 当这些短词被更长术语覆盖时，优先删除
GENERIC_SUBPHRASES = {
    "支付", "金融", "材料", "环保", "能源", "设备", "电子", "机械",
    "系统", "服务", "技术", "控制", "平台", "产品", "开关", "变压器",
    "芯片", "药", "工业", "汽车", "纤维", "面料", "高精度", "高性能",
    "建筑", "医疗", "电力", "交通", "政府", "办公", "装备"
}

KNOWN_SUFFIXES = [
    "注射液", "注射用", "疫苗", "芯片", "电缆", "电池", "逆变器", "发生器", "变压器",
    "开关柜", "断路器", "控制器", "系统", "平台", "设备", "材料", "药物", "药品",
    "药", "试验", "研究", "服务", "组件", "切片", "纤维", "地板", "模具", "机",
    "柜", "管", "板", "胶囊", "巴布膏", "电站", "物联网", "互联网", "终端",
    "空调", "压缩机", "热泵", "刀具", "刀片", "站", "中心", "电机", "总成",
]

SUSPICIOUS_ENDINGS = {
    "中", "及", "与", "和", "或", "等", "类", "原", "化", "销", "售", "电",
}

def normalize_phrase(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\u3000", " ")
    s = s.strip(" \t\r\n\"'“”‘’（）()[]【】")
    s = re.sub(r"\s+", "", s)
    return s.strip()


def phrase_type(s: str) -> str:
    if ALNUM_RE.match(s):
        return "alnum"
    return "mixed"


def phrase_suffix(s: str) -> str:
    for suf in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
        if s.endswith(suf) and len(s) > len(suf):
            return suf
    return ""


def is_suspicious_truncation(s: str) -> bool:
    s = normalize_phrase(s)
    if not s:
        return True
    if len(s) <= 2 and s in GENERIC_BAD_PHRASES:
        return True
    if len(s) >= 2 and s[-1] in SUSPICIOUS_ENDINGS and phrase_suffix(s) == "":
        # 像“丙泊酚中”“发射设备”这类保守拦一下
        if len(s) <= 5:
            return True
    return False


def is_bad_phrase_basic(s: str) -> bool:
    s = normalize_phrase(s)
    if not s:
        return True
    if len(s) <= 1:
        return True
    if len(s) > 30:
        return True
    if PURE_NUM_RE.match(s):
        return True
    if PUNCT_ONLY_RE.match(s):
        return True
    if SPLIT_PUNCT_RE.search(s):
        return True
    if s in GENERIC_BAD_PHRASES:
        return True
    if is_suspicious_truncation(s):
        return True
    return False


def extract_spans_from_entity_dict(ent_dict: Dict, text: str) -> List[str]:
    phrases = []
    for key, spans in ent_dict.items():
        key_norm = normalize_phrase(key)
        extracted_any = False
        if isinstance(spans, list):
            for sp in spans:
                if not (isinstance(sp, list) or isinstance(sp, tuple)) or len(sp) != 2:
                    continue
                try:
                    start, end = int(sp[0]), int(sp[1])
                except Exception:
                    continue
                if start < 0 or end < start or end >= len(text):
                    continue
                frag = normalize_phrase(text[start:end + 1])
                if frag:
                    phrases.append(frag)
                    extracted_any = True
        if not extracted_any and key_norm:
            phrases.append(key_norm)
    return phrases


def remove_subphrase_noise(phrases: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for p in phrases:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    keep = []
    for p in uniq:
        covered_by_longer = False
        for q in uniq:
            if p == q:
                continue
            if len(q) <= len(p):
                continue
            if p in q and (p in GENERIC_SUBPHRASES or len(p) <= 2):
                covered_by_longer = True
                break
        if not covered_by_longer:
            keep.append(p)
    return keep


def extract_gold_phrases(item: Dict) -> List[str]:
    label = item.get("label", {})
    ent_dict = label.get("financial_entity", {}) if isinstance(label, dict) else {}
    text = str(item.get("text", ""))

    raw_phrases = extract_spans_from_entity_dict(ent_dict, text)

    phrases = []
    for p in raw_phrases:
        p = normalize_phrase(p)
        if is_bad_phrase_basic(p):
            continue
        phrases.append(p)

    phrases = remove_subphrase_noise(phrases)

    # 去掉仍然过于泛的 2 字短词（但保留典型缩写词）
    filtered = []
    for p in phrases:
        if len(p) == 2 and phrase_type(p) != "alnum" and phrase_suffix(p) == "" and p in GENERIC_BAD_PHRASES:
            continue
        filtered.append(p)

    seen = set()
    cleaned = []
    for p in filtered:
        if p not in seen:
            seen.add(p)
            cleaned.append(p)
    return cleaned


def convert_raw_item(raw: Dict, idx: int) -> Dict:
    return {
        "raw_id": idx,
        "text": str(raw.get("text", "")).strip(),
        "gold_phrases": extract_gold_phrases(raw),
    }


def filter_raw_items(raw_items: List[Dict], min_gold_phrases: int = 2) -> List[Dict]:
    kept = []
    for i, raw in enumerate(raw_items):
        item = convert_raw_item(raw, i)
        if not item["text"]:
            continue
        if len(item["gold_phrases"]) < min_gold_phrases:
            continue
        kept.append(item)
    return kept


def build_neighbors(items: List[Dict], neighbor_k: int = 30) -> List[List[int]]:
    texts = [x["text"] for x in items]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=50000,
    )
    X = vectorizer.fit_transform(texts)
    sims = linear_kernel(X, X)

    neighbors = []
    for i in range(len(items)):
        scores = sims[i]
        order = scores.argsort()[::-1]
        order = [j for j in order if j != i]
        neighbors.append(order[:neighbor_k])
    return neighbors


def length_bucket(s: str) -> int:
    n = len(s)
    if n <= 2:
        return 2
    if n == 3:
        return 3
    if n == 4:
        return 4
    if n <= 6:
        return 5
    return 6


def build_bucketed_global_pool(items: List[Dict]) -> Dict[int, List[str]]:
    buckets = defaultdict(list)
    seen = set()
    for item in items:
        for p in item["gold_phrases"]:
            if p in seen:
                continue
            seen.add(p)
            buckets[length_bucket(p)].append(p)
    return buckets


def candidate_score(target: str, cand: str) -> Tuple[int, int, int, int]:
    target_suf = phrase_suffix(target)
    cand_suf = phrase_suffix(cand)
    same_suffix = int(target_suf != "" and target_suf == cand_suf)
    same_type = int(phrase_type(target) == phrase_type(cand))
    overlap = len(set(target) & set(cand))
    len_close = -abs(len(cand) - len(target))
    return (same_suffix, same_type, overlap, len_close)


def pick_best_candidate(target: str, candidates: List[str], rng: random.Random) -> str:
    if not candidates:
        return target
    scored = [(candidate_score(target, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:min(10, len(scored))]]
    return rng.choice(top)


def valid_negative_candidate(p: str, current_gold_set: set) -> bool:
    p = normalize_phrase(p)
    if not p:
        return False
    if p in current_gold_set:
        return False
    if is_bad_phrase_basic(p):
        return False
    if p in NEG_BAD_PHRASES:
        return False
    return True


def sample_hard_negative_phrase(
    item_idx: int,
    current_gold: List[str],
    items: List[Dict],
    neighbors: List[List[int]],
    bucketed_global_pool: Dict[int, List[str]],
    replace_target: str,
    rng: random.Random,
) -> str:
    current_gold_set = set(current_gold)
    target_bucket = length_bucket(replace_target)

    candidates = []
    for nb_idx in neighbors[item_idx]:
        for p in items[nb_idx]["gold_phrases"]:
            if valid_negative_candidate(p, current_gold_set) and length_bucket(p) == target_bucket:
                candidates.append(p)
    if candidates:
        return pick_best_candidate(replace_target, candidates, rng)

    global_candidates = [
        p for p in bucketed_global_pool.get(target_bucket, [])
        if valid_negative_candidate(p, current_gold_set)
    ]
    if global_candidates:
        return pick_best_candidate(replace_target, global_candidates, rng)

    all_pool = []
    for _, lst in bucketed_global_pool.items():
        all_pool.extend(lst)
    all_pool = [p for p in all_pool if valid_negative_candidate(p, current_gold_set)]
    if all_pool:
        return pick_best_candidate(replace_target, all_pool, rng)

    return replace_target


def sample_easy_negative_phrase(
    current_gold: List[str],
    bucketed_global_pool: Dict[int, List[str]],
    replace_target: str,
    rng: random.Random,
) -> str:
    current_gold_set = set(current_gold)
    target_bucket = length_bucket(replace_target)

    candidates = [
        p for p in bucketed_global_pool.get(target_bucket, [])
        if valid_negative_candidate(p, current_gold_set)
    ]
    if candidates:
        return rng.choice(candidates)

    all_pool = []
    for _, lst in bucketed_global_pool.items():
        all_pool.extend(lst)
    all_pool = [p for p in all_pool if valid_negative_candidate(p, current_gold_set)]
    if all_pool:
        return rng.choice(all_pool)

    return replace_target


def dedup_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def make_negative_keywords(
    item_idx: int,
    gold_phrases: List[str],
    items: List[Dict],
    neighbors: List[List[int]],
    bucketed_global_pool: Dict[int, List[str]],
    rng: random.Random,
    hard_negative_ratio: float = 0.85,
    num_replace: int = 1,
) -> List[str]:
    kws = list(gold_phrases)
    if not kws:
        return kws

    replace_indices = list(range(len(kws)))
    rng.shuffle(replace_indices)
    replace_indices = replace_indices[:max(1, min(num_replace, len(kws)))]

    used = set(kws)
    for ridx in replace_indices:
        target = kws[ridx]
        if rng.random() < hard_negative_ratio:
            neg_p = sample_hard_negative_phrase(
                item_idx=item_idx,
                current_gold=kws,
                items=items,
                neighbors=neighbors,
                bucketed_global_pool=bucketed_global_pool,
                replace_target=target,
                rng=rng,
            )
        else:
            neg_p = sample_easy_negative_phrase(
                current_gold=kws,
                bucketed_global_pool=bucketed_global_pool,
                replace_target=target,
                rng=rng,
            )

        guard = 0
        while (neg_p in used or neg_p == target) and guard < 20:
            neg_p = sample_easy_negative_phrase(
                current_gold=list(used),
                bucketed_global_pool=bucketed_global_pool,
                replace_target=target,
                rng=rng,
            )
            guard += 1

        kws[ridx] = neg_p
        used.add(neg_p)

    kws = dedup_keep_order([normalize_phrase(x) for x in kws if not is_bad_phrase_basic(x)])
    return kws


def finalize_examples(examples: List[Dict], shuffle_seed: int) -> List[Dict]:
    """
    样本级最后清洗：
    - keyword 去重
    - 删除空 keyword
    - 删除负样本与其正样本完全同词集合的情况（理论上极少）
    """
    cleaned = []
    for ex in examples:
        kws = dedup_keep_order([normalize_phrase(x) for x in ex["keyword"] if not is_bad_phrase_basic(x)])
        if len(kws) == 0:
            continue
        ex["keyword"] = kws
        cleaned.append(ex)

    rng = random.Random(shuffle_seed)
    rng.shuffle(cleaned)
    return cleaned


def build_binary_dataset_for_split(
    raw_split_items: List[Dict],
    split_name: str,
    rng_seed: int,
    neg_per_pos: int = 1,
    neighbor_k: int = 30,
    hard_negative_ratio: float = 0.85,
    num_replace: int = 1,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(rng_seed)
    neighbors = build_neighbors(raw_split_items, neighbor_k=neighbor_k)
    bucketed_global_pool = build_bucketed_global_pool(raw_split_items)

    examples = []
    phrase_counts = []

    for item_idx, item in enumerate(raw_split_items):
        raw_id = item["raw_id"]
        text = item["text"]
        gold_phrases = dedup_keep_order(item["gold_phrases"])
        if len(gold_phrases) == 0:
            continue

        phrase_counts.append(len(gold_phrases))
        examples.append({
            "id": f"{split_name}_{raw_id}_pos",
            "abst": text,
            "keyword": gold_phrases,
            "label": 1,
        })

        for neg_idx in range(neg_per_pos):
            neg_keywords = make_negative_keywords(
                item_idx=item_idx,
                gold_phrases=gold_phrases,
                items=raw_split_items,
                neighbors=neighbors,
                bucketed_global_pool=bucketed_global_pool,
                rng=rng,
                hard_negative_ratio=hard_negative_ratio,
                num_replace=num_replace,
            )
            if neg_keywords == gold_phrases:
                continue
            examples.append({
                "id": f"{split_name}_{raw_id}_neg{neg_idx+1}",
                "abst": text,
                "keyword": neg_keywords,
                "label": 0,
            })

    examples = finalize_examples(examples, shuffle_seed=rng_seed + 999)

    stats = {
        "split": split_name,
        "raw_items": len(raw_split_items),
        "positive_examples": sum(1 for x in examples if x["label"] == 1),
        "negative_examples": sum(1 for x in examples if x["label"] == 0),
        "final_examples": len(examples),
        "label_counter": dict(Counter([x["label"] for x in examples])),
        "avg_gold_phrase_num": round(sum(phrase_counts) / len(phrase_counts), 4) if phrase_counts else 0.0,
    }
    return examples, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_train", type=str, required=True)
    parser.add_argument("--raw_test", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_gold_phrases", type=int, default=2)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--neighbor_k", type=int, default=30)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.85)
    parser.add_argument("--num_replace", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_train = read_jsonl(args.raw_train)
    raw_test = read_jsonl(args.raw_test)

    train_items = filter_raw_items(raw_train, min_gold_phrases=args.min_gold_phrases)
    test_items = filter_raw_items(raw_test, min_gold_phrases=args.min_gold_phrases)

    print(f"[INFO] 原始 train 读取: {len(raw_train)}")
    print(f"[INFO] 原始 test 读取:  {len(raw_test)}")
    print(f"[INFO] 清洗后 train 可用: {len(train_items)}")
    print(f"[INFO] 清洗后 test 可用:  {len(test_items)}")

    if len(train_items) < 100:
        raise ValueError("清洗后训练样本过少，请检查数据格式或放宽 min_gold_phrases。")

    indices = list(range(len(train_items)))
    train_idx, dev_idx = train_test_split(
        indices,
        test_size=args.dev_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    train_raw_split = [train_items[i] for i in train_idx]
    dev_raw_split = [train_items[i] for i in dev_idx]
    test_raw_split = test_items

    print(f"[INFO] 原始样本划分 -> train: {len(train_raw_split)}, dev: {len(dev_raw_split)}, test: {len(test_raw_split)}")

    train_out, train_stats = build_binary_dataset_for_split(
        raw_split_items=train_raw_split,
        split_name="train",
        rng_seed=args.seed + 100,
        neg_per_pos=args.neg_per_pos,
        neighbor_k=args.neighbor_k,
        hard_negative_ratio=args.hard_negative_ratio,
        num_replace=args.num_replace,
    )
    dev_out, dev_stats = build_binary_dataset_for_split(
        raw_split_items=dev_raw_split,
        split_name="dev",
        rng_seed=args.seed + 200,
        neg_per_pos=args.neg_per_pos,
        neighbor_k=args.neighbor_k,
        hard_negative_ratio=args.hard_negative_ratio,
        num_replace=args.num_replace,
    )
    test_out, test_stats = build_binary_dataset_for_split(
        raw_split_items=test_raw_split,
        split_name="test",
        rng_seed=args.seed + 300,
        neg_per_pos=args.neg_per_pos,
        neighbor_k=args.neighbor_k,
        hard_negative_ratio=args.hard_negative_ratio,
        num_replace=args.num_replace,
    )

    write_jsonl(os.path.join(args.output_dir, "train.json"), train_out)
    write_jsonl(os.path.join(args.output_dir, "dev.json"), dev_out)
    write_jsonl(os.path.join(args.output_dir, "test.json"), test_out)

    meta = {
        "config": vars(args),
        "stats": {
            "train": train_stats,
            "dev": dev_stats,
            "test": test_stats,
        }
    }
    write_json(os.path.join(args.output_dir, "reconstruct_meta.json"), meta)

    print("[DONE] 数据集重构完成")
    print(f"[SAVE] train -> {os.path.join(args.output_dir, 'train.json')}")
    print(f"[SAVE] dev   -> {os.path.join(args.output_dir, 'dev.json')}")
    print(f"[SAVE] test  -> {os.path.join(args.output_dir, 'test.json')}")
    print(f"[SAVE] meta  -> {os.path.join(args.output_dir, 'reconstruct_meta.json')}")


if __name__ == "__main__":
    main()
