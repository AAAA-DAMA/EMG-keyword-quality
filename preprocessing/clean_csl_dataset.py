#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-clean a generated CSL-style dataset so it is safer to use in paper experiments.

Input format (jsonl):
{"id": "...", "abst": "...", "keyword": ["...", "...", ...], "label": "0|1"}

Main goals:
1) Remove contradictory samples: same abst + same keyword set appearing in both labels.
2) Remove low-quality / fabricated / overly-generic keywords.
3) Keep only hard negatives that are close to positives for the same abst, but not ambiguous.
4) Downsample repeated variants to a balanced and cleaner dataset.
5) Write a cleaning report for reproducibility.

Recommended usage:
python fix_csl_dataset.py \
  --input /root/autodl-tmp/MVPproject/fintech/train_csl_style.json \
  --output /root/autodl-tmp/MVPproject/fintech/train_csl_style.clean.json \
  --report /root/autodl-tmp/MVPproject/fintech/train_csl_style.clean.report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ======== 可调词表：偏保守，宁可删得严一点，也不要保留脏样本 ========
GENERIC_TERMS = {
    "技术", "技术开发", "技术咨询", "技术服务", "技术转让", "开发", "咨询", "服务", "转让",
    "项目", "业务", "产品", "系统", "平台", "方案", "模式", "领域", "产业", "市场", "建设",
    "运营", "管理", "能力", "应用", "方向", "机会", "趋势", "服务业", "解决方案",
    "国际水平", "商务办公型", "位置", "开放", "制品", "能耗", "综合能力", "综合服务",
    "行业", "政府", "商业服务", "创新协同", "系统集成", "终端产品", "相关终端产品",
}

# 这些是“负样本里常见的伪词模式”，论文实验里建议直接删掉
BAD_TERMS_EXACT = {
    "大医药", "医药工", "高效机", "无线", "开放", "制品", "位置", "能耗",
    "社保创新类", "绿色健康", "女性健康", "天然健康", "非航空", "非电信",
}

# 合法短缩写白名单
SHORT_WHITELIST = {
    "AI", "AIOPC", "GIS", "LAN", "ISDN", "LCD", "AMOLED", "CF", "DNC",
    "TN", "STN", "LTPS", "5G", "4G", "3G", "VoIP", "OGM", "OGS",
    "On-cell", "In-cell", "TFT-LCD", "CSTN-LCD", "LTPSTFT-LCD",
}


@dataclass
class Sample:
    sid: str
    abst: str
    keyword: List[str]
    label: str


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "－": "-", "—": "-", "–": "-",
        "（": "(", "）": ")",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "\t": " ", "\r": " ", "\n": " ", "\u3000": " ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_term(t: str) -> str:
    t = normalize_text(t)
    t = t.strip(" ,，;；:：.。!！?？\"'`()[]{}<>《》")
    return t


def is_ascii_short(term: str) -> bool:
    return term in SHORT_WHITELIST or bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-\.\+]{0,15}", term))


def contains_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def char_bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i+2] for i in range(len(s) - 1)}


def surface_similarity(a: str, b: str) -> float:
    a = normalize_term(a)
    b = normalize_term(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sa, sb = set(a), set(b)
    j1 = len(sa & sb) / max(1, len(sa | sb))
    a2, b2 = char_bigrams(a), char_bigrams(b)
    j2 = len(a2 & b2) / max(1, len(a2 | b2))
    contain = 1.0 if a in b or b in a else 0.0
    prefix = 1.0 if len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2] else 0.0
    suffix = 1.0 if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:] else 0.0
    return 0.25 * j1 + 0.35 * j2 + 0.20 * contain + 0.10 * prefix + 0.10 * suffix


def is_obviously_bad_term(term: str) -> bool:
    term = normalize_term(term)
    if not term:
        return True
    if term in BAD_TERMS_EXACT:
        return True
    if re.fullmatch(r"[\W_\d]+", term):
        return True
    if len(term) == 1 and not is_ascii_short(term):
        return True
    if term in {"医", "药", "来", "配", "自", "正", "工", "类", "化", "性", "新型"}:
        return True
    # “非航空/非电信”这类很程序味，直接删
    if re.fullmatch(r"非[\u4e00-\u9fffA-Za-z0-9\-]{1,8}", term):
        return True
    return False


def is_too_generic(term: str) -> bool:
    term = normalize_term(term)
    if term in GENERIC_TERMS:
        return True
    if len(term) <= 2 and not is_ascii_short(term):
        # 2字中文词不一刀切，但对特别泛的保守过滤
        if term in {"位置", "开放", "制品", "能耗", "视频", "互联网", "软件"}:
            return True
    # 很短且带抽象后缀，通常是截断伪词
    if len(term) <= 3 and term.endswith(("工", "管", "化", "类", "性")) and not is_ascii_short(term):
        return True
    return False


def looks_like_truncated_fragment(term: str, ref_terms: Sequence[str]) -> bool:
    """如“电熔” vs “电熔氧化锆”，“触控” vs “触控一体化”，这类过短截断词删掉。"""
    term = normalize_term(term)
    if not term or len(term) > 4:
        return False
    if is_ascii_short(term):
        return False
    for ref in ref_terms:
        ref = normalize_term(ref)
        if ref == term:
            continue
        if term in ref and len(ref) - len(term) >= 2:
            return True
    return False


def term_quality_score(term: str, abst_norm: str) -> float:
    term = normalize_term(term)
    score = 0.0
    if len(term) >= 2:
        score += 0.6
    if len(term) >= 4:
        score += 0.3
    if len(term) >= 6:
        score += 0.2
    if term in abst_norm:
        score += 0.4
    if is_ascii_short(term):
        score += 0.2
    if not is_too_generic(term):
        score += 0.5
    if is_obviously_bad_term(term):
        score -= 2.0
    if looks_like_truncated_fragment(term, [abst_norm]):
        score -= 0.8
    return score


def sample_quality_score(sample: Sample) -> float:
    abst_norm = normalize_text(sample.abst)
    return sum(term_quality_score(k, abst_norm) for k in sample.keyword) / max(1, len(sample.keyword))


def read_jsonl(path: str) -> List[Sample]:
    records: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                Sample(
                    sid=str(obj.get("id", idx)),
                    abst=str(obj["abst"]),
                    keyword=[str(x) for x in obj["keyword"]],
                    label=str(obj["label"]),
                )
            )
    return records


def write_jsonl(path: str, records: List[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r.sid,
                "abst": r.abst,
                "keyword": r.keyword,
                "label": r.label,
            }, ensure_ascii=False) + "\n")


def canonical_key(sample: Sample, order_sensitive: bool = False) -> Tuple[str, Tuple[str, ...]]:
    abst = normalize_text(sample.abst)
    kws = tuple(normalize_term(x) for x in sample.keyword)
    if not order_sensitive:
        kws = tuple(sorted(kws))
    return abst, kws


def validate_keyword_list(sample: Sample, positive_union: Optional[set[str]] = None) -> Tuple[bool, str]:
    """返回 (是否通过, 原因)"""
    kws = [normalize_term(x) for x in sample.keyword]
    abst_norm = normalize_text(sample.abst)

    if sample.label not in {"0", "1"}:
        return False, "bad_label"
    if not sample.abst.strip():
        return False, "empty_abst"
    if len(kws) < 3 or len(kws) > 4:
        return False, "bad_kw_len"
    if len(set(kws)) != len(kws):
        return False, "dup_keyword_inside_sample"

    for kw in kws:
        if is_obviously_bad_term(kw):
            return False, "obviously_bad_term"
        if is_too_generic(kw):
            return False, "too_generic_term"

    # 正样本：关键词应尽量真的在文本里（经规范化后）
    if sample.label == "1":
        missing = [kw for kw in kws if kw not in abst_norm]
        if missing:
            return False, "positive_term_not_in_abst"
        return True, "ok"

    # 负样本：如果已有正样本核心词集合，可以做更严格检查
    if positive_union is not None:
        outside = [kw for kw in kws if kw not in positive_union]
        if not outside:
            return False, "negative_subset_of_positive_union"

        # 替换出来的词必须“像该领域词”，不能太离谱，也不能是过短截断词
        for kw in outside:
            if looks_like_truncated_fragment(kw, list(positive_union)):
                return False, "negative_truncated_fragment"

            max_sim = max((surface_similarity(kw, p) for p in positive_union), default=0.0)
            in_abst = kw in abst_norm

            # 不在原文里，又和正样本核心词没什么相似度 => 太假
            if (not in_abst) and max_sim < 0.16:
                return False, "negative_too_far_from_positive"

            # 在原文里，但本身过泛，也不要
            if in_abst and is_too_generic(kw):
                return False, "negative_generic_in_abst"

    return True, "ok"


def pick_top_unique(samples: List[Sample], topk: int) -> List[Sample]:
    """按样本质量挑 topk，去掉同 label 下同一 abst 内同义重复（无序关键词集合相同）"""
    uniq: Dict[Tuple[str, Tuple[str, ...], str], Sample] = {}
    for s in samples:
        key = (*canonical_key(s, order_sensitive=False), s.label)
        old = uniq.get(key)
        if old is None or sample_quality_score(s) > sample_quality_score(old):
            uniq[key] = s
    ranked = sorted(uniq.values(), key=sample_quality_score, reverse=True)
    return ranked[:topk]


def clean_group(
    abst: str,
    group_samples: List[Sample],
    max_pos_per_abst: int,
    max_neg_per_abst: int,
    stats: Counter,
) -> List[Sample]:
    abst_norm = normalize_text(abst)
    pos_raw = [s for s in group_samples if s.label == "1"]
    neg_raw = [s for s in group_samples if s.label == "0"]

    # 1) 先去掉 exact duplicate（同 label）
    tmp: Dict[Tuple[str, Tuple[str, ...], str], Sample] = {}
    for s in group_samples:
        key = (*canonical_key(s, order_sensitive=False), s.label)
        if key not in tmp:
            tmp[key] = s
        else:
            stats["removed_exact_dup"] += 1
    group_samples = list(tmp.values())
    pos_raw = [s for s in group_samples if s.label == "1"]
    neg_raw = [s for s in group_samples if s.label == "0"]

    # 2) 去掉“同一关键词集合同时出现 1 和 0”的冲突项：优先保留正样本
    pos_keys = {canonical_key(s, order_sensitive=False)[1] for s in pos_raw}
    cleaned_neg_after_conflict = []
    for s in neg_raw:
        if canonical_key(s, order_sensitive=False)[1] in pos_keys:
            stats["removed_pos_neg_conflict"] += 1
            continue
        cleaned_neg_after_conflict.append(s)
    neg_raw = cleaned_neg_after_conflict

    # 3) 清正样本：正样本必须干净，且关键词应在原文中
    pos_valid: List[Sample] = []
    for s in pos_raw:
        ok, reason = validate_keyword_list(s, positive_union=None)
        if not ok:
            stats[f"drop_pos_{reason}"] += 1
            continue
        pos_valid.append(s)

    if not pos_valid:
        stats["drop_group_no_valid_positive"] += 1
        return []

    pos_kept = pick_top_unique(pos_valid, max_pos_per_abst)
    pos_union = set()
    for s in pos_kept:
        pos_union.update(normalize_term(x) for x in s.keyword)
    pos_set_keys = {canonical_key(s, order_sensitive=False)[1] for s in pos_kept}

    # 4) 清负样本：必须至少包含一个不在正样本 union 里的词；并且替换词要“像该领域词”
    neg_valid: List[Sample] = []
    for s in neg_raw:
        # 如果 negative 其实只是正样本 union 的另一个子集，直接删
        if set(normalize_term(x) for x in s.keyword).issubset(pos_union):
            stats["drop_neg_subset_of_positive_union"] += 1
            continue
        ok, reason = validate_keyword_list(s, positive_union=pos_union)
        if not ok:
            stats[f"drop_neg_{reason}"] += 1
            continue
        # 还要确保它离某个正样本足够“近”（通常只替换 1 个词），这样才是真 hard negative
        best_overlap = 0
        sset = set(normalize_term(x) for x in s.keyword)
        for pk in pos_set_keys:
            if len(pk) != len(s.keyword):
                continue
            best_overlap = max(best_overlap, len(sset & set(pk)))
        if best_overlap < len(s.keyword) - 1:
            stats["drop_neg_not_hard_enough"] += 1
            continue
        neg_valid.append(s)

    # 5) 下采样：每个 abst 最多保留 2 个正、2 个负，避免重复/过拟合
    neg_kept = pick_top_unique(neg_valid, max_neg_per_abst)

    # 若正负严重不平衡，做局部平衡（默认保守）
    if pos_kept and neg_kept:
        target = min(len(pos_kept), len(neg_kept))
        pos_kept = pos_kept[:target]
        neg_kept = neg_kept[:target]

    if not neg_kept:
        # 允许只保留正样本吗？论文实验里建议不要。这里默认整组舍弃，保证任务性一致。
        stats["drop_group_no_valid_negative"] += 1
        return []

    output: List[Sample] = []
    for i, s in enumerate(pos_kept, start=1):
        output.append(Sample(
            sid=f"clean_{abs(hash((abst_norm, 'pos', i))) % 10**12:012d}",
            abst=s.abst,
            keyword=s.keyword,
            label="1",
        ))
    for i, s in enumerate(neg_kept, start=1):
        output.append(Sample(
            sid=f"clean_{abs(hash((abst_norm, 'neg', i))) % 10**12:012d}",
            abst=s.abst,
            keyword=s.keyword,
            label="0",
        ))
    return output


def clean_dataset(
    records: List[Sample],
    max_pos_per_abst: int = 2,
    max_neg_per_abst: int = 2,
) -> Tuple[List[Sample], Dict[str, Any]]:
    stats: Counter = Counter()
    by_abst: Dict[str, List[Sample]] = defaultdict(list)
    for s in records:
        by_abst[s.abst].append(s)

    cleaned: List[Sample] = []
    for abst, group_samples in by_abst.items():
        cleaned.extend(clean_group(
            abst=abst,
            group_samples=group_samples,
            max_pos_per_abst=max_pos_per_abst,
            max_neg_per_abst=max_neg_per_abst,
            stats=stats,
        ))

    report = {
        "input_samples": len(records),
        "output_samples": len(cleaned),
        "input_unique_abst": len(by_abst),
        "output_unique_abst": len(set(s.abst for s in cleaned)),
        "output_pos": sum(s.label == "1" for s in cleaned),
        "output_neg": sum(s.label == "0" for s in cleaned),
        "stats": dict(stats),
    }
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated CSL-style dataset for paper experiments.")
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output", required=True, help="Output cleaned jsonl path")
    parser.add_argument("--report", default=None, help="Optional report json path")
    parser.add_argument("--max_pos_per_abst", type=int, default=2)
    parser.add_argument("--max_neg_per_abst", type=int, default=2)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    cleaned, report = clean_dataset(
        records=records,
        max_pos_per_abst=args.max_pos_per_abst,
        max_neg_per_abst=args.max_neg_per_abst,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(args.output, cleaned)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] cleaned file written to: {args.output}")


if __name__ == "__main__":
    main()
