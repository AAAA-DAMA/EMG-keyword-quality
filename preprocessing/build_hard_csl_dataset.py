# ner_to_csl_hard.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set


# =========================
# 可调参数
# =========================

GENERIC_TERMS = {
    "技术", "管理", "合作", "市场", "业务", "产品", "服务", "项目", "系统", "平台", "方案", "模式",
    "应用", "行业", "产业", "生态", "生态链", "建设", "开发", "升级", "改造", "生产", "制造",
    "销售", "运营", "研究", "研发", "创新", "布局", "领域", "方向", "规划", "机会", "趋势",
    "优势", "竞争力", "能力", "材料", "设备", "装置", "工程", "中心", "网络", "数据", "信息",
    "电子", "智能", "自动化", "数字化", "环保", "节能", "高性能", "绿色", "商业", "政府",
    "住宅", "农业", "林业", "交通", "国土", "教育", "保险", "机械", "化工", "科研"
}

# 这些后缀去掉后，有时仍然能保持关键词语义；但只做“轻度规范化”
REMOVABLE_SUFFIXES = [
    "系统", "设备", "平台", "服务", "方案", "业务", "项目", "装置", "领域", "产品",
    "产业", "体系", "工程", "工厂", "应用", "技术", "材料"
]

# 合法短缩写模式
SHORT_ACRONYM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\+\.]{0,9}$")

# 全是数字/标点
NON_CONTENT_RE = re.compile(r"^[\d\W_]+$", re.UNICODE)

# 中文括号中的英文缩写：卫星导航（GNSS）
BRACKET_ALIAS_RE = re.compile(r"^(.+?)[（(]([A-Za-z0-9\-\+\.]{1,20})[)）]$")


@dataclass
class Candidate:
    term: str
    count: int
    first_pos: int
    raw_spans: List[List[int]] = field(default_factory=list)
    score: float = 0.0
    idf: float = 0.0
    aliases: List[str] = field(default_factory=list)


@dataclass
class DocInfo:
    raw_id: Any
    text: str
    candidates: Dict[str, Candidate]
    core_terms: List[str] = field(default_factory=list)


# =========================
# 基础工具
# =========================

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    trans = {
        "\u3000": " ",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "（": "（", "）": "）",
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
    for k, v in trans.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_term(term: str) -> str:
    t = normalize_text(term)
    t = t.strip(" ,，;；:：.。!！?？\"'`()[]{}<>《》")
    t = re.sub(r"\s+", "", t)
    # 常见全半角/连接符统一
    t = t.replace("－", "-").replace("—", "-").replace("–", "-")
    return t


def is_short_acronym(term: str) -> bool:
    return bool(SHORT_ACRONYM_RE.match(term))


def is_bad_term(term: str) -> bool:
    if not term:
        return True
    if NON_CONTENT_RE.match(term):
        return True
    if len(term) == 1 and not is_short_acronym(term):
        return True
    if term in {"医", "药", "来", "配", "自", "中", "工", "正", "1", "2", "3", "4", "5"}:
        return True
    if len(term) <= 2 and term in GENERIC_TERMS:
        return True
    # 纯数字或比例/年份
    if re.fullmatch(r"[\d\.\-%/]+", term):
        return True
    return False


def is_generic(term: str, df_ratio: float) -> bool:
    if term in GENERIC_TERMS:
        return True
    if len(term) <= 2 and not is_short_acronym(term):
        return True
    if df_ratio >= 0.03 and len(term) <= 4 and not is_short_acronym(term):
        return True
    return False


def char_ngrams(s: str, n: int = 2) -> Set[str]:
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def surface_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a1, b1 = set(a), set(b)
    a2, b2 = char_ngrams(a, 2), char_ngrams(b, 2)

    j1 = len(a1 & b1) / max(1, len(a1 | b1))
    j2 = len(a2 & b2) / max(1, len(a2 | b2))

    contain = 1.0 if (a in b or b in a) else 0.0
    head = 1.0 if len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2] else 0.0
    tail = 1.0 if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:] else 0.0

    return 0.30 * j1 + 0.35 * j2 + 0.20 * contain + 0.075 * head + 0.075 * tail


def length_score(term: str) -> float:
    L = len(term)
    if L == 1:
        return 0.0
    if 2 <= L <= 4:
        return 0.90
    if 5 <= L <= 10:
        return 1.00
    if 11 <= L <= 18:
        return 0.85
    if 19 <= L <= 28:
        return 0.70
    return 0.55


def semantic_variants(term: str) -> List[str]:
    """
    只做高置信、轻度规范化。
    """
    variants = set()

    m = BRACKET_ALIAS_RE.match(term)
    if m:
        zh = normalize_term(m.group(1))
        en = normalize_term(m.group(2))
        if zh and zh != term and not is_bad_term(zh):
            variants.add(zh)
        if en and en != term and not is_bad_term(en):
            variants.add(en)

    # 轻度去尾
    for suf in REMOVABLE_SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= 2:
            stem = term[:-len(suf)]
            stem = normalize_term(stem)
            if stem and stem != term and not is_bad_term(stem) and stem not in GENERIC_TERMS:
                variants.add(stem)

    # 某些“XX建设”“XX改造”等，尽量缩成中心短语
    for suf in ["建设", "改造", "研发", "制造", "生产", "运营", "推广"]:
        if term.endswith(suf) and len(term) - len(suf) >= 2:
            stem = normalize_term(term[:-len(suf)])
            if stem and stem != term and not is_bad_term(stem) and stem not in GENERIC_TERMS:
                variants.add(stem)

    return sorted(variants, key=lambda x: (len(x), x))


def diversity_select(scored_terms: List[Candidate], topn: int = 6, lamb: float = 0.18) -> List[str]:
    """
    MMR 风格：既要高分，也要避免关键词之间太重复。
    """
    selected: List[Candidate] = []
    remain = scored_terms[:]

    while remain and len(selected) < topn:
        if not selected:
            selected.append(remain.pop(0))
            continue

        best_idx = None
        best_value = -1e9
        for i, cand in enumerate(remain):
            max_sim = max(surface_similarity(cand.term, s.term) for s in selected)
            value = cand.score - lamb * max_sim
            if value > best_value:
                best_value = value
                best_idx = i

        selected.append(remain.pop(best_idx))

    return [x.term for x in selected]


# =========================
# 读取与清洗
# =========================

def parse_spans(span_list: Any) -> Tuple[int, int, List[List[int]]]:
    """
    返回 count, first_pos, spans
    """
    if not isinstance(span_list, list):
        return 1, 10**9, []

    valid = []
    for x in span_list:
        if isinstance(x, list) and len(x) == 2:
            try:
                st = int(x[0])
                ed = int(x[1])
                valid.append([st, ed])
            except Exception:
                pass

    if not valid:
        return max(1, len(span_list)), 10**9, []

    first_pos = min(s[0] for s in valid)
    return len(valid), first_pos, valid


def extract_candidates(record: Dict[str, Any]) -> Dict[str, Candidate]:
    text = normalize_text(record.get("text", ""))
    label = record.get("label", {})
    ent_map = label.get("financial_entity", {}) if isinstance(label, dict) else {}

    merged: Dict[str, Candidate] = {}

    if not isinstance(ent_map, dict):
        return merged

    for raw_term, spans in ent_map.items():
        term = normalize_term(raw_term)
        if is_bad_term(term):
            continue

        count, first_pos, valid_spans = parse_spans(spans)

        if term in merged:
            merged[term].count += count
            merged[term].raw_spans.extend(valid_spans)
            merged[term].first_pos = min(merged[term].first_pos, first_pos)
        else:
            merged[term] = Candidate(
                term=term,
                count=count,
                first_pos=first_pos if first_pos != 10**9 else max(0, text.find(term)),
                raw_spans=valid_spans
            )

    # 如果 first_pos 不可靠，再回退到字符串查找
    for cand in merged.values():
        if cand.first_pos < 0 or cand.first_pos == 10**9:
            pos = text.find(cand.term)
            cand.first_pos = pos if pos >= 0 else len(text)
        cand.aliases = semantic_variants(cand.term)

    return merged


def read_jsonl(path: str) -> List[DocInfo]:
    docs: List[DocInfo] = []
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            text = normalize_text(obj.get("text", ""))
            if not text:
                bad += 1
                continue

            candidates = extract_candidates(obj)
            raw_id = obj.get("id", line_idx)
            docs.append(DocInfo(raw_id=raw_id, text=text, candidates=candidates))

    print(f"[INFO] loaded docs={len(docs)}, malformed_or_empty={bad}")
    return docs


# =========================
# 打分与核心关键词选择
# =========================

def build_df(docs: List[DocInfo]) -> Counter:
    df = Counter()
    for doc in docs:
        uniq = set(doc.candidates.keys())
        df.update(uniq)
    return df


def compute_candidate_scores(docs: List[DocInfo], df: Counter) -> None:
    N = max(1, len(docs))

    for doc in docs:
        text_len = max(1, len(doc.text))

        for term, cand in doc.candidates.items():
            term_df = df[term]
            cand.idf = math.log((N + 1) / (term_df + 1)) + 1.0

            freq_part = math.log(1 + cand.count)
            pos_part = 1.0 - min(1.0, cand.first_pos / text_len)
            len_part = length_score(term)
            contain_part = 1.0 if term in doc.text else 0.0

            df_ratio = term_df / N
            generic_pen = 1.0 if is_generic(term, df_ratio) else 0.0
            acronym_bonus = 0.15 if is_short_acronym(term) else 0.0

            cand.score = (
                0.38 * cand.idf +
                0.20 * freq_part +
                0.17 * pos_part +
                0.15 * len_part +
                0.10 * contain_part +
                acronym_bonus -
                0.22 * generic_pen
            )

        ranked = sorted(doc.candidates.values(), key=lambda x: x.score, reverse=True)
        doc.core_terms = diversity_select(ranked, topn=6, lamb=0.18)


def choose_base_keywords(doc: DocInfo, rng: random.Random, min_k: int, max_k: int) -> List[str]:
    core = list(doc.core_terms)

    if len(core) >= max_k:
        k = rng.randint(min_k, max_k)
    else:
        k = min(max_k, len(core))

    if k < min_k:
        # 尽量补齐
        ranked = sorted(doc.candidates.values(), key=lambda x: x.score, reverse=True)
        for cand in ranked:
            if cand.term not in core:
                core.append(cand.term)
            if len(core) >= min_k:
                break
        k = min(len(core), max(min_k, k))

    return core[:k]


# =========================
# 全局桶：用于跨文档近似替换
# =========================

def pattern_signature(term: str) -> str:
    has_digit = any(ch.isdigit() for ch in term)
    has_alpha = any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in term)
    chinese_len = sum('\u4e00' <= ch <= '\u9fff' for ch in term)

    parts = []
    parts.append("D1" if has_digit else "D0")
    parts.append("A1" if has_alpha else "A0")
    if chinese_len <= 2:
        parts.append("L2")
    elif chinese_len <= 4:
        parts.append("L4")
    elif chinese_len <= 8:
        parts.append("L8")
    else:
        parts.append("L9")
    return "_".join(parts)


def build_term_buckets(docs: List[DocInfo]) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[str, Set[int]]]:
    buckets: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    term_to_doc_ids: Dict[str, Set[int]] = defaultdict(set)

    for doc_id, doc in enumerate(docs):
        for term in doc.candidates.keys():
            term_to_doc_ids[term].add(doc_id)

            if len(term) >= 2:
                buckets[("head2", term[:2])].add(term)
                buckets[("tail2", term[-2:])].add(term)
            if len(term) >= 1:
                buckets[("head1", term[:1])].add(term)
                buckets[("tail1", term[-1:])].add(term)

            for bg in char_ngrams(term, 2):
                buckets[("bg", bg)].add(term)

            buckets[("pat", pattern_signature(term))].add(term)

    return buckets, term_to_doc_ids


def global_term_score(term: str, df: Counter, num_docs: int) -> float:
    term_df = df.get(term, 1)
    idf = math.log((num_docs + 1) / (term_df + 1)) + 1.0
    return idf + 0.15 * length_score(term)


# =========================
# 正样本生成
# =========================

def make_positive_variants(
    base_keywords: List[str],
    doc: DocInfo,
    rng: random.Random
) -> List[List[str]]:
    positives = []

    # 正样本1：exact-positive
    positives.append(list(base_keywords))

    # 正样本2：尝试轻度规范化；如果没有可靠别名，就换一个同样正确的核心词
    second = list(base_keywords)
    replaced = False

    for i, kw in enumerate(second):
        cand = doc.candidates.get(kw)
        if not cand:
            continue

        for alias in cand.aliases:
            if alias != kw and alias not in second and not is_bad_term(alias):
                second[i] = alias
                replaced = True
                break
        if replaced:
            break

    if not replaced:
        # 没有 alias，就尝试用另一个更靠后的 core term 替掉一个位置，仍然是正样本
        extras = [x for x in doc.core_terms if x not in second]
        if extras:
            idx = rng.randrange(len(second))
            second[idx] = extras[0]
            replaced = True

    if replaced and second != positives[0]:
        positives.append(second)

    # 去重
    uniq = []
    seen = set()
    for kws in positives:
        key = tuple(kws)
        if key not in seen:
            uniq.append(kws)
            seen.add(key)
    return uniq


# =========================
# 负样本生成
# =========================

def rank_in_doc_distractors(
    target_kw: str,
    base_keywords: List[str],
    doc: DocInfo
) -> List[str]:
    """
    从同一篇文本里找“看起来像对，但不是核心关键词”的词。
    """
    others = set(base_keywords)
    pool = [c for c in doc.candidates.values() if c.term not in others]

    ranked = []
    for cand in pool:
        sim_target = surface_similarity(target_kw, cand.term)
        sim_group = 0.0
        if len(base_keywords) > 1:
            sim_group = sum(surface_similarity(cand.term, x) for x in base_keywords if x != target_kw) / max(1, len(base_keywords) - 1)

        # 文内难负样本：既不能太离谱，也不能和 target 完全一样
        if cand.term == target_kw:
            continue

        score = (
            0.45 * cand.score +
            0.30 * sim_target +
            0.15 * sim_group +
            0.10 * (1.0 if cand.term in doc.text else 0.0)
        )

        # 太泛的词降权
        if cand.term in GENERIC_TERMS:
            score -= 0.25

        ranked.append((score, cand.term))

    ranked.sort(reverse=True)
    return [x[1] for x in ranked]


def rank_global_distractors(
    target_kw: str,
    base_keywords: List[str],
    current_doc: DocInfo,
    docs: List[DocInfo],
    df: Counter,
    buckets: Dict[Tuple[str, str], Set[str]],
    term_to_doc_ids: Dict[str, Set[int]]
) -> List[str]:
    """
    从全局近似词里找替换项：
    共享前后缀、共享 bigram、模式相近、长度相近。
    """
    pool: Set[str] = set()

    if len(target_kw) >= 2:
        pool |= buckets.get(("head2", target_kw[:2]), set())
        pool |= buckets.get(("tail2", target_kw[-2:]), set())
    pool |= buckets.get(("pat", pattern_signature(target_kw)), set())

    for bg in list(char_ngrams(target_kw, 2))[:4]:
        pool |= buckets.get(("bg", bg), set())

    # 删除当前文档已有的候选词，尽量做跨文档干扰
    current_terms = set(current_doc.candidates.keys())
    pool = {t for t in pool if t not in current_terms and t not in base_keywords and t != target_kw}

    ranked = []
    num_docs = max(1, len(docs))
    for term in pool:
        sim_target = surface_similarity(target_kw, term)
        if sim_target < 0.08:
            continue

        sim_group = sum(surface_similarity(term, x) for x in base_keywords) / max(1, len(base_keywords))
        rarity = global_term_score(term, df, num_docs)

        score = 0.48 * sim_target + 0.18 * sim_group + 0.34 * rarity

        if term in GENERIC_TERMS:
            score -= 0.20

        ranked.append((score, term))

    ranked.sort(reverse=True)
    return [x[1] for x in ranked]


def replace_at(lst: List[str], idx: int, value: str) -> List[str]:
    out = list(lst)
    out[idx] = value
    return out


def generate_negatives(
    doc: DocInfo,
    base_keywords: List[str],
    docs: List[DocInfo],
    df: Counter,
    buckets: Dict[Tuple[str, str], Set[str]],
    term_to_doc_ids: Dict[str, Set[int]],
    rng: random.Random,
    need_n: int = 3
) -> List[List[str]]:
    """
    负样本策略：
    1) 文内单替换
    2) 文内再单替换或双替换
    3) 跨文档近似替换
    """
    negatives: List[List[str]] = []
    used = {tuple(base_keywords)}

    # 按 target 的“核心程度”从低到高尝试替换，避免老是替换最关键术语
    core_scored = []
    for kw in base_keywords:
        cand = doc.candidates.get(kw)
        score = cand.score if cand else 0.0
        core_scored.append((score, kw))
    core_scored.sort()  # 分数低的先替换

    def add_if_valid(kws: List[str]):
        key = tuple(kws)
        if key in used:
            return
        if len(set(kws)) != len(kws):
            return
        used.add(key)
        negatives.append(kws)

    # ---- 负样本1：文内单替换 ----
    for _, target_kw in core_scored:
        idx = base_keywords.index(target_kw)
        candidates = rank_in_doc_distractors(target_kw, base_keywords, doc)
        for rep in candidates[:10]:
            sample = replace_at(base_keywords, idx, rep)
            add_if_valid(sample)
            if len(negatives) >= 1:
                break
        if len(negatives) >= 1:
            break

    # ---- 负样本2：文内再替换，尽量和第一次不同 ----
    if len(negatives) < 2:
        for _, target_kw in reversed(core_scored):
            idx = base_keywords.index(target_kw)
            candidates = rank_in_doc_distractors(target_kw, base_keywords, doc)
            for rep in candidates[1:12]:
                sample = replace_at(base_keywords, idx, rep)
                add_if_valid(sample)
                if len(negatives) >= 2:
                    break
            if len(negatives) >= 2:
                break

    # 如果还不够，尝试双替换（仍然优先文内）
    if len(negatives) < 2 and len(base_keywords) >= 3:
        idxs = list(range(len(base_keywords)))
        rng.shuffle(idxs)
        for i in idxs:
            target1 = base_keywords[i]
            cand1 = rank_in_doc_distractors(target1, base_keywords, doc)[:6]
            if not cand1:
                continue
            for j in idxs:
                if j == i:
                    continue
                target2 = base_keywords[j]
                cand2 = rank_in_doc_distractors(target2, base_keywords, doc)[:6]
                if not cand2:
                    continue
                for r1 in cand1:
                    for r2 in cand2:
                        if r1 == r2:
                            continue
                        tmp = list(base_keywords)
                        tmp[i] = r1
                        tmp[j] = r2
                        add_if_valid(tmp)
                        if len(negatives) >= 2:
                            break
                    if len(negatives) >= 2:
                        break
                if len(negatives) >= 2:
                    break
            if len(negatives) >= 2:
                break

    # ---- 负样本3：跨文档近似替换 ----
    if len(negatives) < need_n:
        for _, target_kw in core_scored:
            idx = base_keywords.index(target_kw)
            candidates = rank_global_distractors(
                target_kw=target_kw,
                base_keywords=base_keywords,
                current_doc=doc,
                docs=docs,
                df=df,
                buckets=buckets,
                term_to_doc_ids=term_to_doc_ids
            )
            for rep in candidates[:15]:
                sample = replace_at(base_keywords, idx, rep)
                add_if_valid(sample)
                if len(negatives) >= need_n:
                    break
            if len(negatives) >= need_n:
                break

    return negatives[:need_n]


# =========================
# 主流程
# =========================

def build_csl_samples(
    docs: List[DocInfo],
    min_keywords: int,
    max_keywords: int,
    positives_per_doc: int,
    negatives_per_doc: int,
    seed: int
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    df = build_df(docs)
    compute_candidate_scores(docs, df)
    buckets, term_to_doc_ids = build_term_buckets(docs)

    out_samples: List[Dict[str, Any]] = []
    skipped = 0

    for di, doc in enumerate(docs):
        # 候选太少，跳过
        if len(doc.candidates) < 2:
            skipped += 1
            continue

        base_keywords = choose_base_keywords(doc, rng, min_k=min_keywords, max_k=max_keywords)
        if len(base_keywords) < min_keywords:
            skipped += 1
            continue

        # 正样本
        pos_variants = make_positive_variants(base_keywords, doc, rng)[:positives_per_doc]
        for pi, kws in enumerate(pos_variants, start=1):
            sample = {
                "id": f"fin_{di:06d}_pos_{pi}",
                "abst": doc.text,
                "keyword": kws,
                "label": "1"
            }
            out_samples.append(sample)

        # 负样本
        neg_variants = generate_negatives(
            doc=doc,
            base_keywords=base_keywords,
            docs=docs,
            df=df,
            buckets=buckets,
            term_to_doc_ids=term_to_doc_ids,
            rng=rng,
            need_n=negatives_per_doc
        )

        for ni, kws in enumerate(neg_variants, start=1):
            sample = {
                "id": f"fin_{di:06d}_neg_{ni}",
                "abst": doc.text,
                "keyword": kws,
                "label": "0"
            }
            out_samples.append(sample)

    print(f"[INFO] skipped_docs={skipped}, output_samples={len(out_samples)}")
    return out_samples


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert financial NER jsonl to CSL-style hard-negative keyword classification jsonl.")
    parser.add_argument("--input", type=str, required=True, help="输入 NER jsonl 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 CSL-style jsonl 文件路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_keywords", type=int, default=3)
    parser.add_argument("--max_keywords", type=int, default=4)
    parser.add_argument("--positives_per_doc", type=int, default=2)
    parser.add_argument("--negatives_per_doc", type=int, default=3)
    args = parser.parse_args()

    docs = read_jsonl(args.input)
    samples = build_csl_samples(
        docs=docs,
        min_keywords=args.min_keywords,
        max_keywords=args.max_keywords,
        positives_per_doc=args.positives_per_doc,
        negatives_per_doc=args.negatives_per_doc,
        seed=args.seed
    )
    write_jsonl(args.output, samples)
    print(f"[DONE] wrote => {args.output}")


if __name__ == "__main__":
    main()
