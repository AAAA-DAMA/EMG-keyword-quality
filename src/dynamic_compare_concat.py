#!/usr/bin/env python3
# coding: utf-8
"""
train_improved_attention.py

完整单文件实现（数据加载 / 改进 Attention / 训练流程）
修复 TypeError: ImprovedBertSelfAttention.forward() got an unexpected keyword argument 'encoder_hidden_states'

用法示例:
python train_improved_attention.py --original_train /path/train.json --original_dev /path/dev.json --model_name bert-base-chinese --epochs 6
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import argparse
import logging
import sys
import time
import re
from typing import List, Dict
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------------------------
# Repro helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def worker_init_fn(worker_id):
    seed = GLOBAL_SEED
    worker_seed = seed + worker_id + 1
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# ---------------------------
# Dataset
# ---------------------------
class AbstKeywordDataset(Dataset):
    def __init__(self, json_path: str):
        assert os.path.exists(json_path), f"数据集不存在：{json_path}"
        items = []
        logger.info(f"正在加载文件: {json_path}")
        success_count = 0
        total_lines = 0

        with open(json_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    logger.warning(f"行 {lineno} JSON 解析失败，跳过。错误：{e}")
                    continue

                # 字段校验（你的数据示例使用 id, abst, keyword, label）
                if 'id' not in obj or 'abst' not in obj or 'keyword' not in obj:
                    logger.warning(f"行 {lineno} 缺少 id/abst/keyword 字段，跳过，现有字段: {list(obj.keys())}")
                    continue

                abst = str(obj['abst']).strip()
                if not abst:
                    logger.warning(f"行 {lineno} 摘要为空，跳过")
                    continue

                kw = obj['keyword']
                if isinstance(kw, str):
                    for sep in [';', ',', '，', '；', ' ']:
                        if sep in kw:
                            kw_list = [k.strip() for k in kw.split(sep) if k.strip()]
                            break
                    else:
                        kw_list = [kw.strip()] if kw.strip() else []
                elif isinstance(kw, list):
                    kw_list = [str(x).strip() for x in kw if str(x).strip()]
                else:
                    kw_list = []

                label = None
                if 'label' in obj:
                    try:
                        label_val = obj['label']
                        if isinstance(label_val, (int, float)):
                            label = int(label_val)
                        else:
                            s = str(label_val).strip()
                            if s.isdigit():
                                label = int(s)
                            elif s.lower() in ['true', 'yes', '1']:
                                label = 1
                            else:
                                label = 0
                        if label not in (0, 1):
                            label = 1 if label > 0 else 0
                    except Exception:
                        logger.warning(f"行 {lineno} 标签解析异常，设置为0")
                        label = 0

                items.append({
                    'id': obj.get('id'),
                    'abst': abst,
                    'keywords': kw_list,
                    'label': label
                })
                success_count += 1

        if len(items) == 0:
            raise RuntimeError(f"{json_path} 中未找到有效样本（共处理 {total_lines} 行，成功 {success_count} 行）")
        logger.info(f"加载 {json_path} 成功，共 {len(items)} 个有效样本（总计 {total_lines} 行）")
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# ---------------------------
# Collate function (term_mask building)
# ---------------------------
def make_collate_fn(tokenizer: BertTokenizer, max_length: int = 512):
    try:
        import jieba
        have_jieba = True
    except Exception:
        have_jieba = False

    def collate(batch: List[Dict]):
        texts_a = [it['abst'] for it in batch]
        texts_b = []
        ids = [it['id'] for it in batch]
        labels = [(-1 if it['label'] is None else it['label']) for it in batch]

        for it in batch:
            if it['keywords']:
                texts_b.append('；'.join(it['keywords']))
            else:
                texts_b.append('')

        enc = tokenizer(
            text=texts_a,
            text_pair=texts_b,
            padding='longest',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = enc['input_ids']  # (B, L)
        bsz, seq_len = input_ids.shape
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
                    sub_enc = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)
                    off_map = sub_enc['offset_mapping']
                    start_char = decoded.find(kw)
                    if start_char >= 0:
                        end_char = start_char + len(kw)
                        for t_idx, (s, e) in enumerate(off_map):
                            if s < end_char and e > start_char:
                                if t_idx < seq_len:
                                    term_mask[i, t_idx] = 1
                if not matched and have_jieba:
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
        batch_dict = {
            'ids': ids,
            'input_ids': input_ids,
            'attention_mask': enc['attention_mask'],
            'token_type_ids': token_type_ids,
            'term_mask': term_mask.long(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        return batch_dict

    return collate

def save_predictions_csv(path, ids, labels, preds, probs):
    """
    保存预测结果到 CSV 格式
    """
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "pred", "prob0", "prob1"])
        for _id, label, pred, prob in zip(ids, labels, preds, probs):
            writer.writerow([_id, label, pred, prob[0], prob[1]])
def save_gates_csv(path, gates):
    """
    保存 GMU 门控分布到 CSV
    gates: list of tensors (B, 3)
    """
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["g_token", "g_sentence", "g_paragraph"])
        for batch in gates:
            for row in batch.tolist():
                writer.writerow(row)

# ---------------------------
# Improved Self-Attention (compatible HF signature)
# ---------------------------
class ImprovedBertSelfAttention(nn.Module):
    """
    HF-compatible implementation with per-head bias.
    forward signature matches HuggingFace's BertSelfAttention.
    Stores bias regularization in self._last_bias_reg (tensor).
    """
    def __init__(self, config, bias_scale: float = 0.5, bias_reg_weight: float = 5e-6):
        super().__init__()
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        assert hidden_size % num_attention_heads == 0
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        # QKV
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.bias_scale = bias_scale
        self.bias_reg_weight = bias_reg_weight

        # per-head bias - 使用常数初始化替代正态分布
        self.head_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))
        nn.init.constant_(self.head_bias, 0.1)  # 小正值初始偏置

        # placeholder for last bias reg loss
        self._last_bias_reg = torch.tensor(0.0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_len, head_dim)

    def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    **kwargs  # 添加这一行来接收所有未知参数，包括past_key_values
):
        mixed_query_layer = self.query(hidden_states)
        
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # add per-head bias (broadcasted)
        attention_scores = attention_scores + (self.bias_scale * self.head_bias)
        
        # handle attention mask: (b, s) or already extended
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_mask = attention_mask[:, None, None, :].to(attention_scores.dtype)
                extended_mask = (1.0 - extended_mask) * -10000.0
                attention_scores = attention_scores + extended_mask
            else:
                attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs) 
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        
        # bias reg
        bias_reg_loss = self.bias_reg_weight * torch.sum(self.head_bias ** 2)
        self._last_bias_reg = bias_reg_loss
        
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        
        return outputs

# ---------------------------
# Helper to replace bert self-attention layers
# ---------------------------
def replace_bert_self_attention_with_improved(bert_model: BertModel, bias_scale=0.5, bias_reg_weight=5e-6):
    """
    Replace each layer.attention.self in bert_model with ImprovedBertSelfAttention,
    trying to copy QKV weights to preserve pretrained initialization.
    """
    for i, layer in enumerate(bert_model.encoder.layer):
        old_self = layer.attention.self
        cfg = bert_model.config
        new_self = ImprovedBertSelfAttention(cfg, bias_scale=bias_scale, bias_reg_weight=bias_reg_weight)
        # copy weights if possible
        try:
            new_self.query.weight.data.copy_(old_self.query.weight.data)
            new_self.query.bias.data.copy_(old_self.query.bias.data)
            new_self.key.weight.data.copy_(old_self.key.weight.data)
            new_self.key.bias.data.copy_(old_self.key.bias.data)
            new_self.value.weight.data.copy_(old_self.value.weight.data)
            new_self.value.bias.data.copy_(old_self.value.bias.data)
            logger.info(f"[replace] copied QKV weights for layer {i}")
        except Exception as e:
            logger.warning(f"[replace] failed to copy QKV weights for layer {i}: {e}")
        layer.attention.self = new_self

# ---------------------------
# TermAttentionEnhancedBERT (uses replaced internal attention)
# ---------------------------
class TermAttentionEnhancedBERT(nn.Module):
    def __init__(self, pretrained_name='bert-base-chinese',
                 dropout_prob=0.2,
                 term_bias_scale=2.0,
                 term_bias_reg_weight=5e-6,
                 internal_bias_scale=0.4,
                 internal_bias_reg_weight=5e-6):
        super().__init__()
        # load bert
        self.bert = BertModel.from_pretrained(pretrained_name)
        # replace internal self-attention
        replace_bert_self_attention_with_improved(self.bert,
                                                 bias_scale=internal_bias_scale,
                                                 bias_reg_weight=internal_bias_reg_weight)

        hidden = self.bert.config.hidden_size
        self.term_attention_bias = nn.Linear(hidden, 1)
        self.term_feature_projection = nn.Linear(hidden, hidden // 4)
        
        # 修复：确保分类器输入维度正确
        self.classifier_input_dim = hidden + (hidden // 4)
        self.classifier = nn.Linear(self.classifier_input_dim, 2)
        
        self.dropout = nn.Dropout(dropout_prob)

        self.term_bias_scale = term_bias_scale
        self.term_bias_reg_weight = term_bias_reg_weight
        self.current_epoch = 0  # 添加训练轮次记录

        logger.info(f"初始化 TermAttentionEnhancedBERT: 分类器输入维度={self.classifier_input_dim}")

    def set_epoch(self, epoch):
        """设置当前训练轮次，用于渐进式训练"""
        self.current_epoch = epoch

    def forward(self, input_ids, attention_mask, token_type_ids=None, term_mask=None, labels=None, class_weights=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # gather internal bias regs
        internal_bias_reg = torch.tensor(0.0, device=sequence_output.device, dtype=sequence_output.dtype)
        for layer in self.bert.encoder.layer:
            try:
                v = layer.attention.self._last_bias_reg
                if isinstance(v, torch.Tensor):
                    internal_bias_reg = internal_bias_reg + v.to(sequence_output.device)
            except Exception:
                pass

        external_term_bias_reg = torch.tensor(0.0, device=sequence_output.device, dtype=sequence_output.dtype)
        
        # 渐进式训练：前2个epoch不使用术语注意力
        use_term_attention = term_mask is not None and (not self.training or self.current_epoch >= 2)
        
        if use_term_attention:
            raw_term_bias = self.term_attention_bias(sequence_output).squeeze(-1)
            
            # 动态偏置尺度：根据训练进度调整
            if self.training:
                # 从第2个epoch开始，逐步增加偏置强度
                progress_ratio = min(1.0, (self.current_epoch - 1) / 4.0)  # 4个epoch达到最大强度
                current_scale = self.term_bias_scale * progress_ratio
            else:
                current_scale = self.term_bias_scale
                
            scaled_term_bias = torch.tanh(raw_term_bias) * current_scale

            # pad mask -> large negative logits for pads
            base_mask_logits = (1.0 - attention_mask.float()) * -10000.0
            term_bias = torch.where(term_mask.bool(), scaled_term_bias, -5.0 * torch.ones_like(scaled_term_bias))

            enhanced_attention = base_mask_logits + term_bias
            term_weights = torch.softmax(enhanced_attention, dim=-1)
            term_enhanced_output = torch.sum(sequence_output * term_weights.unsqueeze(-1), dim=1)
            term_features = self.term_feature_projection(term_enhanced_output)
            combined_features = torch.cat([pooled_output, term_features], dim=-1)

            external_term_bias_reg = self.term_bias_reg_weight * torch.sum(raw_term_bias ** 2)
        else:
            # 修复：在不使用术语注意力时，创建一个零向量以保持维度一致
            batch_size = pooled_output.size(0)
            device = pooled_output.device
            zero_features = torch.zeros(batch_size, self.term_feature_projection.out_features, device=device)
            combined_features = torch.cat([pooled_output, zero_features], dim=-1)

        # 调试信息：检查维度
        if self.training and hasattr(self, 'debug_count') and self.debug_count < 5:
            logger.info(f"combined_features shape: {combined_features.shape}, classifier input dim: {self.classifier_input_dim}")
            if not hasattr(self, 'debug_count'):
                self.debug_count = 0
            self.debug_count += 1

        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
            loss = main_loss + internal_bias_reg + external_term_bias_reg

        return loss, logits, combined_features

#-----------------------------------------------------------------------------------------------------------------------------------词句段
# === 修正版 TermSentBERT ===
class TermSentBERT(nn.Module):
    """
    v1: 在 TermAttentionEnhancedBERT 的设计思想上实现的句级拼接版本。
    注意：
      - 该类内部直接使用 self.bert 来获取 pooled_output/sequence_output；
      - term attention 采用与 TermAttentionEnhancedBERT 同样的 term_attention_bias 计算流程；
      - 句向量从 raw_texts 用 tokenizer 分句并取每句 pooler_output 平均获得（with torch.no_grad）。
      - forward 接收 raw_texts 与 tokenizer（collate_fn 需提供 raw_texts）。
    """
    def __init__(self, pretrained_name='bert-base-chinese', dropout_prob=0.2,
                 term_bias_scale=2.0, term_bias_reg_weight=5e-6,
                 internal_bias_scale=0.4, internal_bias_reg_weight=5e-6):
        super().__init__()
        # reuse TermAttentionEnhancedBERT internals where possible
        self.bert = BertModel.from_pretrained(pretrained_name)
        replace_bert_self_attention_with_improved(self.bert,
                                                 bias_scale=internal_bias_scale,
                                                 bias_reg_weight=internal_bias_reg_weight)
        hidden = self.bert.config.hidden_size

        # term attention projection
        self.term_attention_bias = nn.Linear(hidden, 1)
        self.term_feature_projection = nn.Linear(hidden, hidden // 4)

        # sentence projection
        self.sentence_projection = nn.Linear(hidden, hidden // 4)

        # classifier dimension: pooled + term + sentence
        self.classifier_input_dim = hidden + (hidden // 4) + (hidden // 4)
        self.classifier = nn.Linear(self.classifier_input_dim, 2)

        self.dropout = nn.Dropout(dropout_prob)
        self.term_bias_scale = term_bias_scale
        self.term_bias_reg_weight = term_bias_reg_weight

        logger.info(f"[TermSentBERT] init classifier_input_dim={self.classifier_input_dim}")

    def _compute_internal_bias_reg(self, device, sequence_output):
        internal_bias_reg = torch.tensor(0.0, device=device, dtype=sequence_output.dtype)
        for layer in self.bert.encoder.layer:
            try:
                v = layer.attention.self._last_bias_reg
                if isinstance(v, torch.Tensor):
                    internal_bias_reg = internal_bias_reg + v.to(device)
            except Exception:
                pass
        return internal_bias_reg

    def _compute_term_features(self, sequence_output, attention_mask, term_mask):
        # term bias external module (same idea as earlier)
        raw_term_bias = self.term_attention_bias(sequence_output).squeeze(-1)  # (B, L)
        scaled_term_bias = torch.tanh(raw_term_bias) * self.term_bias_scale
        base_mask_logits = (1.0 - attention_mask.float()) * -10000.0
        if term_mask is not None:
            term_bias = torch.where(term_mask.bool(), scaled_term_bias, -5.0 * torch.ones_like(scaled_term_bias))
        else:
            term_bias = -5.0 * torch.ones_like(scaled_term_bias)
        enhanced_attention = base_mask_logits + term_bias
        term_weights = torch.softmax(enhanced_attention, dim=-1)  # (B, L)
        term_enhanced_output = torch.sum(sequence_output * term_weights.unsqueeze(-1), dim=1)  # (B, hidden)
        term_features = self.term_feature_projection(term_enhanced_output)  # (B, hidden//4)
        external_term_bias_reg = self.term_bias_reg_weight * torch.sum(raw_term_bias ** 2)
        return term_features, external_term_bias_reg

    def _get_sentence_level_vecs(self, raw_texts, tokenizer, device, max_sentences=5, max_sent_len=128):
        """
        简易句级向量提取：对每篇文本分句（正则），对每句求 pooler_output，再对句向量平均。
        使用 no_grad() 来避免额外反向图（MVP 实验常用）。
        """
        import re
        sent_vecs = []
        for text in raw_texts:
            sents = re.split(r'[。；;？！?!]', text)
            sents = [s.strip() for s in sents if s.strip()]
            if len(sents) == 0:
                sents = [text[:max_sent_len]]
            cur_vecs = []
            for sent in sents[:max_sentences]:
                enc = tokenizer(sent, return_tensors='pt', truncation=True, max_length=max_sent_len).to(device)
                with torch.no_grad():
                    out = self.bert(**enc, return_dict=True)
                # pooler_output shape (1, hidden)
                cur_vecs.append(out.pooler_output.squeeze(0))
            if len(cur_vecs) == 0:
                cur_mean = torch.zeros(self.bert.config.hidden_size, device=device)
            else:
                cur_mean = torch.stack(cur_vecs, dim=0).mean(dim=0)
            sent_vecs.append(cur_mean)
        return torch.stack(sent_vecs, dim=0)  # (B, hidden)

    def forward(self, input_ids, attention_mask, token_type_ids=None, term_mask=None,
                labels=None, class_weights=None, raw_texts=None, tokenizer=None):
        device = input_ids.device
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        pooled_output = outputs.pooler_output  # (B, hidden)
        sequence_output = outputs.last_hidden_state  # (B, L, hidden)

        internal_bias_reg = self._compute_internal_bias_reg(device, sequence_output)

        term_features, external_term_bias_reg = self._compute_term_features(sequence_output, attention_mask, term_mask)

        # sentence-level features (if provided)
        if raw_texts is not None and tokenizer is not None:
            with torch.no_grad():
                sent_vec = self._get_sentence_level_vecs(raw_texts, tokenizer, device)
            sent_features = self.sentence_projection(sent_vec)
        else:
            sent_features = torch.zeros(pooled_output.size(0), self.sentence_projection.out_features, device=device)

        # final fusion and classification
        fusion = torch.cat([pooled_output, term_features, sent_features], dim=-1)  # (B, classifier_input_dim)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
            # combine bias regs (internal + external)
            loss = main_loss + internal_bias_reg + external_term_bias_reg

        return loss, logits, fusion


# === 修正版 TermSentParaBERT ===
class TermSentParaBERT(TermSentBERT):
    """
    v2: 在 v1 基础上再加入段级（document-level）向量拼接。
    - 段向量以整篇摘要的 pooler_output 为代表（可以与句级平均互补）。
    - 同样使用 with torch.no_grad() 获取 doc pooler（MVP 简化策略）。
    """
    def __init__(self, pretrained_name='bert-base-chinese', dropout_prob=0.2,
                 term_bias_scale=2.0, term_bias_reg_weight=5e-6,
                 internal_bias_scale=0.4, internal_bias_reg_weight=5e-6):
        super().__init__(pretrained_name, dropout_prob,
                         term_bias_scale, term_bias_reg_weight,
                         internal_bias_scale, internal_bias_reg_weight)
        hidden = self.bert.config.hidden_size
        # add doc projection
        self.doc_projection = nn.Linear(hidden, hidden // 4)
        # update classifier to take one more vector
        self.classifier_input_dim = hidden + (hidden // 4) + (hidden // 4) + (hidden // 4)
        self.classifier = nn.Linear(self.classifier_input_dim, 2)
        logger.info(f"[TermSentParaBERT] init classifier_input_dim={self.classifier_input_dim}")

    def _get_doc_vecs(self, raw_texts, tokenizer, device, max_len=512):
        """对整篇文本取 pooler_output 作为 doc 向量（no_grad）"""
        doc_vecs = []
        for text in raw_texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len).to(device)
            with torch.no_grad():
                out = self.bert(**enc, return_dict=True)
            doc_vecs.append(out.pooler_output.squeeze(0))
        return torch.stack(doc_vecs, dim=0)  # (B, hidden)

    def forward(self, input_ids, attention_mask, token_type_ids=None, term_mask=None,
                labels=None, class_weights=None, raw_texts=None, tokenizer=None):
        # reuse TermSentBERT forward for pooled/term/sent features and bias regs
        device = input_ids.device
        # call parent to compute pooled/term/sent but parent returns loss, logits, fusion for v1 shape;
        # we re-implement here to capture internal regs properly (so copy steps)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state

        internal_bias_reg = self._compute_internal_bias_reg(device, sequence_output)
        term_features, external_term_bias_reg = self._compute_term_features(sequence_output, attention_mask, term_mask)

        # sentence-level
        if raw_texts is not None and tokenizer is not None:
            with torch.no_grad():
                sent_vec = self._get_sentence_level_vecs(raw_texts, tokenizer, device)
            sent_features = self.sentence_projection(sent_vec)
        else:
            sent_features = torch.zeros(pooled_output.size(0), self.sentence_projection.out_features, device=device)

        # doc-level
        if raw_texts is not None and tokenizer is not None:
            with torch.no_grad():
                doc_vec = self._get_doc_vecs(raw_texts, tokenizer, device)
            doc_features = self.doc_projection(doc_vec)
        else:
            doc_features = torch.zeros(pooled_output.size(0), self.doc_projection.out_features, device=device)

        # fusion and classify
        fusion = torch.cat([pooled_output, term_features, sent_features, doc_features], dim=-1)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
            loss = main_loss + internal_bias_reg + external_term_bias_reg

        return loss, logits, fusion

# === v3: TermSentParaFusionBERT (cross-attention + GMU 门控融合) ===
# === v3: TermSentParaFusionBERT (cross-attention + GMU 门控融合) ===
class TermSentParaFusionBERT(TermSentParaBERT):
    """
    v3 融合：
    - cross-attn (token<-sentence, token<-paragraph)
    - GMU门控：对 token / sent_attn / para_attn 进行加权融合
    - 保留词级术语偏置、句向量、段向量的全部逻辑
    """
    def __init__(self, pretrained_name='bert-base-chinese', dropout_prob=0.2,
                 term_bias_scale=2.0, term_bias_reg_weight=5e-6,
                 internal_bias_scale=0.4, internal_bias_reg_weight=5e-6):
        super().__init__(pretrained_name, dropout_prob,
                         term_bias_scale, term_bias_reg_weight,
                         internal_bias_scale, internal_bias_reg_weight)

        hidden = self.bert.config.hidden_size

        # === 修复：添加维度转换层 ===
        self.sent_dim_adapter = nn.Linear(hidden // 4, hidden)
        self.para_dim_adapter = nn.Linear(hidden // 4, hidden)

        # === New: Cross Attention 层 ===
        self.cross_attn_token_sent = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True
        )
        self.cross_attn_token_para = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True
        )

        # GMU 门控
        self.gate_linear = nn.Linear(hidden * 3, 3)

        # 更新分类器
        self.classifier = nn.Linear(hidden, 2)
        logger.info("[v3] Initialized TermSentParaFusionBERT (cross-attn + GMU)")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, term_mask=None,
                labels=None, class_weights=None, raw_texts=None, tokenizer=None):

        # Step1: 复用 v2 的 pooled_output / sequence_output / term / sent / para
        device = input_ids.device
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state

        internal_bias_reg = self._compute_internal_bias_reg(device, sequence_output)
        term_features, external_term_bias_reg = self._compute_term_features(sequence_output, attention_mask, term_mask)

        # sentence-level (v2)
        if raw_texts is not None:
            with torch.no_grad():
                sent_vec = self._get_sentence_level_vecs(raw_texts, tokenizer, device)
            sent_vec = self.sentence_projection(sent_vec)  # (B, hidden//4)
        else:
            sent_vec = torch.zeros_like(term_features)

        # paragraph-level (v2)
        if raw_texts is not None:
            with torch.no_grad():
                doc_vec = self._get_doc_vecs(raw_texts, tokenizer, device)
            doc_vec = self.doc_projection(doc_vec)  # (B, hidden//4)
        else:
            doc_vec = torch.zeros_like(term_features)

        # ======== v3 关键：Cross-attention ========
        token_q = pooled_output.unsqueeze(1)          # (B,1,H=768)
        
        # 修复：将维度从 hidden//4 转换回 hidden
        sent_kv = self.sent_dim_adapter(sent_vec).unsqueeze(1)  # (B,1,H=768)
        para_kv = self.para_dim_adapter(doc_vec).unsqueeze(1)   # (B,1,H=768)

        attn_token_sent, _ = self.cross_attn_token_sent(token_q, sent_kv, sent_kv)
        attn_token_para, _ = self.cross_attn_token_para(token_q, para_kv, para_kv)

        attn_token_sent = attn_token_sent.squeeze(1)
        attn_token_para = attn_token_para.squeeze(1)

        # ======== GMU门控融合 ========
        gmu_input = torch.cat([pooled_output, attn_token_sent, attn_token_para], dim=-1)
        gates = torch.softmax(self.gate_linear(gmu_input), dim=-1)  # (B,3)

        g_token = gates[:, :1]
        g_sent = gates[:, 1:2]
        g_para = gates[:, 2:3]

        fused_rep = (
            g_token * pooled_output +
            g_sent * attn_token_sent +
            g_para * attn_token_para
        )

        fused_rep = self.dropout(fused_rep)
        logits = self.classifier(fused_rep)

        # ===== Loss =====
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
            loss = main_loss + internal_bias_reg + external_term_bias_reg

        return loss, logits, fused_rep

class TermSentParaFinalBERT(TermSentParaFusionBERT):
    """
    回归 V2 架构核心，融合 V4 交互机制：
    1. Single Encoder: 统一使用 self.bert 提取所有特征（消除语义割裂）。
    2. Cross-Attention: 替代 V2 的简单 Pooling，进行深度交互。
    3. Concat Fusion: 替代 GMU，强制融合。
    """
    def __init__(self, pretrained_name='bert-base-chinese', dropout_prob=0.1, # <--- 降回 0.1，防止欠拟合
                 term_bias_scale=2.0, term_bias_reg_weight=5e-6,
                 internal_bias_scale=0.4, internal_bias_reg_weight=5e-6):
        super(TermSentParaFusionBERT, self).__init__(pretrained_name, dropout_prob,
                                                     term_bias_scale, term_bias_reg_weight,
                                                     internal_bias_scale, internal_bias_reg_weight)
        
        hidden = self.bert.config.hidden_size
        
        # === 移除 Frozen Encoder，只用 self.bert ===

        # === Cross-Attention (Bottleneck) ===
        # 保持降维设计，节省参数
        self.attn_dim = 256  
        self.proj_token_q = nn.Linear(hidden, self.attn_dim)
        self.proj_sent_kv = nn.Linear(hidden, self.attn_dim)
        self.proj_para_kv = nn.Linear(hidden, self.attn_dim)

        self.cross_sent = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=4, batch_first=True)
        self.cross_para = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=4, batch_first=True)

        self.proj_sent_out = nn.Linear(self.attn_dim, hidden)
        self.proj_para_out = nn.Linear(self.attn_dim, hidden)

        # === Concat Fusion ===
        # LayerNorm 很重要，稳定拼接特征
        self.fusion_norm = nn.LayerNorm(hidden * 3)
        self.classifier = nn.Linear(hidden * 3, 2)
        
        logger.info(f"[Final Integrated] Single-Encoder + CrossAttn + Concat. Dropout={dropout_prob}")

    def _get_dynamic_features(self, raw_texts, tokenizer, device, max_sent_len=64):
        """
        使用 self.bert (no_grad) 提取特征。
        虽然叫 no_grad，但因为 self.bert 在训练中会更新，
        所以这里的特征是 '动态' 适应任务的，优于 Frozen。
        """
        import re
        
        # 1. Doc Level
        enc_doc = tokenizer(raw_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            # 使用 self.bert 而不是 frozen_encoder
            out_doc = self.bert(**enc_doc, return_dict=True)
        doc_batch_vecs = out_doc.pooler_output

        # 2. Sentence Level (Vectorized)
        all_sents = []
        sent_counts = []
        
        for text in raw_texts:
            sents = re.split(r'[。；;？！?!]', text)
            sents = [s.strip() for s in sents if s.strip()]
            if not sents: sents = [text[:max_sent_len]]
            sents = sents[:5]
            all_sents.extend(sents)
            sent_counts.append(len(sents))
            
        enc_sents = tokenizer(all_sents, return_tensors='pt', padding=True, truncation=True, max_length=max_sent_len).to(device)
        
        with torch.no_grad():
            # 使用 self.bert
            out_sents = self.bert(**enc_sents, return_dict=True)
        
        all_sent_vecs = out_sents.pooler_output
        
        # Restore batch
        final_sent_vecs = []
        cursor = 0
        hidden_size = all_sent_vecs.size(-1)
        for count in sent_counts:
            if count == 0:
                final_sent_vecs.append(torch.zeros(hidden_size, device=device))
            else:
                final_sent_vecs.append(all_sent_vecs[cursor : cursor + count].mean(dim=0))
                cursor += count
        sent_batch_vecs = torch.stack(final_sent_vecs)

        return sent_batch_vecs, doc_batch_vecs

    def forward(self, input_ids, attention_mask, token_type_ids=None, term_mask=None,
                labels=None, class_weights=None, raw_texts=None, tokenizer=None):
        device = input_ids.device

        # 1. Main BERT Forward (Training)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        pooled = out.pooler_output
        seq = out.last_hidden_state
        internal_bias_reg = self._compute_internal_bias_reg(device, seq)
        _, external_term_bias_reg = self._compute_term_features(seq, attention_mask, term_mask)

        # 2. Dynamic Feature Extraction
        # 使用当前正在微调的 BERT 提取辅助特征
        if raw_texts is not None and tokenizer is not None:
            sent_vec, para_vec = self._get_dynamic_features(raw_texts, tokenizer, device)
        else:
            sent_vec = torch.zeros_like(pooled)
            para_vec = torch.zeros_like(pooled)
            
        # 3. Cross Attention
        q_token = self.proj_token_q(pooled).unsqueeze(1)
        kv_sent = self.proj_sent_kv(sent_vec).unsqueeze(1)
        kv_para = self.proj_para_kv(para_vec).unsqueeze(1)

        sent_attn_out, _ = self.cross_sent(q_token, kv_sent, kv_sent)
        para_attn_out, _ = self.cross_para(q_token, kv_para, kv_para)

        sent_feat = self.proj_sent_out(sent_attn_out.squeeze(1))
        para_feat = self.proj_para_out(para_attn_out.squeeze(1))

        # 4. Concat Fusion
        fused = torch.cat([pooled, sent_feat, para_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            main_loss = loss_fn(logits, labels)
            loss = main_loss + internal_bias_reg + external_term_bias_reg

        return loss, logits, fused
# ---------------------------
# Training / Eval functions
# ---------------------------
# ---------------------------
# Training / Eval functions
# ---------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1, class_weights=None, epoch=None, tokenizer=None):
    model.train()
    if epoch is not None and hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    
    total_loss = 0.0
    optimizer.zero_grad()
    max_grad_norm = 1.0

    for i, batch in enumerate(tqdm(dataloader, desc=f"训练中 (Epoch {epoch})", ncols=100)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        term_mask = batch['term_mask'].to(device)
        labels = batch['labels'].to(device)
        raw_texts = batch.get('raw_texts', None)

        loss, logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            term_mask=term_mask,
            labels=labels,
            class_weights=class_weights,
            raw_texts=raw_texts,
            tokenizer=tokenizer
        )

        if loss is None:
            continue

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


# def evaluate(model, dataloader, device, tokenizer=None):
#     model.eval()
#     preds_all, labels_all, ids_all, probs_all = [], [], [], []
#     gates_all = []   # ★ NEW ★


#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="评估中", ncols=100):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             token_type_ids = batch['token_type_ids'].to(device)
#             term_mask = batch['term_mask'].to(device)
#             labels = batch['labels']
#             ids = batch['ids']
#             raw_texts = batch.get('raw_texts', None)

#             _, logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 term_mask=term_mask,
#                 labels=None,
#                 raw_texts=raw_texts,
#                 tokenizer=tokenizer
#             )
#             probs = torch.softmax(logits, dim=-1).cpu().numpy()
#             preds = np.argmax(probs, axis=1).tolist()

#             labels_cpu = labels.cpu().numpy().tolist()
#             preds_all.extend(preds)
#             labels_all.extend(labels_cpu)
#             ids_all.extend(ids)
#             probs_all.extend(probs.tolist())

#     valid_pairs = [(l, p) for l, p in zip(labels_all, preds_all) if l != -1]
#     if valid_pairs:
#         lbls = [l for l, _ in valid_pairs]
#         prs = [p for _, p in valid_pairs]
#         acc = accuracy_score(lbls, prs)
#         f1 = f1_score(lbls, prs, average='macro')
#     else:
#         acc, f1 = 0.0, 0.0

#     return acc, f1, ids_all, labels_all, preds_all, probs_all

def evaluate(model, dataloader, device, tokenizer=None):
    model.eval()
    preds_all, labels_all, ids_all, probs_all = [], [], [], []
    gates_all = []   # ★ NEW：收集 GMU 门控

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            term_mask = batch['term_mask'].to(device)
            labels = batch['labels']
            ids = batch['ids']
            raw_texts = batch.get('raw_texts', None)

            _, logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                term_mask=term_mask,
                labels=None,
                raw_texts=raw_texts,
                tokenizer=tokenizer
            )

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()

            labels_cpu = labels.cpu().numpy().tolist()
            preds_all.extend(preds)
            labels_all.extend(labels_cpu)
            ids_all.extend(ids)
            probs_all.extend(probs.tolist())

            # ★ NEW：如果模型保存了 last_gates，则收集
            if hasattr(model, "last_gates"):
                gates_all.append(model.last_gates)

    # 计算 Acc / F1
    valid_pairs = [(l, p) for l, p in zip(labels_all, preds_all) if l != -1]
    if valid_pairs:
        lbls = [x[0] for x in valid_pairs]
        prs = [x[1] for x in valid_pairs]
        acc = accuracy_score(lbls, prs)
        f1 = f1_score(lbls, prs, average='macro')
    else:
        acc, f1 = 0.0, 0.0

    return acc, f1, ids_all, labels_all, preds_all, probs_all, gates_all

# ---------------------------
# Main function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_train', type=str, default='/root/autodl-tmp/MVPproject/train.json')
    parser.add_argument('--original_dev', type=str, default='/root/autodl-tmp/MVPproject/dev.json')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=112233)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/MVPproject/output_attention_optimized_cijuduan')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--model_version', type=str, default='base',
                        choices=['base', 'v1', 'v2', 'v3', 'v4'])
    args = parser.parse_args()

    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    set_seed(GLOBAL_SEED)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")

    # ===== 数据切分 =====
    with open(args.original_train, 'r', encoding='utf-8') as f:
        all_train = [json.loads(line.strip()) for line in f if line.strip()]
    labels = [int(it.get('label', 0)) for it in all_train]

    if len(set(labels)) > 1:
        from sklearn.model_selection import train_test_split
        train_data, dev_data = train_test_split(all_train, test_size=0.2,
                                                random_state=args.seed, stratify=labels)
    else:
        train_data, dev_data = train_test_split(all_train, test_size=0.2,
                                                random_state=args.seed)

    with open(args.original_dev, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]

    new_data_dir = os.path.join(args.output_dir, 'split_data')
    os.makedirs(new_data_dir, exist_ok=True)

    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as fw:
            for it in data:
                fw.write(json.dumps(it, ensure_ascii=False) + '\n')

    train_path = os.path.join(new_data_dir, 'train.json')
    dev_path = os.path.join(new_data_dir, 'dev.json')
    test_path = os.path.join(new_data_dir, 'test.json')
    save_jsonl(train_data, train_path)
    save_jsonl(dev_data, dev_path)
    save_jsonl(test_data, test_path)

    # ===== 初始化 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    train_dataset = AbstKeywordDataset(train_path)
    dev_dataset = AbstKeywordDataset(dev_path)
    test_dataset = AbstKeywordDataset(test_path)

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_len)

    def collate_with_text(batch):
        data = collate_fn(batch)
        data['raw_texts'] = [it['abst'] for it in batch]
        return data

    train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_with_text,
    worker_init_fn=worker_init_fn,
    num_workers=0
)

    dev_loader = DataLoader(
    dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_with_text,
    num_workers=0
)

    test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_with_text,
    num_workers=0
)

    # ===== 模型选择 =====
    if args.model_version == "v1":
        model = TermSentBERT(pretrained_name=args.model_name).to(device)
    elif args.model_version == "v2":
        model = TermSentParaBERT(pretrained_name=args.model_name).to(device)
    elif args.model_version == "v3":
        model = TermSentParaFusionBERT(pretrained_name=args.model_name).to(device)
    
    # === 修改这里 ===
    elif args.model_version == "v4":
        # 注意：dropout_prob 设为 0.1 (默认值)
        model = TermSentParaFinalBERT(pretrained_name=args.model_name, dropout_prob=0.1).to(device)
    
    else:
        model = TermAttentionEnhancedBERT(pretrained_name=args.model_name).to(device)

   # ===== 优化器 (回归简单版) =====
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # ===== 训练 =====
    best_dev_f1 = -1
    for epoch in range(1, args.epochs + 1):
        model.set_epoch(epoch) if hasattr(model, "set_epoch") else None

        train_loss = train_epoch(model, train_loader, optimizer, None, device,
                                 args.gradient_accumulation_steps, None, epoch, tokenizer)
        logger.info(f"训练损失: {train_loss:.5f}")

        dev_acc, dev_f1, *_ = evaluate(model, dev_loader, device, tokenizer)
        logger.info(f"验证集 Acc={dev_acc:.4f}, F1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("更新最佳模型")

    # ===== 测试集 =====
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    test_acc, test_f1, ids, labels, preds, probs, gates = evaluate(model, test_loader, device, tokenizer)

    logger.info(f"测试 Acc={test_acc:.4f}, F1={test_f1:.4f}")

    save_predictions_csv(os.path.join(args.output_dir, "predictions.csv"), ids, labels, preds, probs)
    save_gates_csv(os.path.join(args.output_dir, "gmu_gates.csv"), gates)

    logger.info("预测结果与 GMU 门控已保存。")



if __name__ == "__main__":
    main()
