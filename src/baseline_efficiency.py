#!/usr/bin/env python3
# coding: utf-8
"""
针对RTX 4090D 24GB完整优化版本
任务：根据(摘要, 关键词列表)预测关键词是否全部真实有效（label 0/1）
数据集路径：/root/autodl-tmp/MVPproject/{train,dev,test}.json
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 解决确定性计算问题
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 抑制警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import json
import random
import argparse
import logging
import time
import sys
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModel

# 日志配置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------------------------
# 可复现性设置（固定随机种子）
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
# 数据集加载（适配JSON格式与AutoDL路径）
# ---------------------------
class AbstKeywordDataset(Dataset):
    def __init__(self, json_path: str):
        assert os.path.exists(json_path), f"数据集不存在：{json_path}"
        items = []
        
        logger.info(f"正在加载文件: {json_path}")
        
        # 直接按JSON Lines处理（根据您的文件格式）
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
                    # 解析单个样本
                    if self._parse_item(obj, lineno, items):
                        success_count += 1
                except Exception as e:
                    logger.warning(f"行 {lineno} 解析失败: {e}")
                    continue
        
        if len(items) == 0:
            raise RuntimeError(f"{json_path} 中未找到有效样本（共处理 {total_lines} 行，成功 {success_count} 行）")
        
        logger.info(f"加载 {json_path} 成功，共 {len(items)} 个有效样本（总计 {total_lines} 行）")
        self.items = items

    def _parse_item(self, obj: Dict, idx: int, items: List[Dict]) -> bool:
        """解析单个样本，返回是否成功"""
        try:
            # 检查必需字段
            required = ['id', 'abst', 'keyword', 'label']
            if not all(k in obj for k in required):
                logger.warning(f"样本 {idx} 缺少必需字段，现有字段: {list(obj.keys())}")
                return False
            
            # 处理摘要
            abst = str(obj['abst']).strip()
            if not abst:
                logger.warning(f"样本 {idx} 摘要为空")
                return False
            
            # 处理关键词
            kw = obj['keyword']
            if isinstance(kw, str):
                # 尝试多种分隔符
                for sep in [';', ',', '，', '；', ' ']:
                    if sep in kw:
                        kw_list = [k.strip() for k in kw.split(sep) if k.strip()]
                        if kw_list:
                            break
                else:
                    # 如果没有分隔符，直接使用整个字符串
                    kw_list = [kw.strip()] if kw.strip() else []
            elif isinstance(kw, list):
                kw_list = [str(x).strip() for x in kw if str(x).strip()]
            else:
                kw_list = []
            
            # 处理标签
            try:
                label_val = obj['label']
                # 更灵活地处理标签值
                if isinstance(label_val, (int, float)):
                    label = int(label_val)
                else:
                    label_str = str(label_val).strip()
                    label = int(label_str) if label_str.isdigit() else 1 if label_str.lower() in ['true', 'yes', '有效'] else 0
                
                if label not in (0, 1):
                    logger.warning(f"样本 {idx} 标签值 {label} 不是0或1，自动映射到 {1 if label > 0 else 0}")
                    label = 1 if label > 0 else 0
                    
            except Exception as e:
                logger.warning(f"样本 {idx} 标签解析失败: {e}，使用默认标签0")
                label = 0
            
            # 添加到样本列表
            items.append({
                'id': obj['id'],
                'abst': abst,
                'keywords': kw_list,
                'label': label
            })
            return True
            
        except Exception as e:
            logger.warning(f"样本 {idx} 解析异常: {e}")
            return False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# ---------------------------
# 数据集重新划分函数
# ---------------------------
def split_dataset(args):
    """重新划分数据集：将train.json分成train和dev，将dev.json作为test"""
    logger.info("开始重新划分数据集...")
    
    # 读取原始训练集
    original_train_data = []
    with open(args.original_train, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                original_train_data.append(json.loads(line))
    
    logger.info(f"原始训练集样本数: {len(original_train_data)}")
    
    # 检查标签分布
    labels = [item['label'] for item in original_train_data]
    from collections import Counter
    label_dist = Counter(labels)
    logger.info(f"原始训练集标签分布: {dict(label_dist)}")
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_data, dev_data = train_test_split(
        original_train_data,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels  # 保持标签分布
    )
    
    # 读取原始开发集作为测试集
    test_data = []
    with open(args.original_dev, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
    logger.info(f"新训练集样本数: {len(train_data)}")
    logger.info(f"新验证集样本数: {len(dev_data)}")
    logger.info(f"新测试集样本数: {len(test_data)}")
    
    # 创建新的数据集目录
    new_data_dir = os.path.join(args.output_dir, 'split_data')
    os.makedirs(new_data_dir, exist_ok=True)
    
    # 保存新的数据集
    def save_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    new_train_path = os.path.join(new_data_dir, 'train.json')
    new_dev_path = os.path.join(new_data_dir, 'dev.json')
    new_test_path = os.path.join(new_data_dir, 'test.json')
    
    save_jsonl(train_data, new_train_path)
    save_jsonl(dev_data, new_dev_path)
    save_jsonl(test_data, new_test_path)
    
    logger.info(f"新数据集已保存到: {new_data_dir}")
    
    # 检查新数据集的标签分布
    train_labels = [item['label'] for item in train_data]
    dev_labels = [item['label'] for item in dev_data]
    test_labels = [item['label'] for item in test_data]
    
    logger.info(f"新训练集标签分布: {dict(Counter(train_labels))}")
    logger.info(f"新验证集标签分布: {dict(Counter(dev_labels))}")
    logger.info(f"新测试集标签分布: {dict(Counter(test_labels))}")
    
    return new_train_path, new_dev_path, new_test_path

# ---------------------------
# 数据批处理（适配BERT输入）
# ---------------------------
def make_collate_fn(tokenizer: BertTokenizer, max_length: int = 1024):
    def collate(batch: List[Dict]):
        texts_a = [it['abst'] for it in batch]
        texts_b = []
        ids = [it['id'] for it in batch]
        labels = [it['label'] for it in batch]
        # 关键词拼接（用中文分号，减少与关键词内符号冲突）
        for it in batch:
            if it['keywords']:
                texts_b.append('；'.join(it['keywords']))
            else:
                texts_b.append('')
        # BERT编码
        enc = tokenizer(
            text=texts_a,
            text_pair=texts_b,
            padding='longest',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'ids': ids,
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'token_type_ids': enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    return collate

# ---------------------------
# 模型定义（保持原逻辑）
# ---------------------------
class TransformerForBinaryClassification(nn.Module):
    def __init__(self, pretrained_name, dropout_prob=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits, pooled


# ---------------------------
# 训练与评估函数（添加梯度累积）
# ---------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn=None, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="训练中", ncols=100)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
        labels = batch['labels'].to(device)
        
        _, logits, _ = model(input_ids, attention_mask, token_type_ids)
        if loss_fn:
            loss = loss_fn(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds_all, labels_all, ids_all, probs_all = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            _, logits, _ = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            # 收集结果
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy().tolist())
            ids_all.extend(batch['ids'])
            probs_all.extend(probs.tolist())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return acc, f1, ids_all, labels_all, preds_all, probs_all

# ---------------------------
# 工具函数（结果保存）
# ---------------------------
def save_predictions_csv(outfile, ids, labels, preds, probs):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', '真实标签', '预测标签', '概率_0', '概率_1'])
        for i, t, p, pr in zip(ids, labels, preds, probs):
            writer.writerow([i, t, p, round(pr[0], 4), round(pr[1], 4)])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_efficiency_metrics(save_path, metrics: dict):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def measure_inference_latency_baseline(model, dataloader, device, warmup_steps=10, measure_steps=50):
    """
    统一统计 baseline 模型推理时延。
    返回: (ms_per_sample, samples_per_sec)
    """
    model.eval()
    cached_batches = []
    for i, batch in enumerate(dataloader):
        cached_batches.append(batch)
        if len(cached_batches) >= measure_steps:
            break

    if len(cached_batches) == 0:
        return None, None

    with torch.no_grad():
        for batch in cached_batches[:min(warmup_steps, len(cached_batches))]:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_samples = 0
    start = time.time()
    with torch.no_grad():
        for batch in cached_batches:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )
            total_samples += input_ids.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    ms_per_sample = elapsed * 1000.0 / max(total_samples, 1)
    samples_per_sec = total_samples / max(elapsed, 1e-8)
    return ms_per_sample, samples_per_sec

# ---------------------------
# 文本长度分析函数
# ---------------------------
# ---------------------------
# 文本长度分析函数 (已修复)
# ---------------------------
def analyze_text_lengths(dataset, tokenizer, max_length=1024, sample_size=100):
    """分析文本长度分布"""
    logger.info("正在分析文本长度分布...")
    
    lengths = []
    truncated_count = 0
    
    # 限制采样数量，防止列表越界
    actual_sample_size = min(sample_size, len(dataset))
    
    for i, item in enumerate(dataset.items[:actual_sample_size]):
        text_a = item['abst']
        text_b = '；'.join(item['keywords']) if item['keywords'] else ''
        
        # tokenize但不截断
        # 修复点：不使用 return_length=True，直接计算 input_ids 的长度
        encoding = tokenizer(
            text_a, 
            text_b, 
            truncation=False,
            verbose=False
        )
        
        # 直接获取 input_ids 的长度，这一定是一个整数
        length = len(encoding['input_ids'])
        
        lengths.append(length)
        
        if length > max_length:
            truncated_count += 1
    
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        max_observed = max(lengths)
        p95 = sorted(lengths)[int(0.95 * len(lengths))]
        
        logger.info(f"文本长度分析 (基于{len(lengths)}个样本):")
        logger.info(f"  平均长度: {avg_length:.1f} tokens")
        logger.info(f"  最大长度: {max_observed} tokens")
        logger.info(f"  95%分位数: {p95} tokens")
        logger.info(f"  超过{max_length}的样本: {truncated_count}/{len(lengths)} ({truncated_count/len(lengths)*100:.1f}%)")
        
        if truncated_count / len(lengths) > 0.1:
            logger.warning(f"⚠️  超过10%的数据被截断，建议检查max_length设置")
        else:
            logger.info("✓ 数据截断情况良好")
    
    return truncated_count / len(lengths) if lengths else 0

# ---------------------------
# 主函数（适配你的数据集路径）
# ---------------------------
def main():
    # 解析命令行参数（针对RTX 4090D 24GB优化）
    parser = argparse.ArgumentParser()
    # 原始数据集路径
    parser.add_argument('--original_train', type=str, 
                        default='/root/autodl-tmp/MVPproject/train.json',
                        help='原始训练集路径')
    parser.add_argument('--original_dev', type=str, 
                        default='/root/autodl-tmp/MVPproject/dev.json',
                        help='原始验证集路径（将用作新测试集）')
    # 针对RTX 4090D 24GB优化的参数
    parser.add_argument(
    '--model_name',
    type=str,
    default='hfl/chinese-roberta-wwm-ext'
)

    parser.add_argument('--batch_size', type=int, default=8, help='批次大小（针对4090D优化）')
    parser.add_argument('--max_len', type=int, default=512, help='文本最大长度（针对4090D优化）')
    parser.add_argument('--epochs', type=int, default=4, help='训练轮数（增加训练轮数）')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--seed', type=int, default=666, help='随机种子')
    parser.add_argument('--output_dir', type=str, 
                        default='/root/autodl-tmp/MVPproject/output',
                        help='结果保存路径')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='梯度累积步数（如需更大max_length可增加此值）')
    args = parser.parse_args()

    # 全局种子设置
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    set_seed(GLOBAL_SEED)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"结果将保存到：{args.output_dir}")

    # 重新划分数据集
    new_train_path, new_dev_path, new_test_path = split_dataset(args)

    # 保存配置信息（包括新的数据集路径）
    config = vars(args)
    config['new_train_path'] = new_train_path
    config['new_dev_path'] = new_dev_path
    config['new_test_path'] = new_test_path
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")
    if torch.cuda.is_available():
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载分词器并保存
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    # 加载重新划分后的数据集
    logger.info("开始加载重新划分后的数据集...")
    train_dataset = AbstKeywordDataset(new_train_path)
    dev_dataset = AbstKeywordDataset(new_dev_path)
    test_dataset = AbstKeywordDataset(new_test_path)

    # 分析文本长度
    truncation_rate = analyze_text_lengths(train_dataset, tokenizer, args.max_len)

    # 计算类别权重（处理不平衡）
    labels = [item['label'] for item in train_dataset.items]
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = None
    if len(classes) == 2:
        inv_freq = 1.0 / counts
        inv_freq = inv_freq / inv_freq.sum() * len(classes)
        class_weights = torch.tensor(inv_freq, dtype=torch.float).to(device)
        logger.info(f"类别权重（处理不平衡）：{class_weights.cpu().numpy()}")
    else:
        logger.warning("训练集类别不足2类，可能影响模型效果")

    # 数据加载器
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_len)
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start_time = time.time()
    epoch_times = []

    # 初始化模型
    model = TransformerForBinaryClassification(args.model_name).to(device)
    param_count = count_parameters(model)
    logger.info(f"模型参数量: {param_count / 1e6:.2f} M")


    # 优化器与调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 加权损失函数（若需要）
    loss_fn = None
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 训练循环
    best_dev_f1 = -1.0
    best_epoch = -1
    logger.info("开始训练...")
    logger.info(f"训练配置: batch_size={args.batch_size}, max_len={args.max_len}, epochs={args.epochs}, accumulation_steps={args.gradient_accumulation_steps}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"\n===== 第 {epoch}/{args.epochs} 轮 =====")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn, args.gradient_accumulation_steps)
        logger.info(f"训练损失：{train_loss:.4f}")

        # 验证
        dev_acc, dev_f1, _, _, _, _ = evaluate(model, dev_loader, device)
        logger.info(f"验证集准确率：{dev_acc:.4f}，F1分数：{dev_f1:.4f}")

        epoch_elapsed = time.time() - epoch_start_time
        epoch_times.append(epoch_elapsed)
        logger.info(f"Epoch {epoch} 耗时: {epoch_elapsed:.2f} s")

        # 保存最佳模型
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_1.pt"))
            logger.info(f"更新最佳模型（第 {epoch} 轮）")

    # 测试最佳模型
    logger.info(f"\n最佳模型在第 {best_epoch} 轮，验证集F1：{best_dev_f1:.4f}")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model_1.pt"), map_location=device))
    test_acc, test_f1, ids, labels, preds, probs = evaluate(model, test_loader, device)
    logger.info(f"测试集准确率：{test_acc:.4f}，F1分数：{test_f1:.4f}")
    logger.info("\n分类报告：")
    logger.info(classification_report(labels, preds))

    # 保存预测结果
    save_predictions_csv(os.path.join(args.output_dir, "predictions_1.csv"), ids, labels, preds, probs)
    logger.info(f"预测结果已保存到：{os.path.join(args.output_dir, 'predictions_1.csv')}")

    total_train_time = time.time() - train_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else None

    infer_ms, infer_sps = measure_inference_latency_baseline(
        model=model,
        dataloader=test_loader,
        device=device,
        warmup_steps=10,
        measure_steps=50
    )

    peak_gpu_memory_gb = None
    if torch.cuda.is_available():
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"峰值GPU内存使用: {peak_gpu_memory_gb:.2f} GB")

    logger.info(f"总训练耗时: {total_train_time:.2f} s")
    if avg_epoch_time is not None:
        logger.info(f"平均每轮耗时: {avg_epoch_time:.2f} s")
    if infer_ms is not None:
        logger.info(f"推理时延: {infer_ms:.4f} ms/sample")
    if infer_sps is not None:
        logger.info(f"推理吞吐: {infer_sps:.2f} samples/s")

    efficiency_metrics = {
        "model_name": args.model_name,
        "script": "xinjizuo_efficiency.py",
        "params_m": round(param_count / 1e6, 4),
        "total_train_time_sec": round(total_train_time, 4),
        "avg_epoch_time_sec": round(avg_epoch_time, 4) if avg_epoch_time is not None else None,
        "peak_gpu_memory_gb": round(peak_gpu_memory_gb, 4) if peak_gpu_memory_gb is not None else None,
        "inference_ms_per_sample": round(infer_ms, 6) if infer_ms is not None else None,
        "inference_samples_per_sec": round(infer_sps, 4) if infer_sps is not None else None,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "seed": args.seed
    }
    save_efficiency_metrics(os.path.join(args.output_dir, "efficiency_metrics.json"), efficiency_metrics)
    logger.info(f"效率统计已保存到：{os.path.join(args.output_dir, 'efficiency_metrics.json')}")

    logger.info("所有任务完成！")

if __name__ == "__main__":
    main()
