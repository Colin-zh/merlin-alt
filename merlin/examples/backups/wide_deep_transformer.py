"""
https://colab.research.google.com/drive/1tB3N9Ue8Xq3PGHnpCJ7t8O6Mr0PJm9sp#scrollTo=3OKIU7AOFrKH
"""

import gc
import math
import matplotlib.pyplot as plt
import os
import random
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
np.random.seed(42)

print(torch.cuda.is_available())
print(torch.cuda.empty_cache())

# !nvidia-smi

##### 数据准备 #####
class FullDataset(Dataset):
    def __init__(self, data, mask, max_seq_len):
        # 训练均匀采样，测试评估不采样
        self.ffn_data = data["bas"].float()         # 保留格式 double -> float
        self.seq_data = data["trx"].float().clone() # mask 后需要变换

        # 标签转换
        labels = data["labels"]
        self.labels = torch.zeros(len(labels), 2, dtype=torch.float)
        self.labels[torch.arange(len(labels)), labels] = 1

        self.mask = self._create_and_apply_mask(mask, max_seq_len)

        # # mask转换
        # src_key_padding_mask = torch.tensor([
        #     [False] * eff_seq_len + [True] * (max_seq_len - eff_seq_len) 
        #     for eff_seq_len in mask
        # ])
        # self.mask = src_key_padding_mask
        
        # # batch_norm -inf 会异常，seq_data 根据mask填充-1
        # self.seq_data[self.mask.unsqueeze(1).repeat(1, self.seq_data.size(1), 1)] = -1
    
    def _create_and_apply_mask(self, mask, max_seq_len):
        batch_size = len(mask)
        padding_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        for i, eff_seq_len in enumerate(mask):
            if eff_seq_len < max_seq_len:
                padding_mask[i, eff_seq_len:] = True
                # 应用到 -inf mask
                self.seq_data[i, :, eff_seq_len:] = float("-inf") # -1

        return padding_mask

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ffn_data[idx], self.seq_data[idx], self.mask[idx], self.labels[idx]

p1_data = {
    "train": {
        "bas": torch.rand(10000, 10),
        "trx": torch.rand(10000, 20, 500),
        "labels": torch.randint(0, 2, (10000,))
    },
    "test": {
        "bas": torch.rand(2000, 10),
        "trx": torch.rand(2000, 20, 500),
        "labels": torch.randint(0, 2, (2000,))
    }
}

seq_mask = {
    "train": torch.randint(0, 500, (10000,)),
    "test": torch.randint(0, 500, (2000,)),
}

for i, mask in enumerate(seq_mask["train"]):
    p1_data["train"]["trx"][i, :, mask:] = float("-inf")

for i, mask in enumerate(seq_mask["test"]):
    p1_data["test"]["trx"][i, :, mask:] = float("-inf")

# 创建数据集实例
train_dataset = FullDataset(p1_data["train"], seq_mask["train"], max_seq_len=p1_data["train"]["trx"].shape[-1])
val_dataset   = FullDataset(p1_data["test"], seq_mask["test"], max_seq_len=p1_data["test"]["trx"].shape[-1])

print(train_dataset.labels[:, 1].sum(), val_dataset.labels[:, 1].sum())

##### 均匀采样 #####
class BalancedSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # 由白样本确定
        self.num_batches = int(self.dataset.labels[:, 0].sum()) // (self.batch_size // 2) + 1
        
        # 分离正负样本
        self.pos_indices = [i for i, label in enumerate(self.dataset.labels) if label[1] == 1]
        self.neg_indices = [i for i, label in enumerate(self.dataset.labels) if label[0] == 1]
        
        # 初始化白样本排列，用于不放回采样
        self.neg_permutation = np.random.permutation(self.neg_indices)
        self.cur_neg_idx = 0
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # 每个批次都包含黑样本
            batch_indices = random.sample(self.pos_indices, k=self.batch_size // 2)
            # 添加白样本（不放回）
            if self.cur_neg_idx + self.batch_size // 2 > len(self.neg_permutation):
                # 当前排列已使用完毕重新生成
                self.neg_permutation = np.random.permutation(self.neg_indices)
                self.cur_neg_idx = 0
            batch_indices.extend(self.neg_permutation[self.cur_neg_idx:(self.cur_neg_idx + self.batch_size // 2)])
            self.cur_neg_idx += self.batch_size // 2
            
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

def create_data_loader(dataset, batch_size=128):
    sampler = BalancedSampler(dataset, batch_size)
    data_loader = DataLoader(dataset, batch_sampler=sampler)
    return data_loader

train_data_loader = create_data_loader(train_dataset)
val_data_loader  = create_data_loader(val_dataset)

print(len(train_data_loader), len(val_data_loader))

# 验证 DataLoader
for batch_idx, (ffn_data, seq_data, mask, labels) in enumerate(train_data_loader):
    print(f"##### Batch {batch_idx}: #####\n  FFN Data shape {ffn_data.shape},\n"
          f"  SEQ Data shape {seq_data.shape},\n  MASK Data shape {mask.shape},\n" 
          f"  Labels {labels.shape} with Pos {labels[:, 1].sum()}\n")
    if batch_idx == 2:
        break


##### 模型定义 #####
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, in_features) 或 (batch_size, in_features)
            mask: (batch_size, seq_len) 或 None, True表示masked位置
        """
        if mask is None:
            return self.linear(x)
        
        batch_size, seq_len, in_features = x.shape

        # 扩展mask
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        
        # 计算每个样本每个特征维度的均值（只基于有效位置）
        valid_mask = ~mask_expanded

        # 有效位置数量 per (batch, feature)
        valid_count = valid_mask.sum(dim=1)  # (batch_size, in_features)
        valid_count = torch.clamp(valid_count, min=1)

        # 将mask位置设为中性值（该样本有效数据的均值）
        x_masked_for_mean = x * valid_mask.float()
        sum_x = x_masked_for_mean.sum(dim=1)  # (batch_size, in_features)
        feature_means = sum_x / valid_count    # (batch_size, in_features)

        # 创建填充后的tensor
        x_filled = torch.where(mask_expanded, 
                              feature_means.unsqueeze(1).expand(-1, seq_len, -1),
                              x)
        
        return self.linear(x_filled)


class MaskedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x, mask=None):
        """ 采用向量化实现，避免for循环
        Args:
            x: (batch_size, seq_len, d_model) 或 (batch_size, d_model)
            mask: (batch_size, seq_len) 或 None, True表示masked位置
        """
        if mask is None:
            # 标准LayerNorm
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normalized = (x - mean) / (std + self.eps)
            return normalized * self.weight + self.bias
        
        batch_size, seq_len, d_model = x.shape
        
        # 扩展 mask
        mask_expanded = mask.unsqueeze(-1).expand_as(x) # (batch_size, seq_len, d_model)

        # 创建有效位置编码
        valid_mask = ~mask_expanded

        # 计算每个样本的有效元素数量
        valid_count = valid_mask.sum(dim=1) # (batch_size, d_model)
        valid_count = torch.clamp(valid_count, min=1)  # 避免除0

        # 将mask位置设为0以便求和
        x_masked = x * valid_mask.float()

        # 计算均值和方差
        mean = (x_masked.sum(dim=1) / valid_count).unsqueeze(1) # (batch_size, 1, d_model)
        var = (x_masked - mean).pow(2) * valid_mask.float()
        std = (var.sum(dim=1) / valid_count + self.eps).sqrt().unsqueeze(1) # (batch_size, 1, d_model)

        # 应用归一化
        normalized = (x - mean) / std

        return normalized * self.weight + self.bias


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        batch_size = x.size(0)
        # sum
        return x + self.pe.expand(batch_size, -1, -1)

class TransformerClassifier(nn.Module):
    def __init__(self, input_ffn_dim, input_seq_dim, max_seq_len, nhead=4, d_model=64, num_encoder_layers=1, 
                 dim_feedforward=256, dropout=.1):
        super().__init__()
        
        # SubLayer 1. ffn 处理 基本信息特征
        
        # 定义一个ModuleList存储所有层
        layers = []
        # 第一层手动添加
        layers.append(nn.Linear(input_ffn_dim, dim_feedforward))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 隐层
        for i in range(num_encoder_layers):
            layers.append(nn.Linear(dim_feedforward, dim_feedforward))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_feedforward, d_model))
        self.layers = nn.Sequential(*layers)
        
        # SubLayer 2. TFM 处理 交易序列特征
        
        # 嵌入层
        # # 方案一：使用 batch_norm + linear 替代
        # # 不加 bn 在 val_epoch 中 with torch.no_grad() 时，会出现 logits为 [nan, nan]
        # self.bn = nn.BatchNorm1d(input_seq_dim)
        # self.emb = nn.Linear(input_seq_dim, d_model)

        # 方案二：使用 linear + layer_norm 替代（适用于特征相关度高）
        self.emb = MaskedLinear(input_seq_dim, d_model)
        # self.ln = nn.LayerNorm(d_model)
        self.ln = MaskedLayerNorm(d_model)

        # # 方案三：使用 instance_norm 替代，也即每个样本的每个特征，在max_seq_len时间步归一化（适用于特征间独立）
        # self.in_norm = nn.InstanceNorm1d(input_seq_dim)
        # self.emb = nn.Linear(input_seq_dim, d_model)
        
        # PE
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len=max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 更稳定的激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # OutputLayer 3. Concat 
        # 分类头 
        self.classifier = nn.Linear(d_model * 2, 2)

        # 初始化权重
        self._init_weight()
    
    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x_ffn, x_seq, mask):
        # 处理FFN特征
        x_ffn = self.layers(x_ffn) # （batch_size, d_model)
        
        # input x is (batch_size, input_dim, seq_len)

        # # 方案一：使用 batch_norm + linear 替代
        # embedded_x = self.emb(
        #     self.bn(
        #         x_seq
        #     ).permute(0, 2, 1).contiguous()
        # ) # (batch_size, seq_len, d_model)

        # # 方案二：使用 linear + layer_norm 替代
        embedded_x = self.ln(
            self.emb(x_seq.permute(0, 2, 1).contiguous(), mask=mask),
            mask=mask
        ) # (batch_size, seq_len, d_model)

        # # 方案三：使用 instance_norm 替代，也即每个样本的每个特征，在max_seq_len时间步归一化
        # embedded_x = self.emb(
        #     self.in_norm(x_seq).permute(0, 2, 1).contiguous()
        # ) # (batch_size, seq_len, d_model)

        # 添加PE, add
        pe_x = self.positional_encoding(embedded_x) # (batch_size, seq_len, d_model)

        # transformer
        transformer_output = self.transformer_encoder(pe_x, src_key_padding_mask=mask) # (batch_size, seq_len, d_model)

        # 使用第一个时间步（i.e. [CLS] token）输出分类
        cls_output = transformer_output[:, 0, :] # (batch_size, d_model)

        # 分类
        logits = self.classifier(torch.cat([x_ffn, cls_output], dim=-1)) # (batch_size, 2)
        return logits

    def predict_prob(self, x_ffn, x_seq, mask):
        """输出概率"""
        with torch.no_grad():
            logits = self.forward(x_ffn, x_seq, mask)
        return F.softmax(logits, dim=-1)[:, 1]
    
    def predict(self, x_ffn, x_seq, mask):
        """输出预测标签"""
        with torch.no_grad():
            logits = self.forward(x_ffn, x_seq, mask)
        return torch.argmax(logits, dim=-1)
  
    def get_model_size(self):
      param_size = 0
      for param in self.parameters():
          param_size += param.nelement() * param.element_size()
      total_size_mb = param_size / (1024 * 1024)
      print(f"Model size: {total_size_mb:.2f} MB")
      return total_size_mb

transformer_classifer = TransformerClassifier(p1_data["train"]["bas"].size(1), p1_data["train"]["trx"].size(1), p1_data["train"]["trx"].size(2))
print(transformer_classifer)
print(transformer_classifer.get_model_size())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = transformer_classifer.to(device)


##### 训练与评估 #####
def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    epoch_start_time = time.time()
    
    for i, (ffn_data, seq_data, mask, labels) in enumerate(data_loader):
        ffn_data, seq_data, mask, labels = ffn_data.to(device), seq_data.to(device), mask.to(device), labels.to(device)
        
        # 正确处理标签
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)
        
        optimizer.zero_grad()
        
        outputs = model(ffn_data, seq_data, mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # 调试：打印梯度信息（可选）
        if i == 0:  # 只在第一个batch打印
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"First batch gradient norm: {total_norm:.6f}")
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # 修复：使用detach()分离张量
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    epoch_time = (time.time() - epoch_start_time) / 60
    
    return {
        "train_loss": epoch_loss,
        "train_accuracy": epoch_acc,
        "train_precision": epoch_prec,
        "train_recall": epoch_recall,
        "train_f1": epoch_f1,
        "train_auc": epoch_auc,
        "train_time": epoch_time,
    }

def val_epoch(model, data_loader, criterion):
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    epoch_start_time = time.time()
    
    with torch.no_grad():  # 验证时已经不需要梯度，但为了统一还是用detach
        for i, (ffn_data, seq_data, mask, labels) in enumerate(data_loader):
            ffn_data, seq_data, mask, labels = ffn_data.to(device), seq_data.to(device), mask.to(device), labels.to(device)
            
            # 正确处理验证集的标签
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)

            outputs = model(ffn_data, seq_data, mask)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 同样使用detach()保持一致性
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    epoch_time = (time.time() - epoch_start_time) / 60
    
    return {
        "val_loss": epoch_loss,
        "val_accuracy": epoch_acc,
        "val_precision": epoch_prec,
        "val_recall": epoch_recall,
        "val_f1": epoch_f1,
        "val_auc": epoch_auc,
        "val_time": epoch_time,
    }

def train_and_validate(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=100, experiment_name=None):
    # 创建TensorBoard writer
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_dir = f"runs/{experiment_name}"
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"📊 TensorBoard logs saved to: {log_dir}")
    print(f"💡 View with: tensorboard --logdir={log_dir}")
    
    start_time = time.time()
    
    # 存储每个epoch指标
    history = {}
    
    # 最佳模型变量
    best_val_auc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 记录模型图（可选）
    try:
        # 获取一个batch的数据来记录模型图
        sample_ffn, sample_seq, sample_mask, _ = next(iter(train_data_loader))
        # 使用 torch.no_grad() 和 eval 模式
        with torch.no_grad():
            model.eval()
            writer.add_graph(model, (sample_ffn.to(device), sample_seq.to(device), sample_mask.to(device)))
    except Exception as e:
        print(f"⚠️ Could not add model graph: {e}")
    finally:
        model.train()  # 恢复模型为训练模式
    
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        
        # 训练和验证
        train_metrics = train_epoch(model, train_data_loader, criterion, optimizer)
        val_metrics = val_epoch(model, val_data_loader, criterion)
        history[epoch] = {**train_metrics, **val_metrics}
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ==================== TensorBoard记录 ====================
        
        # 1. 记录标量指标
        # 损失
        writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        
        # 准确率
        writer.add_scalar('Accuracy/train', train_metrics['train_accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['val_accuracy'], epoch)
        
        # AUC
        writer.add_scalar('AUC/train', train_metrics['train_auc'], epoch)
        writer.add_scalar('AUC/val', val_metrics['val_auc'], epoch)
        
        # 其他指标
        writer.add_scalar('Precision/train', train_metrics['train_precision'], epoch)
        writer.add_scalar('Precision/val', val_metrics['val_precision'], epoch)
        writer.add_scalar('Recall/train', train_metrics['train_recall'], epoch)
        writer.add_scalar('Recall/val', val_metrics['val_recall'], epoch)
        writer.add_scalar('F1/train', train_metrics['train_f1'], epoch)
        writer.add_scalar('F1/val', val_metrics['val_f1'], epoch)
        
        # 学习率
        writer.add_scalar('LR/lr', current_lr, epoch)
        
        # 2. 记录时间
        writer.add_scalar('Time/train_time', train_metrics['train_time'], epoch)
        writer.add_scalar('Time/val_time', val_metrics['val_time'], epoch)
        
        # 3. 记录梯度和权重分布（可选，每10个epoch记录一次）
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                writer.add_histogram(f'Weights/{name}', param, epoch)
        
        # ==================== 训练逻辑 ====================
        
        # 更新学习率
        scheduler.step(val_metrics['val_auc'])
        
        # 最佳模型保留
        if history[epoch]["val_auc"] > best_val_auc:
            best_val_auc = history[epoch]["val_auc"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'history': history
            }, f'best_model_{experiment_name}.pth')
            
            print(f"🎯 New best model! Val AUC: {best_val_auc:.4f}")
            
            # 记录最佳指标
            writer.add_scalar('Best/val_auc', best_val_auc, epoch)
        else:
            patience_counter += 1
        
        epoch_time = (time.time() - epoch_start_time) / 60
        
        # 控制台输出
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f} mins - LR: {current_lr:.2e}")
        print(f"    Train - Loss: {train_metrics['train_loss']:.4f}, AUC: {train_metrics['train_auc']:.4f}, ACC: {train_metrics['train_accuracy']:.4f}")
        print(f"    Val   - Loss: {val_metrics['val_loss']:.4f}, AUC: {val_metrics['val_auc']:.4f}, ACC: {val_metrics['val_accuracy']:.4f}")
        print(f"    Early stopping counter: {patience_counter}/{patience}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            writer.add_text('Training', f'Early stopping at epoch {epoch + 1}', epoch)
            break
    
    # ==================== 训练结束记录 ====================
    
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed in {total_time:.2f} hours")
    
    # 记录最终结果
    writer.add_text('Training', f'Completed in {total_time:.2f} hours', epoch)
    writer.add_text('Training', f'Best Val AUC: {best_val_auc:.4f}', epoch)
    
    # 记录超参数
    writer.add_hparams(
        {
            'lr': optimizer.param_groups[0]['lr'],
            'batch_size': train_data_loader.batch_size,
            'epochs': epoch + 1,
            'weight_decay': optimizer.param_groups[0].get('weight_decay', 0)
        },
        {
            'hparam/best_val_auc': best_val_auc,
            'hparam/final_train_loss': train_metrics['train_loss'],
            'hparam/final_val_loss': val_metrics['val_loss']
        }
    )
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with Val AUC: {best_val_auc:.4f}")
    
    # 关闭writer
    writer.close()
    
    return model, history, experiment_name

# # 分层学习率（FFN和Transformer部分需要不同的学习率）
# ffn_params = []
# transformer_params = []
# for name, param in model.named_parameters():
#     if 'layers' in name:  # FFN部分
#         ffn_params.append(param)
#     else:  # Transformer部分
#         transformer_params.append(param)

# optimizer = optim.AdamW([
#     {'params': ffn_params, 'lr': 1e-4},
#     {'params': transformer_params, 'lr': 5e-5}
# ], weight_decay=0.01)

# 设置优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 训练并记录到TensorBoard
model, history, exp_name = train_and_validate(
    model=model,
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50,
    experiment_name="transformer_classifier_v2"  # 可选的实验名称
)

print(f"实验名称: {exp_name}")

def plot_history(history):
    # 创建一个图形对象和一个子图对象数组
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    
    for i, kw in enumerate(["loss", "auc", "accuracy", "precision", "recall", "f1"]):
        ax = axs[i // 2, i % 2]
        
        x = range(len(history))
        y_train = [h[f"train_{kw}"] for h in history.values()]
        y_val   = [h[f"val_{kw}"] for h in history.values()]
        
        line1, = ax.plot(x, y_train, label='train')
        line2, = ax.plot(x, y_val, label='val')
        
        # 设置标题和图例
        ax.set_title(f'Subplot {kw}')
        ax.legend()

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()

plot_history(history)
# %reload_ext tensorboard
# %tensorboard --logdir runs/transformer_classifier_v2

##### 模型保存 #####
torch.save(transformer_classifer.state_dict(), f"transformer_classifer_{int(time.time())}.pth")
