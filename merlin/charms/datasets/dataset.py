# MIT License
# Copyright (c) 2026 Colin-zh
import random
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler


class BalancedSampler(BatchSampler):
    """
    自定义均匀采样器，确保每个批次中正负样本数量相等。适用于二分类任务，假设标签格式为one-hot编码。
    """
    def __init__(self, dataset: Dataset, batch_size: int):
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


class BERTDataset(Dataset):
    """Dataset for BERT Language Model"""
    
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=self.encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines = (self.corpus_lines or 0) + 1
            
            if on_memory:
                self.lines = [
                    line[:-1].split("\t")
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines)
                ]
                self.corpus_lines = len(self.lines)
        
        if not on_memory:
            self.file = open(corpus_path, "r", encoding=self.encoding)
            self.random_file = open(corpus_path, "r", encoding=self.encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
    
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, index):
        t1, t2, is_next_label = self.random_sentences(index)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_sentences(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            # 50% isNext
            return t1, t2, 1
        else:
            # 50% isNotNext
            return t1, self.get_random_line(t2), 0
    
    def get_corpus_line(self, index):
        if self.on_memory:
            return self.lines[index][0], self.lines[index][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
        
        t1, t2 = line[:-1].split("\t")
        return t1, t2
    
    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]
        
        line = self.lines.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # 15% chance to mask token
            if prob < 0.15:
                prob /= 0.15

                # 80% change to replace with [MASK]
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% chance to replace with random word
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% chance to keep the same, if not exists, replace with [UNK]
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                
                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)  # no prediction
        return tokens, output_label


def create_data_loader(dataset: Dataset, batch_size: int = 128) -> DataLoader:
    """基于均匀采样创建数据加载器"""
    sampler = BalancedSampler(dataset, batch_size)
    data_loader = DataLoader(dataset, batch_sampler=sampler)
    return data_loader
