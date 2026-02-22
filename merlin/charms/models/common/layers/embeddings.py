import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super().__init__(vocab_size, embed_size, padding_idx=0)


# Tips: Bert中的位置编码是可学习的，这里实现一个固定的位置编码作为参考
class PositionalEmbedding(nn.Module):
    """Fixed Positional Embedding as described in "Attention is All You Need".
    $$PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$$
    $$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$$
    """

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, d_model).float()
        pe.requires_grad = False

        # position shape: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        # div_term shape: (d_model/2,)
        # div_term = exp(-log(10000) * (2i/d_model)) = 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Return shape: (1, seq_len, d_model)
        return self.pe[:, :x.size(1)]  # Torch的广播机制


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size):
        super().__init__(3, embed_size, padding_idx=0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        # 依赖 PositionalEmbedding
        self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量, 形状为 (batch_size, seq_len, d_model)
        返回:
            x + positional_embedding, 形状为 (batch_size, seq_len, d_model)
        """
        # 获取位置编码
        pos_emb = self.pos_embedding(x)  # 形状: (1, seq_len, d_model)
        
        # 将位置编码加到输入上，利用广播机制
        # pos_emb 会广播到 (batch_size, seq_len, d_model)
        return x + pos_emb

