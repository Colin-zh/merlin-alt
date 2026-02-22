import math
from typing import Callable, Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from . import activations


class AddNorm(nn.Module):
    """Add & Norm Module in Transformers. 
    
    Correct implementation:
    - Pre-Norm (norm_first=True): output = x + Dropout(Sublayer(LayerNorm(x)))
    - Post-Norm (norm_first=False): output = LayerNorm(x + Dropout(Sublayer(x)))

    Args:
        d_model: 输入和输出的特征维度
        dropout: Dropout概率  
        eps: LayerNorm的epsilon参数
        norm_first: 是否采用Pre-Norm结构
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5, norm_first: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: Tensor, sublayer_fn: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Args:
            x: 输入张量
            sublayer_fn: 子层函数，接受输入并返回输出
        """
        if self.norm_first:
            # Pre-Norm: x + Dropout(Sublayer(LayerNorm(x)))
            normalized_x = self.norm(x)
            sublayer_output = sublayer_fn(normalized_x)
            return x + self.dropout(sublayer_output)
        else:
            # Post-Norm: LayerNorm(x + Dropout(Sublayer(x)))
            sublayer_output = sublayer_fn(x)
            return self.norm(x + self.dropout(sublayer_output))


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Block in Transformers.

    Args:
        d_model: 输入和输出的特征维度
        nhead: 多头注意力机制中的头数
        head_dim: 每个注意力头的维度，如果为None则自动计算为d_model/nhead
        dropout: 注意力权重的dropout概率
        bias: 输出线性层是否使用偏置
        keep_attn: 是否保留注意力权重以供后续分析
        batch_first: 输入是否为(batch, seq, features)格式
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = True,
        keep_attn: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.keep_attn = keep_attn
        self.batch_first = batch_first

        # 使用分开的线性层而不是单个qkv_proj，提高可读性
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 注册缓冲区用于因果掩码
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query, key, value: 输入序列 (batch_size, seq_len, d_model)或(seq_len, batch_size, d_model)
            attn_mask: 注意力掩码 (seq_len, seq_len)
            key_padding_mask: 键填充掩码 (batch_size, seq_len)
            is_causal: 是否使用因果掩码
        """
        if not self.batch_first:
            # 转换为batch_first格式处理
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 投影到Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2).contiguous()

        # 计算注意力分数 (batch_size, nhead, seq_len, seq_len)
        # 代表第 batch_size 个样本的第 nhead 个头，对应 query 序列中的第 i 个位置与 key 序列中的第 j 个位置的注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if is_causal:
            causal_mask = self._get_causal_mask(seq_len, attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 适用于因果掩码、局部注意力、稀疏注意力
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # 处理变长序列的填充部分  
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到value (batch_size, nhead, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        attn_output = self.out_proj(attn_output)
        
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        # 保存注意力权重
        saved_attn_weights = attn_weights.detach() if self.keep_attn else None
        
        return attn_output, saved_attn_weights

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        keep_attn: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        
        # 自注意力机制
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            keep_attn=keep_attn,
            batch_first=batch_first,
        )
        
        # 前馈网络
        self.ffn = activations.PositionWiseFeedForward(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=activation,
            bias1=bias,
            bias2=bias,
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Add & Norm 模块
        self.add_norm1 = AddNorm(d_model, dropout, layer_norm_eps, norm_first)
        self.add_norm2 = AddNorm(d_model, dropout, layer_norm_eps, norm_first)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: 输入序列
            src_mask: 注意力掩码
            src_key_padding_mask: 键填充掩码
            is_causal: 是否使用因果掩码
        """
        attn_weights = None
        
        # 自注意力子层
        def self_attn_fn(x):
            nonlocal attn_weights
            output, weights = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            attn_weights = weights
            return output
        
        src = self.add_norm1(src, self_attn_fn)
        
        # 前馈子层
        src = self.add_norm2(src, self.ffn)
        
        return src, attn_weights


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder with multiple layers."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        keep_attn: bool = False,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.keep_attn = keep_attn
        activation = activations.get_activation_fn(activation)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                bias=bias,
                keep_attn=keep_attn,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = norm if norm is not None else nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self._init_weights()

    def _init_weights(self):
        # TODO kaiming/xavier initialization
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Optional[Tensor]]]]:
        """
        Args:
            src: 输入序列
            mask: 注意力掩码
            src_key_padding_mask: 键填充掩码
            is_causal: 是否使用因果掩码
        """
        attention_weights = [] if self.keep_attn else None
        
        output = src
        for layer in self.layers:
            output, attn_weights = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            if self.keep_attn and attn_weights is not None:
                attention_weights.append(attn_weights)
        
        output = self.norm(output)
        
        if self.keep_attn:
            return output, attention_weights
        return output


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        keep_attn: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first
        
        # (掩码)自注意力机制
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            keep_attn=keep_attn,
            batch_first=batch_first,
        )
        
        # 编码器-解码器(交叉)注意力机制
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            keep_attn=keep_attn,
            batch_first=batch_first,
        )
        
        # 前馈网络
        self.ffn = activations.PositionWiseFeedForward(
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=activation,
            bias1=bias,
            bias2=bias,
        )
    
    def forward(
        self,
        s
    )