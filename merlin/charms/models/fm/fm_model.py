import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Dict, List
from ..base_model import BaseModel


class NumEmbedding(nn.Module):
    """
    input_shape: (batch_size, features_num(n), d_in), # d_in is 1 for numerical features
    output_shape: (batch_size, features_num(n), d_out)
    """

    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(n, d_in, d_out)) # Tensor随机初始化
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias
    
    def forward(self, x_num: Tensor) -> Tensor:
        # x_num: (batch_size, features_num(n), d_in)
        assert x_num.dim() == 3, "x_num must be 3-dimensional"
        # 方法一: 广播乘法 + 求和
        # out = x_num[..., None] * self.weight[None] # 等价于 out = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)
        # 广播规则
        # 1. 从最后一个维度开始向前对齐
        # 2. 维度为1的可以扩展到任意大小
        # 3. 缺失的维度视为1
        # 广播对齐可视化
        # x_num[...,None]: [batch_size, features_num(n), d_in, 1    ]
        # weight[None].  : [1,          features_num(n), d_in, d_out]
        # 结果形状: (batch_size, features_num(n), d_in, d_out)
        # out = out.sum(-2) # (batch_size, features_num(n), d_out) 
        # # remember d_in is 1 for numerical features, so this sum equals to removing the d_in dimension
        # 方法二: 爱因斯坦求和约定
        out = torch.einsum('bni,nio->bno', x_num, self.weight)  # (batch_size, features_num(n), d_out)
        if self.bias is not None:
            out += self.bias[None] # 广播添加偏置, 等价于 out = out + self.bias.unsqueeze(0)
        return out  # (batch_size, features_num(n), d_out)


class CatEmbedding(nn.Module):
    """
    input_shape: (batch_size, features_num(n))
    output_shape: (batch_size, features_num(n), d_embed), each categorical feature 
    is mapped to a d_embed-dimensional vector
    """

    def __init__(self, categories: List[int], d_embed: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(sum(categories), d_embed)
        # 使用offset避免为每个类别单独创建embedding层
        # len(categories) = features_num(n)
        self.offsets = nn.Parameter(
            torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False
        )
        torch.nn.init.xavier_uniform_(self.embeddings.weight.data)
    
    def forward(self, x_cat: Tensor) -> Tensor:
        """
        :param x_cat: Long tensor of shape (batch_size, features_num(n))
        """
        x = x_cat + self.offsets[None]  # (batch_size, features_num(n))
        return self.embeddings(x)  # (batch_size, features_num(n), d_embed)


class CatLinear(nn.Module):
    """
    input_shape: (batch_size, features_num(n))
    output_shape: (batch_size, d_out)
    """

    def __init__(self, categories: List[int], d_out: int = 1) -> None:
        super().__init__()
        self.fc = nn.Embedding(sum(categories), d_out)
        self.bias = nn.Parameter(torch.zeros((d_out,)))
        self.offsets = nn.Parameter(
            torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False
        )
    
    def forward(self, x_cat: Tensor) -> Tensor:
        """
        :param x_cat: Long tensor of shape (batch_size, features_num(n))
        """
        x = x_cat + self.offsets[None]  # (batch_size, features_num(n))
        return torch.sum(self.fc(x), dim=1) + self.bias  # (batch_size, d_out)
    

class FMLayer(nn.Module):
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum
    
    def forward(self, x: Tensor) -> Tensor: #注意：这里的x是公式中的 <v_i> * xi
        """Factorization Machine layer for pairwise feature interactions
        
        Args:
            x: Float tensor of size ``(batch_size, num_features, k)``
                where k is the embedding dimension.
                Note: x already contains weighted embeddings: v_i * x_i
                
        Returns:
            Float tensor of size ``(batch_size, 1)`` if reduce_sum=True,
            or ``(batch_size, k)`` if reduce_sum=False
            
        Math derivation:
            Original FM pairwise interaction term:
            ∑∑⟨v_i, v_j⟩x_i x_j for i<j
            
            Using mathematical identity:
            (∑a_i)² = ∑a_i² + 2∑∑a_i a_j for i<j
            
            Let a_i = v_i x_i, then:
            ∑∑⟨v_i, v_j⟩x_i x_j = 0.5 * [(∑v_i x_i)² - ∑(v_i x_i)²]
            
            This reduces complexity from O(n²k) to O(nk)
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # (batch_size, d_embed)
        sum_of_square = torch.sum(x ** 2, dim=1)  # (batch_size, d_embed)
        ix = square_of_sum - sum_of_square  # (batch_size, d_embed)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)  # (batch_size, 1)
        return 0.5 * ix  # (batch_size, 1) or (batch_size, d_embed)


class FMBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config

        d_numerical = self.hparams.continuous_dim or 0
        d_embed = self.hparams.input_embed_dim

        categories = self.hparams.categorical_cardinality or []
        n_classes = self.hparams.output_dim

        self.d_numerical = d_numerical
        self.categories = categories
        self.n_classes = n_classes

        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None

        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

        self.fm = FMLayer(reduce_sum=False)
        self.fm_linear = nn.Linear(d_embed, n_classes)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x_cat, x_num = x['categorical'], x['numerical']
        # Linear part
        linear_out = 0.0
        if self.num_linear:
            linear_out += self.num_linear(x_num)  # (batch_size, n_classes)
        if self.cat_linear:
            linear_out += self.cat_linear(x_cat)  # (batch_size, n_classes)
        # Interaction part
        embed_out = []
        if self.num_embedding:
            embed_out.append(self.num_embedding(x_num[..., None]))  # (batch_size, n_num, d_embed)
        if self.cat_embedding:
            embed_out.append(self.cat_embedding(x_cat))  # (batch_size, n_cat, d_embed)
        x_embed = torch.cat(embed_out, dim=1)  # (batch_size, n_total, d_embed)
        embed_out = self.fm_linear(self.fm(x_embed))  # (batch_size, n_classes)
        
        return linear_out + embed_out  # (batch_size, n_classes)
    

class FMModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head
        
    def _build_network(self):
        self._embedding_layer = nn.Identity()
        self._backbone = FMBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self._head = nn.Identity()
        
    def forward(self, x: Dict) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by FMModel.")
    
