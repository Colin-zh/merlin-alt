"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
Reference: https://arxiv.org/pdf/1703.04247

FM部分的交叉项可能会丢失一些高阶特征交互信息（e.g. 非线性关系）。DeepFM通过引入深度神
经网络来捕捉这些复杂的非线性交互，从而提升模型的表达能力和预测性能。DeepFM模型结合了
FM的高效特征交互建模和深度神经网络的强大表达能力，适用于推荐系统等任务。

例如：收入和购买力的复杂关系，可能不仅仅是线性的，而是受到多种因素（边际效应递减）的非
线性影响。因此，DeepFM通过深度网络捕捉这些复杂关系，从而提升预测准确性。
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Dict, List
from ..base_model import BaseModel
from ..fm.fm_model import NumEmbedding, CatEmbedding, CatLinear, FMLayer


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_in: int, d_layers: List[int], dropout: float = 0.0, d_out: int = 1) -> None:
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = d
        layers.append(nn.Linear(d_layers[-1], d_out))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch_size, d_in)
        """
        return self.mlp(x)  # (batch_size, d_out)


class DeepFMBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config

        d_numerical = self.hparams.continuous_dim or 0
        d_embed = self.hparams.input_embed_dim

        categories = self.hparams.categorical_cardinality or []
        n_classes = self.hparams.output_dim

        deep_layers = [int(d) for d in self.hparams.deep_layers.split("-")]
        deep_dropout = self.hparams.deep_dropout

        self.d_numerical = d_numerical
        self.categories = categories
        self.n_classes = n_classes

        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None

        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

        self.fm = FMLayer(reduce_sum=False)
        self.fm_linear = nn.Linear(d_embed, n_classes)

        self.deep_in = d_numerical * d_embed + len(categories) * d_embed
        self.deep = MultiLayerPerceptron(
            d_in=self.deep_in,
            d_layers=deep_layers,
            dropout=deep_dropout,
            d_out=n_classes
        )
    
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
        # Deep part
        deep_out = self.deep(x_embed.view(-1, self.deep_in))  # (batch_size, n_classes)

        return linear_out + embed_out + deep_out  # (batch_size, n_classes)
        

class DeepFMModel(BaseModel):
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
        self._backbone = DeepFMBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self._head = nn.Identity()
        
    def forward(self, x: Dict) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by DeepFMModel.")
