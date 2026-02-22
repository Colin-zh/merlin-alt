"""Deep & Cross Network Model (DCN)

DCN: Deep & Cross Network for Ad Click Predictions
References: https://arxiv.org/pdf/1708.05123

模型架构类似 Wide & Deep，但使用 Cross Network 替代了 Wide 部分来建模低阶特征交互。
Cross Network 通过显式地计算特征的交叉项来捕捉特征之间的低阶交互关系，而 Deep 部分则
负责学习高阶特征交互。DCN 通过结合 Cross Network 和 Deep Neural Network 的优势，
实现了对特征交互的全面建模，从而提升了预测性能。

通过 Automatic Feature Crossing，是交叉不仅局限于FM的二阶交叉，而是可以捕捉更高阶的
交叉特征。例如：用户的购买行为可能受到多个因素的共同影响，如年龄、收入和兴趣爱好等，
这些因素之间的复杂交互关系可以通过 Cross Network 来建模，从而提升预测的准确性。

x_cross_l+1 = x_0 * (x_cross_l · w_l) + b_l + x_cross_l
其中，x_0 是输入特征，x_cross_l 是第 l 层的交叉特征，w_l 和 b_l 分别是第 l 层的权重
和偏置。* 本质是收到残差网络的启发，通过逐层叠加交叉特征来捕捉更复杂的交互关系。* 

* 叉乘阶数由网络深度决定，深度l对应最高l+1阶交叉特征。
* 自动叉乘，模型参数量仅随维度线性增长：2l*d，其中d为输入维度，l为网络深度。
* 参数共享：每一层的权重w_l和偏置b_l在不同样本间共享，提升泛化能力。
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Dict, List
from ..base_model import BaseModel


class CatEmbeddingSqrt(nn.Module):
    def __init__(self, categories: List[int], d_embed_max: int = 100) -> None:
        super().__init__()
        self.categories = categories
        self.d_embed_list = [min(max(int(x ** 0.5), 2), d_embed_max) for x in categories]
        self.embedding_list = nn.ModuleList([
            nn.Embedding(self.categories[i], self.d_embed_list[i]) 
            for i in range(len(categories))
        ])
        self.d_cat_sum = sum(self.d_embed_list)
    
    def forward(self, x_cat: Tensor) -> Tensor:
        return torch.cat([
            self.embedding_list[i](x_cat[:, i])
            for i in range(len(self.categories))
        ], dim=1)  # (batch_size, sum(d_embed_list))


class MLP(nn.Module):
    def __init__(self, d_in: int, d_layers: List[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = d
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch_size, d_in)
        """
        return self.mlp(x)  # (batch_size, d_out)


class CrossNetVector(nn.Module):
    def __init__(self, d_in: int, n_cross: int = 2) -> None:
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList([
            nn.Linear(d_in, 1, bias=False) for _ in range(self.n_cross)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_in)) for _ in range(self.n_cross)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = x # (batch_size, d_in)
        xi = x # (batch_size, d_in)
        for i in range(self.n_cross):
            # x_cross_l+1 = x_0 * (x_cross_l · w_l) + b_l + x_cross_l
            xi = x0 * self.linears[i](xi) + self.biases[i] + xi  # (batch_size, d_in)
        return xi  # (batch_size, d_in)


class CrossNetMatrix(nn.Module):
    def __init__(self, d_in: int, n_cross: int = 2) -> None:
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList([
            nn.Linear(d_in, d_in) for _ in range(self.n_cross)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = x # (batch_size, d_in)
        xi = x # (batch_size, d_in)
        for i in range(self.n_cross):
            xi = x0 * self.linears[i](xi) + xi  # (batch_size, d_in)
        return xi  # (batch_size, d_in)


class CrossNetMix(nn.Module):
    def __init__(self, d_in: int, n_cross: int = 2, low_rank: int = 32, n_experts: int = 4) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_cross = n_cross
        self.low_rank = low_rank
        self.n_experts = n_experts

        # U: (d_in, low_rank)
        self.U_list = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_normal_(torch.empty(n_experts, d_in, low_rank))
            ) for _ in range(self.n_cross)
        ])
        # V: (d_in, low_rank)
        self.V_list = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_normal_(torch.empty(n_experts, d_in, low_rank))
            ) for _ in range(self.n_cross)
        ])
        # C: (low_rank, low_rank)
        self.C_list = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_normal_(torch.empty(n_experts, low_rank, low_rank))
            ) for _ in range(self.n_cross)
        ])
        # gating network G: (d_in, 1)
        self.gating = nn.ModuleList([
            nn.Linear(d_in, 1, bias=False) for _ in range(self.n_experts)
        ])
        # Bias
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_in)) for _ in range(self.n_cross)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x0 = x # (batch_size, d_in)
        xi = x # (batch_size, d_in)
        for i in range(self.n_cross):
            output_of_experts = []
            gating_scores_of_experts = []
            for expert_id in range(self.n_experts):
                # (1) G(xi)
                # compute the gating score by xi
                gating_scores_of_experts.append(self.gating[expert_id](xi))  # (batch_size, 1)
                # (2) E(xi)
                # project the input xi to low-rank space
                v_x = torch.matmul(xi, self.V_list[i][expert_id])  # (batch_size, low_rank)
                # nonlinear activation in low-rank space
                v_x = torch.tanh(v_x)  # (batch_size, low_rank)
                v_x = torch.matmul(v_x, self.C_list[i][expert_id])  # (batch_size, low_rank)
                v_x = torch.tanh(v_x)  # (batch_size, low_rank)
                # project back to original(d_in) space
                uv_x = torch.matmul(v_x, self.U_list[i][expert_id].T)  # (batch_size, d_in)
                expert_out = x0 * (uv_x + self.biases[i])  # (batch_size, d_in)
                output_of_experts.append(expert_out)
            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, dim=2)  # (batch_size, d_in, n_experts)
            gating_scores_of_experts = torch.stack(gating_scores_of_experts, dim=1)  # (batch_size, n_experts, 1)
            moe_out = torch.bmm(output_of_experts, gating_scores_of_experts.softmax(dim=1))  # (batch_size, d_in, 1)
            xi = torch.squeeze(moe_out) + xi  # (batch_size, d_in)
        return xi  # (batch_size, d_in)


class DeepCrossBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config

        d_numerical = self.hparams.continuous_dim
        d_embed_max = self.hparams.input_embed_max

        cross_type = self.hparams.cross_type
        n_cross = self.hparams.cross_order

        low_rank = self.hparams.low_rank
        n_experts = self.hparams.experts_num

        mlp_layers = [int(x) for x in self.hparams.mlp_layers.split("-")]
        mlp_dropout = self.hparams.mlp_dropout
        stacked = self.hparams.stacked

        categories = self.hparams.categorical_cardinality
        n_classes = self.hparams.output_dim

        if cross_type == "mix":
            assert low_rank is not None and n_experts is not None, \
                "CrossNetMix requires low_rank and n_experts parameters."
        
        self.categories = categories
        self.n_classes = n_classes
        self.stacked = stacked

        self.cat_embedding = CatEmbeddingSqrt(categories, d_embed_max) if categories else None

        self.d_in = d_numerical
        if self.cat_embedding:
            self.d_in += self.cat_embedding.d_cat_sum
        
        if cross_type == "vector":
            self.cross_layer = CrossNetVector(self.d_in, n_cross)
        elif cross_type == "matrix":
            self.cross_layer = CrossNetMatrix(self.d_in, n_cross)
        elif cross_type == "mix":
            self.cross_layer = CrossNetMix(self.d_in, n_cross, low_rank, n_experts)
        else:
            raise ValueError(f"Unsupported cross_type: {cross_type}")

        self.mlp = MLP(self.d_in, mlp_layers, mlp_dropout)

        if self.stacked:
            self.last_linear = nn.Linear(mlp_layers[-1], n_classes)
        else:
            self.last_linear = nn.Linear(self.d_in + mlp_layers[-1], n_classes)
    
    def forward(self, x: Dict):
        x_cat, x_num = x["categorical"], x["continuous"]

        # embedding
        x_total = []
        if x_num is not None:
            x_total.append(x_num)
        if self.cat_embedding is not None:
            x_total.append(self.cat_embedding(x_cat))
        x_total = torch.cat(x_total, dim=1)  # (batch_size, d_in)

        # cross part
        x_cross = self.cross_layer(x_total)  # (batch_size, d_in)

        # deep part
        if self.stacked:
            x_deep = self.mlp(x_cross)  # (batch_size, d_mlp_out)
            x_out = self.last_linear(x_deep)  # (batch_size, n_classes)
        else:
            x_deep = self.mlp(x_total)  # (batch_size, d_mlp_out)
            x_deep_cross = torch.cat([x_deep, x_cross], dim=1)  # (batch_size, d_mlp_out + d_in)
            x_out = self.last_linear(x_deep_cross)  # (batch_size, n_classes)
        return x_out


class DeepCrossModel(BaseModel):
    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
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
        self._backbone = DeepCrossBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self._head = nn.Identity()
        
    def forward(self, x: Dict) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by DeepCrossModel.")
