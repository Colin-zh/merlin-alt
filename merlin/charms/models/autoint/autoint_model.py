"""AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks."""

from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from ..base_model import BaseModel
from ..fm.fm_model import NumEmbedding, CatEmbedding, CatLinear


class InteractingLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_residual: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_fields, d_model)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.dropout(attn_out)
        if self.use_residual:
            x = self.norm(x + attn_out)
        else:
            x = self.norm(attn_out)
        x_ff = self.linear(x)
        if self.use_residual:
            x = self.norm(x + self.dropout(x_ff))
        else:
            x = self.norm(self.dropout(x_ff))
        return F.relu(x)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_layers: List[int], dropout: float = 0.0, d_out: int = 1):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = d
        layers.append(nn.Linear(d_in, d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AutoIntBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config

        d_embed = self.hparams.input_embed_dim
        categories = self.hparams.categorical_cardinality or []
        d_numerical = self.hparams.continuous_dim or 0
        n_classes = self.hparams.output_dim

        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None

        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

        self.n_fields = (len(categories) if categories else 0) + (d_numerical if d_numerical else 0)

        self.interactions = nn.ModuleList([
            InteractingLayer(
                d_model=d_embed,
                n_heads=self.hparams.attn_heads,
                dropout=self.hparams.attn_dropout,
                use_residual=self.hparams.use_residual,
            )
            for _ in range(self.hparams.attn_layers)
        ])

        mlp_layers = [int(d) for d in self.hparams.deep_layers.split("-")]
        self.deep = MLP(d_in=self.n_fields * d_embed, d_layers=mlp_layers, dropout=self.hparams.deep_dropout, d_out=n_classes)

        self.output_dim = n_classes

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_cat, x_num = x.get("categorical"), x.get("numerical") or x.get("continuous")

        linear_out = 0.0
        if self.num_linear is not None and x_num is not None:
            linear_out = linear_out + self.num_linear(x_num)
        if self.cat_linear is not None and x_cat is not None:
            linear_out = linear_out + self.cat_linear(x_cat)

        embeds = []
        if self.num_embedding is not None and x_num is not None:
            embeds.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding is not None and x_cat is not None:
            embeds.append(self.cat_embedding(x_cat))

        if not embeds:
            raise ValueError("AutoIntBackbone expects numerical or categorical inputs.")

        x_embed = torch.cat(embeds, dim=1)  # (B, n_fields, d_embed)

        for layer in self.interactions:
            x_embed = layer(x_embed)

        deep_in = x_embed.reshape(x_embed.size(0), -1)
        deep_out = self.deep(deep_in)

        return linear_out + deep_out


class AutoIntModel(BaseModel):
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
        self._backbone = AutoIntBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self._head = nn.Identity()

    def _build_model(self):
        return self._build_network()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by AutoIntModel.")
