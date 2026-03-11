"""Feature Tokenizer Transformer (FT-Transformer) model for tabular data."""

import torch
from torch import nn
from torch.nn import functional as F

from ..base_model import BaseModel


class AppendCLSToken(nn.Module):
    """Learnable [CLS] token that is prepended to the sequence."""

    def __init__(self, d_token: int):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        batch_size = x.size(0)
        cls_token = self.cls.expand(batch_size, -1, -1)
        return torch.cat([cls_token, x], dim=1)


class Embedding2dLayer(nn.Module):
    """Embed categorical + numerical columns into a shared token space."""

    def __init__(self, config):
        super().__init__()
        self.hparams = config

        self.categorical_cardinality = getattr(config, "categorical_cardinality", []) or []
        self.continuous_dim = getattr(config, "continuous_dim", 0) or 0
        self.d_token = config.input_embed_dim

        self.share = bool(getattr(config, "share_embedding", False))
        self.share_strategy = getattr(config, "share_embedding_strategy", "fraction")
        self.share_frac = getattr(config, "shared_embedding_fraction", 0.25)

        # Build categorical embeddings
        self.cat_embeddings = nn.ModuleList()
        self.shared_cat_embed = None
        if self.categorical_cardinality:
            if self.share:
                if self.share_strategy not in {"add", "fraction"}:
                    raise ValueError("share_embedding_strategy must be 'add' or 'fraction'")
                if self.share_strategy == "fraction":
                    self.share_dim = max(1, int(self.d_token * self.share_frac))
                    self.cat_dim = self.d_token - self.share_dim
                else:  # add
                    self.share_dim = self.d_token
                    self.cat_dim = self.d_token
                self.shared_cat_embed = nn.Parameter(
                    torch.zeros(len(self.categorical_cardinality), self.share_dim)
                )
                nn.init.xavier_uniform_(self.shared_cat_embed)
            else:
                self.cat_dim = self.d_token
                self.share_dim = 0

            for card in self.categorical_cardinality:
                emb = nn.Embedding(card, self.cat_dim)
                nn.init.xavier_uniform_(emb.weight)
                self.cat_embeddings.append(emb)

        # Build numerical embeddings
        if self.continuous_dim:
            self.num_linear = nn.ModuleList([
                nn.Linear(1, self.d_token, bias=getattr(config, "embedding_bias", True))
                for _ in range(self.continuous_dim)
            ])
            self.num_bn = nn.BatchNorm1d(self.continuous_dim) if getattr(
                config, "batch_norm_continuous_input", True
            ) else None
        else:
            self.num_linear, self.num_bn = None, None

        self.dropout = nn.Dropout(getattr(config, "embedding_dropout", 0.0))

    def forward(self, x: dict) -> torch.Tensor:
        tokens = []
        x_cat = x.get("categorical") if isinstance(x, dict) else None
        x_num = x.get("continuous") if isinstance(x, dict) else None

        if x_cat is not None and self.categorical_cardinality:
            for i, emb in enumerate(self.cat_embeddings):
                cat_tok = emb(x_cat[:, i])  # (B, cat_dim)
                if self.share:
                    shared_vec = self.shared_cat_embed[i].unsqueeze(0).expand(cat_tok.size(0), -1)
                    if self.share_strategy == "add":
                        if cat_tok.size(-1) != shared_vec.size(-1):
                            raise ValueError("For 'add' strategy, shared and cat dims must match")
                        cat_tok = cat_tok + shared_vec
                    else:  # fraction
                        cat_tok = torch.cat([cat_tok, shared_vec], dim=-1)
                tokens.append(cat_tok.unsqueeze(1))

        if x_num is not None and self.continuous_dim:
            if self.num_bn is not None:
                x_num = self.num_bn(x_num)
            for i, linear in enumerate(self.num_linear):
                num_tok = linear(x_num[:, i : i + 1])  # (B, d_token)
                tokens.append(num_tok.unsqueeze(1))

        if not tokens:
            raise ValueError("No input tokens were created. Check categorical/continuous inputs.")

        tokens = torch.cat(tokens, dim=1)  # (B, N, d_token)
        return self.dropout(tokens)


class TransformerBlock(nn.Module):
    """Lightweight Transformer encoder block with optional attention logging."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, activation: str, keep_attn: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        act = getattr(F, activation) if hasattr(F, activation) else F.gelu
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Dropout(dropout),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.keep_attn = keep_attn

    def forward(self, x: torch.Tensor):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=self.keep_attn, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class FTTransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hparams = config
        if getattr(config, "share_embedding_strategy", "fraction") not in {"add", "fraction"}:
            raise ValueError("share_embedding_strategy should be 'add' or 'fraction'")
        self._build_network()

    def _build_network(self):
        self.embedding_layer = Embedding2dLayer(self.hparams)
        self.cls_token = AppendCLSToken(self.hparams.input_embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.hparams.input_embed_dim,
                n_heads=self.hparams.num_heads,
                dim_ff=self.hparams.input_embed_dim * self.hparams.ff_hidden_multiplier,
                dropout=self.hparams.attn_dropout,
                activation=self.hparams.transformer_activation if hasattr(self.hparams, "transformer_activation") else "gelu",
                keep_attn=bool(getattr(self.hparams, "attn_feature_importance", False)),
            )
            for _ in range(self.hparams.num_attn_blocks)
        ])
        self.output_dim = self.hparams.input_embed_dim
        self._cached_importance = None

    def forward(self, x):
        tokens = self.embedding_layer(x)
        tokens = self.cls_token(tokens)

        attn_maps = []
        for block in self.blocks:
            tokens, attn = block(tokens)
            if attn is not None:
                attn_maps.append(attn)

        if getattr(self.hparams, "attn_feature_importance", False) and attn_maps:
            self._cached_importance = self._calculate_feature_importance(attn_maps)

        # Return CLS representation as backbone output
        return tokens[:, 0, :]

    @property
    def feature_importance_(self):
        return self._cached_importance

    def _calculate_feature_importance(self, attn_maps):
        # attn_maps: List[num_blocks] of (B, heads, seq, seq)
        stacked = torch.stack(attn_maps, dim=0)  # (L, B, H, S, S)
        # attention from CLS token (index 0) to feature tokens (skip CLS itself)
        cls_attn = stacked[:, :, :, 0, 1:]  # (L, B, H, num_features)
        importance = cls_attn.mean(dim=(0, 1, 2))  # (num_features,)
        return importance.detach().cpu().numpy()


class FTTransformer(BaseModel):
    """Feature Tokenizer Transformer (FT-Transformer) model for tabular data.

    This model is based on the paper "Tabular Data: Deep Learning is Not All You Need"
    (https://arxiv.org/abs/2106.11959) by Borisov et al.

    Args:
        config (FTTransformerConfig): Configuration object for the FT-Transformer model.
    """

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
        # Backbone
        self._backbone = FTTransformerBackbone(self.hparams)
        # Embedding layer comes from backbone
        self._embedding_layer = self._backbone.embedding_layer
        # Head
        self._head = self._get_head_from_config()

    def _build_model(self):
        return self._build_network()
    
    def feature_importance(self):
        if self.hparams.attn_feature_importance:
            if self.backbone.feature_importance_ is None:
                raise ValueError("Run a forward pass before requesting feature importance.")
            return super().feature_importance()
        else:
            raise ValueError("If you want Feature Importance, `attn_feature_weights` should be `True`.")
