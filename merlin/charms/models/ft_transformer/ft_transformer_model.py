"""Feature Tokenizer Transformer (FT-Transformer) model for tabular data."""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..base_model import BaseModel
from ..common.layers import AppendCLSToken, Embedding2dLayer, TransformerEncoderBlock


class FTTransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.share_embedding_strategy in [
            "add",
            "fraction",
        ], (
            "`share_embedding_strategy` should be one of `add` or `fraction`,"
            f" not {self.hparams.share_embedding_strategy}"
        )
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.add_cls = AppendCLSToken(
            d_token = self.hparams.input_embed_dim,
            initialization = self.hparams.embedding_initialization,
        )
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                d_model = self.hparams.input_embed_dim,
                n_heads = self.hparams.num_heads,
                d_head = self.hparams.transformer_head_dim,
                d_ff = self.hparams.input_embed_dim * self.hparams.ff_hidden_multiplier,
                attn_dropout = self.hparams.attn_dropout,
                ff_dropout = self.hparams.ff_dropout,
                add_norm_dropout = self.hparams.add_norm_dropout,
                activation = self.hparams.transformer_activation,
            )
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
    
    def _build_embedding_layers(self):
        pass

    def forward(self, x):
        pass

    def _calculate_feature_importance(self):
        pass


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
        # Embedding layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self._head = self._get_head_from_config()
    
    def feature_importance(self):
        if self.hparams.attn_feature_importance:
            return super().feature_importance()
        else:
            raise ValueError("If you want Feature Importance, `attn_feature_weights` should be `True`.")
