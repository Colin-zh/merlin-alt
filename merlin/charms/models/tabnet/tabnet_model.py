"""TabNet Model"""
from typing import Any, Dict

import torch
from torch import nn, Tensor

from ..base_model import BaseModel


class TabNetBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()
    
    def _build_network(self):
        pass