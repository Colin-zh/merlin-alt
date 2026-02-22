import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

from . import config as head_config
from ..utils.nn_utils import _initialize_layers, _linear_dropout_bn


def config_link(r):
    """This is a helper function decorator to link the config to the head."""
    def wrapper(f):
        f.config_template = r
        return f
    return wrapper


class Head(nn.Module):
    def __init__(self, layers, config_template, **kwargs):
        super(Head, self).__init__()
        self.layers = layers
        self._config_template = config_template
    
    def forward(self, x):
        return self.layers(x)


class LinearHead(Head):
    _config_template = head_config.LinearHeadConfig

    def __init__(self, input_dim: int, output_dim: int, config, **kwargs):
        """Linear head for classification/regression tasks.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            config (heads.config.LinearHeadConfig): Configuration object for the linear head.
            **kwargs: Additional keyword arguments.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        # Linear layers
        _layers = []
        _curr_units = input_dim
        for units in config.layers.split('-'):
            try:
                int(units)
            except ValueError:
                # If no hidden layers, just a linear layer
                if units != '':
                    raise ValueError(f"Invalid units: {units} in layers: {config.layers}")
            _layers.extend(
                _linear_dropout_bn(
                    _curr_units,
                    int(units),
                    config.activation,
                    config.initialization,
                    config.use_batch_norm,
                    config.use_bias,
                    config.dropout,
                )
            )
            _curr_units = int(units)
        # Append final layer
        _layers.append(nn.Linear(_curr_units, output_dim, bias=config.use_bias))
        linear_layers = nn.Sequential(*_layers)
        _initialize_layers(config.activation, config.initialization, linear_layers)
        super(LinearHead, self).__init__(layers=linear_layers, config_template=head_config.LinearHeadConfig)
