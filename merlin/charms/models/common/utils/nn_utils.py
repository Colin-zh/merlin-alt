from typing import Union

import torch
import torch.nn as nn

from ..layers.batch_norm import BatchNorm1d

def _initialize_layers(activation: str, initialization: str, layers: Union[nn.Module, nn.Sequential]):
    """Initialize the weights of the layers in a neural network.

    Args:
        activation (str): The activation function used in the network.
        initialization (str): The weight initialization method.
        layers (Union[nn.Module, nn.Sequential]): The neural network layers to be initialized.
    """
    if type(layers) is nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=(nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1),
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)

def _linear_dropout_bn(in_units, out_units, activation, initialization, use_batch_norm, use_bias, dropout):
    if isinstance(activation, str):
        # TODO: support more activations
        _activation = getattr(nn, activation.capitalize())() if activation != 'leaky_relu' else nn.LeakyReLU()
    layers = []
    if use_batch_norm:
        layers.append(BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units, bias=use_bias)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return layers

def reset_all_weights(model: nn.Module):
    """Reset all model weights to their initial state.

    Args:
        model (nn.Module): The model whose weights need to be reset.
    """
    @torch.no_grad()
    def _reset_weights(m: nn.Module):
        # check if the module has a method "reset_parameters" and if it's callable.
        if callable(getattr(m, "reset_parameters", None)):
            m.reset_parameters()
    # Apply the weight reset function to all modules in the model.
    model.apply(_reset_weights)

def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == 'kaiming_uniform':
        nn.init.kaiming_uniform_(x, a=d_sqrt_inv)
        # nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == 'kaiming_normal':
        nn.init.kaiming_normal_(x, a=d_sqrt_inv)
        # nn.init.normal_(x, mean=0.0, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError("initialization should be either of `kaiming_normal`, `kaiming_uniform`," " `None`")


