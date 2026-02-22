import torch
from torch import nn, einsum
from torch.nn import functional as F


__all__ = [
    "PositionWiseFeedForward",
    "Sigmoid",
    "Tanh",
    "Swish",
    "SwishGLU",
    "ReLU",
    "LeakyReLU",
    "ReGLU",
    "GELU",
    "GEGLU",
    "ELU",
    "CELU",
    "get_activation_fn",
]


def get_activation_fn(name: str):
    """Return an activation function given its name."""
    name = name.lower()
    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "swish" or name == "silu":
        return Swish()
    elif name == "swishglu" or name == "siluglu":
        return SwishGLU()
    elif name == "relu":
        return ReLU()
    elif name == "leakyrelu":
        return LeakyReLU()
    elif name == "reglu":
        return ReGLU()
    elif name == "gelu":
        return GELU()
    elif name == "geglu":
        return GEGLU()
    elif name == "elu":
        return ELU()
    elif name == "celu":
        return CELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class PositionWiseFeedForward(nn.Module):
    """"Position-wise Feed-Forward Network (FFN)
    
    This module implements the position-wise feed-forward network used in Transformer models.
    FFN consists of two linear transformations with an activation function in between. Number
    of dimensions in the hidden layer $d_{ff}$ is generally set to around 4 times of the token
    embedding dimension $d_{model}$. So it is sometimes also called the expand-and-contract
    network. 
    
    There is an activation at the hidden layer, which is ReLU in the original Transformer,
    $$\\max(0, x)$$. That is, the FFN function can be written as:
        $$FFN(x, W_1, W_2, b_1, b_2) = \\max(0, xW_1 + b_1)W_2 + b_2$$
    where $W_1$, $b_1$, $W_2$, and $b_2$ are learnable parameters of the two linear transformations.

    Sometimes, other activation functions are used, such as GELU (Gaussian Error Linear Unit) in 
    BERT, $$x * \\Phi(x)$$, where $\\Phi(x) = P(X \\le x), X \\sim \\mathcal{N}(0, 1)$.

    GLU (Gated Linear Unit) and its variants are also used in some Transformer models, such as
    ReGLU and GEGLU (Gated Linear Unit with ReLU and GELU activations, respectively) in the
    "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202). GLU splits the input
    tensor into two halves along the last dimension, applies an activation function to the second
    half, and then performs element-wise multiplication between the first half and the activated
    second half. For example, the ReGLU function can be written as:
        $$ReGLU(x) = x_1 * \\max(0, x_2)$$
    where $x_1$ and $x_2$ are the two halves of the input tensor $x$.

    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation=nn.ReLU(),
                 is_gated: bool = False, bias1: bool = True, bias2: bool = True, 
                 bias_gate: bool = True):
        """
        Args:
            d_model (int): Input and output dimension of the FFN.
            d_ff (int): Hidden layer dimension of the FFN.
            dropout (float): Dropout rate.
            activation (callable): Activation function to use. Default is ReLU.
            is_gated (bool): Whether to use a gated activation function (GLU or its variants).
            bias1 (bool): Whether to use bias in the first linear layer.
            bias2 (bool): Whether to use bias in the second linear layer.
            bias_gate (bool): Whether to use bias in the gating linear layer (if is_gated is True).
        """
        super().__init__()
        # Layer one parameterized by weights $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer two parameterized by weights $W_2$ and bias $b_2$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether to use gated activation function
        self.is_gated = is_gated
        if self.is_gated:
            # If there is a gate the linear layer to transform inputs to be multiplied
            # by the gate, parameterized by weights $W_g$ and bias $b_g$
            self.gate_layer = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # $f(xW_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(xW_1 + b_1) \otimes (xW_g + b_g)$
        if self.is_gated:
            x = g * self.gate_layer(x)
        else:
            x = g
        # Dropout
        x = self.dropout(x)
        # $(f(xW_1 + b_1))W_2 + b_2 or (f(xW_1 + b_1) \otimes (xW_g + b_g))W_2 + b_2$
        return self.layer2(x)


class Sigmoid(nn.Module):
    """Sigmoid activation function."""
    def forward(self, x):
        return torch.sigmoid(x)
    

class Tanh(nn.Module):
    """Tanh activation function."""
    def forward(self, x):
        return torch.tanh(x)


class Swish(nn.Module):
    """
    Swish activation function. Also known as SiLU (Sigmoid Linear Unit).
        ϕ(x) = x * σ(x)
    where σ(x) is the sigmoid function.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishGLU(nn.Module):
    """
    Swish activation function with gating mechanism.
        ϕ(x) = x1 * σ(x2)
    where x is split into two halves: x1 and x2, and σ(x2) is the sigmoid function.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class ReLU(nn.Module):
    """ReLU activation function."""
    def forward(self, x):
        return F.relu(x)


class LeakyReLU(nn.Module):
    """Leaky ReLU activation function."""
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ReGLU(nn.Module):
    """
    ReLU activation function with gating mechanism.
        ϕ(x) = x1 * ReLU(x2)
    where x is split into two halves: x1 and x2.

    ReGLU is a variant of the GLU (Gated Linear Unit) activation function.
    It splits the input tensor into two halves along the last dimension,
    applies the ReLU activation to the second half, and then performs
    element-wise multiplication between the first half and the activated second half.

    Reference:
        - "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    
    Example:
        >>> import torch
        >>> from merlin.tabular.models.common.layers.activations import ReGLU
        >>> regl = ReGLU()
        >>> x = torch.randn(2, 4)  # Input tensor with shape (batch_size, features)
        >>> output = regl(x)
        >>> print(output.shape)  # Output tensor will have shape (batch_size, features / 2)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ReGLU activation function.

        Args:
            x (torch.Tensor): Input tensor of shape (..., 2 * d)

        Returns:
            torch.Tensor: Output tensor of shape (..., d)
        """
        assert x.size(-1) % 2 == 0, "The last dimension of input must be even."
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.relu(x2)


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""
    def forward(self, x):
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return F.gelu(x)


class GEGLU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function with gating mechanism.
        ϕ(x) = x1 * GELU(x2)
    where x is split into two halves: x1 and x2.

    GEGLU is a variant of the GLU (Gated Linear Unit) activation function.
    It splits the input tensor into two halves along the last dimension,
    applies the GELU activation to the second half, and then performs
    element-wise multiplication between the first half and the activated second half.

    Reference:
        - "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    
    Example:
        >>> import torch
        >>> from merlin.tabular.models.common.layers.activations import GEGULU
        >>> geglu = GEGULU()
        >>> x = torch.randn(2, 4)  # Input tensor with shape (batch_size, features)
        >>> output = geglu(x)
        >>> print(output.shape)  # Output tensor will have shape (batch_size, features / 2)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GEGLU activation function.

        Args:
            x (torch.Tensor): Input tensor of shape (..., 2 * d)

        Returns:
            torch.Tensor: Output tensor of shape (..., d)
        """
        assert x.size(-1) % 2 == 0, "The last dimension of input must be even."
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.gelu(x2)


class ELU(nn.Module):
    """Exponential Linear Unit (ELU) activation function."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)

class CELU(nn.Module):
    """Continuously Differentiable Exponential Linear Unit (CELU) activation function."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.celu(x, alpha=self.alpha)
