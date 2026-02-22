from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LinearHeadConfig:
    """Configuration for a linear head in a neural network.

    Args:
        layers (str): Hyphen-separated number of layers and units in the classification/regression head.
            E.g. 32-64-32. Default is just a mapping from input to output.

        activation (str, optional): Activation function to use between layers. Default is relu.
            Supported activations: 'relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'leaky_relu'.
        
        dropout (float, optional): Dropout rate to use between layers. Default is 0.0 (no dropout).

        use_bias (bool, optional): Whether to use bias in the linear layers. Default is True.

        use_batch_norm (bool, optional): Whether to use batch normalization between layers. Default is False.

        initialization (str, optional): Weight initialization method. Default is 'kaiming'. Supported 
            initializations: 'kaiming', 'xavier', 'random', 'normal', 'uniform', 'orthogonal'.
    """
    
    layers: str = field(
        default="",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification/regression head. "
                    "E.g. 32-64-32. Default is just a mapping from input to output."
        }
    )
    activation: Optional[str] = field(
        default="relu",
        metadata={
            "help": "Activation function to use between layers. Default is relu. "
                    "Supported activations: 'relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'leaky_relu'."
        }
    )
    dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Dropout rate to use between layers. Default is 0.0 (no dropout)."
        }
    )
    use_bias: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use bias in the linear layers. Default is True."
        }
    )
    use_batch_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use batch normalization between layers. Default is False."
        }
    )
    initialization: Optional[str] = field(
        default="kaiming",
        metadata={
            "help": "Weight initialization method. Default is 'kaiming'. Supported "
                    "initializations: 'kaiming', 'xavier', 'random', 'normal', 'uniform', 'orthogonal'."
        }
    )
