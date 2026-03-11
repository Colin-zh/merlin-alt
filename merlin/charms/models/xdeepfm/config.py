from dataclasses import dataclass, field
from typing import Optional

from ..base_config import ModelConfig


@dataclass
class XDeepFMConfig(ModelConfig):
    input_embed_dim: int = field(
        default=32,
        metadata={"help": "Embedding dimension for categorical/numerical features."},
    )

    cin_layer_sizes: str = field(
        default="128-64-32",
        metadata={"help": "Hyphen-separated feature map sizes for CIN layers."},
    )

    cin_split_half: bool = field(
        default=True,
        metadata={"help": "Whether to halve feature maps per CIN layer (except last)."},
    )

    deep_layers: str = field(
        default="128-64-32",
        metadata={"help": "Hyphen-separated MLP sizes for the deep component."},
    )

    deep_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout applied in the deep MLP."},
    )

    embedding_initialization: Optional[str] = field(
        default="xavier_uniform",
        metadata={"help": "Initialization scheme for embeddings."},
    )

    _module_src: str = field(default="models.xdeepfm")
    _model_name: str = field(default="XDeepFMModel")
    _backbone_name: str = field(default="XDeepFMBackbone")
    _config_name: str = field(default="XDeepFMConfig")
