from dataclasses import dataclass, field
from typing import Optional

from ..base_config import ModelConfig


@dataclass
class AutoIntConfig(ModelConfig):
    input_embed_dim: int = field(
        default=32,
        metadata={"help": "Embedding dimension for categorical/numerical tokens."},
    )

    attn_layers: int = field(
        default=2,
        metadata={"help": "Number of interacting (self-attention) layers."},
    )

    attn_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads."},
    )

    attn_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout applied to attention and residuals."},
    )

    use_residual: bool = field(
        default=True,
        metadata={"help": "Whether to add residual connections in interacting layers."},
    )

    deep_layers: str = field(
        default="128-64-32",
        metadata={"help": "Hyphen-separated MLP layer sizes after interactions."},
    )

    deep_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout for the post-attention MLP."},
    )

    embedding_initialization: Optional[str] = field(
        default="xavier_uniform",
        metadata={"help": "Initialization scheme for embeddings."},
    )

    _module_src: str = field(default="models.autoint")
    _model_name: str = field(default="AutoIntModel")
    _backbone_name: str = field(default="AutoIntBackbone")
    _config_name: str = field(default="AutoIntConfig")
