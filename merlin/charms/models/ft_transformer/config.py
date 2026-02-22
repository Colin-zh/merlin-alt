from dataclasses import dataclass, field
from typing import Optional

from ..base_config import ModelConfig


@dataclass
class FTTransformerConfig(ModelConfig):
    """Configuration for FT-Transformer model.
    
    Args:
        input_embed_dim (int): The embedding dimension for the input categorical features. Defaults to 32

        embedding_initialization (Optional[str]]): The initialization method for the embedding layers. 
                Defaults to "kaiming_unifrom". Choices are ['kaiming_unifrom', 'kaiming_normal', 
                'xavier_uniform', 'xavier_normal', 'random'].
        
        embedding_bias (bool): Whether to use bias in the embedding layers. Defaults to True.

        share_embedding (bool):  The flag turns on shared embeddings in the input embedding process. The 
                key idea here is to have an embedding for the feature as a whole along with embeddings of 
                each unique values of that column. Defaults to False.
    
        share_embedding_strategy (str): The strategy to use for sharing embeddings. Choices are 'add' and
                'fraction'. Defaults to 'fraction'. 1. `add` - A separate embedding for the feature is 
                added to the embedding of the unique values of the feature. 2. `fraction` - A fraction of 
                the input embedding is reserved for the shared embedding of the feature.
        
        shared_embedding_fraction (float): The fraction of the input embedding reserved for the shared 
                embedding when `share_embedding_strategy` is 'fraction'. Should be less than one. 
                Defaults to 0.25.
    
        attn_feature_importance (bool): Whether to compute feature importance using attention weights. 
                If you are facing memory issues, you can turn off feature importance which will not save 
                the attention weights. Defaults to True
    
        num_heads (int): The number of attention heads in the transformer blocks. Defaults to 8

        num_attn_blocks (int): The number of transformer blocks in the backbone. Defaults to 6

        transformer_head_dim (Optional[int]]): The dimension of each attention head. If None, it will be 
                set to be same as input_dim. Defaults to None.
        
        attn_dropout (float): The dropout rate for the attention layers. Defaults to 0.1

        add_norm_dropout (float): The dropout rate for the add & norm layers. Defaults to 0.1

        ff_dropout (float): The dropout rate for the feedforward layers. Defaults to 0.1

        ff_hidden_multiplier (int): Multiple by which the Positionwise FF layer scales the input. Defaults
                to 4
        
        transformer_activation (str): The activation function to use in the transformer blocks. Choices 
                are 'relu', 'gelu', 'silu', 'leaky_relu', geglu, etc. Defaults to 'geglu'.
        
        task (str): The type of task for the model. `backbone` is a task which considers the model as a
                backbone to generate features. Mostly used internally for SSL and related tasks. Choices
                are 'backbone', 'regression', 'classification'.
        
        head (Optional[str]]): The type of head to use for the model. Should be one of the heads defined in
                `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`, `LinearHead`, `MixtureDensityHead`].

        head_config (Optional[dict]]): The configuration for the head. If empty, default configuration
                will be used as linear head. Defaults to None.

        embedding_dims (Optional[List]): The dimensions of the embeddings for each categorical feature as
                a list of tuples (cardinality, dimension). IF left empty, will infer using the cardinality
                of the categorical column using the rule min(50, (x + 1) // 2)
        
        embedding_dropout (float): The dropout rate for the embedding layers. Defaults to 0.0

        batch_norm_continuous_input (bool): Whether to apply batch normalization to continuous inputs.
                Defaults to True.
        
        loss (Optional[str]]): The loss function to use for training. If None, will use default loss based
                on the task. By default, uses MSELoss for regression and CrossEntropyLoss for classification.

    """

    input_embed_dim: int = field(
        default=32,
        metadata={"description": "The embedding dimension for the input categorical features."},
    )
    embedding_initialization: Optional[str] = field(
        default="kaiming_unifrom",
        metadata={
            "description": (
                "The initialization method for the embedding layers. Choices are "
                "['kaiming_unifrom', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'random']"
            )
        },
    )
    embedding_bias: bool = field(
        default=True,
        metadata={"description": "Whether to use bias in the embedding layers."},
    )
    share_embedding: bool = field(
        default=False,
        metadata={
            "description": (
                "The flag turns on shared embeddings in the input embedding process. The key idea "
                "here is to have an embedding for the feature as a whole along with embeddings of "
                "each unique values of that column."
            )
        },
    )
    share_embedding_strategy: str = field(
        default="fraction",
        metadata={
            "description": (
                "The strategy to use for sharing embeddings. Choices are 'add' and 'fraction'. 1. `add` - "
                "A separate embedding for the feature is added to the embedding of the unique values "
                "of the feature. 2. `fraction` - A fraction of the input embedding is reserved for the shared "
                "embedding of the feature."
            )
        },
    )
    shared_embedding_fraction: float = field(
        default=0.25,
        metadata={
            "description": (
                "The fraction of the input embedding reserved for the shared embedding when "
                "`share_embedding_strategy` is 'fraction'. Should be less than one."
            )
        },
    )
    attn_feature_importance: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to compute feature importance using attention weights. If you are facing "
                "memory issues, you can turn off feature importance which will not save the attention weights."
            )
        },
    )
    num_heads: int = field(
        default=8,
        metadata={"description": "The number of attention heads in the transformer blocks."},
    )
    num_attn_blocks: int = field(
        default=6,
        metadata={"description": "The number of transformer blocks in the backbone."},
    )
    transformer_head_dim: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "The dimension of each attention head. If None, it will be set to be same as input_dim."
            )
        },
    )
    attn_dropout: float = field(
        default=0.1,
        metadata={"description": "The dropout rate for the attention layers."},
    )
    add_norm_dropout: float = field(
        default=0.1,
        metadata={"description": "The dropout rate for the add & norm layers."},
    )
    ff_dropout: float = field(
        default=0.1,
        metadata={"description": "The dropout rate for the feedforward layers."},
    )
    ff_hidden_multiplier: int = field(
        default=4,
        metadata={
            "description": "Multiple by which the Positionwise FF layer scales the input."
        },
    )
    transformer_activation: str = field(
        default="geglu",
        metadata={
            "description": (
                "The activation function to use in the transformer blocks. Choices are 'relu', "
                "'gelu', 'silu', 'leaky_relu', geglu, etc."
            )
        },
    )

    _module_scr: str = field(default="tabular.models.ft_transformer")
    _model_name: str = field(default="FTTransformer")
    _backbone_name: str = field(default="FTTransformerBackbone")
    _config_name: str = field(default="FTTransformerConfig")
