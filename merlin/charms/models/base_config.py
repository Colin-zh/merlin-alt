"""
Modified from torchkeras, licensed under Apache 2.0.
Original source: https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/
Modifications made: Adapted for use in this project.

Original copyright: Copyright (c) lyhue1991
Modified work copyright: Copyright (c) Colin-zh

See LICENSE-APACHE for full license terms.
"""

import os
import re
from dataclasses import MISSING, dataclass, make_dataclass, fields, field, asdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from merlin.charms.models.common import heads


def get_inferred_config(ds):
    pass

def safe_merge_config(config, inferred_config):
    pass

@dataclass
class ModelConfig:
    """
    Base class for model configuration.

    Args:
        task (str): The type of task, choices are: [`regression`, `binary`, `classification`,
                `backbone`]. `backbone` is a task which considers the model as a backbone to 
                generate features. Mostly used internally for SSL and related tasks.
        
        head (Optional[str]): The type of head to use. Should be one of the heads defined in 
                `merlin.charms.common.heads`. Defaults to `LinearHead`.

        head_config (Optional[Dict]): The config as a dict which defines the head. If empty, will 
                be initialized as default linear head.

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical 
                column as a list of tuples (cardinality, embedding_dim). If empty, will infer 
                using the cardinality of the categorical column using the rule 
                min(50, (x + 1) // 2).
        
        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults 
                to 0.0.
        
        batch_norm_continuouse_input (bool): Whether to apply batch norm to the continuous 
                input. Defaults to True.

        loss (Optional[str]): The loss function to be applied. By Default, it is MSELoss for 
                regression, BCEWithLogitsLoss for binary and CrossEntropyLoss for classification. 
                Unless you are sure what you are doing, leave it at MSELoss or L1Loss for 
                regression and CrossEntropyLoss for classification
    """

    task: str = field(
        metadata={
            "help": "The type of task, choices are: [`regression`, `binary`, `classification`, "
                    "`backbone`]. `backbone` is a task which considers the model as a backbone to "
                    "generate features. Mostly used internally for SSL and related tasks.",
            "choices": ["regression", "binary", "classification", "backbone"],
        }
    )

    head: Optional[str] = field(
        default="LinearHead",
        metadata={
            "help": "The type of head to use. Should be one of the heads defined in "
                    "`merlin.charms.common.heads`. Defaults to `LinearHead`.",
        }
    )

    head_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"layers": ""},
        metadata={
            "help": "The config as a dict which defines the head. If empty, will be initialized "
            "as default linear head.",
        },
    )

    embedding_dims: Optional[List[tuple]] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of "
                    "tuples (cardinality, embedding_dim). If empty, will infer using the "
                    "cardinality of the categorical column using the rule min(50, (x + 1) // 2).",
        },
    )

    embedding_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.0.",
        },
    )

    batch_norm_continuouse_input: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply batch norm to the continuous input. Defaults to True.",
        },
    )

    loss: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to be applied. By Default, it is MSELoss for regression, "
                    "BCEWithLogitsLoss for binary and CrossEntropyLoss for classification. Unless you "
                    "are sure what you are doing, leave it at MSELoss or L1Loss for regression and "
                    "CrossEntropyLoss for classification.",
        },
    )

    _module_src: str = field(default="models")
    _model_name: str = field(default="Model")
    _backbone_name: str= field(default="Backbone")
    _config_name: str = field(default="Config")

    def __post_init__(self):
        if self.task == "regression":
            self.loss = self.loss or "MSELoss"
        elif self.task == "binary":
            self.loss = self.loss or "BCEWithLogitsLoss"
        elif self.task in ("multiclass", "classification"):
            self.loss = self.loss or "CrossEntropyLoss"
        # TODO backbone
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of "
                f"{self.__dataclass_fields__['task'].metadata['choices']}"
            )
        if self.task !="backbone":
            assert self.head in dir(heads.blocks), f"{self.head} is not a valid head."
            _head_callable = getattr(heads.blocks, self.head)
            ideal_head_config = _head_callable._config_template()
            invvalid_keys = set(self.head_config).difference(set(ideal_head_config.__dict__))
            assert len(invvalid_keys) == 0, f"`head_config` has invalid keys: {invvalid_keys}"
        
        # For Custom models, setting these values for compatibility
        if not hasattr(self, "_config_name"):
            self._config_name = type(self).__name__
        if not hasattr(self, "_model_name"):
            self._model_name = re.sub(r"[Cc]onfig", "Model", self._config_name)
        if not hasattr(self, "_backbone_name"):
            self._backbone_name = re.sub(r"[Cc]onfig", "Backbone", self._config_name)
        
    def merge_dataset_config(self, ds):
        inferred_config = get_inferred_config(ds)
        merged_config = safe_merge_config(self, inferred_config)
        return merged_config