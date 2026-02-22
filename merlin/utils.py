"""
This file is directly copied from torchkeras licensed under Apache 2.0.
Original source: https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/
Original copyright: Copyright (c) lyhue1991

See full Apache 2.0 license in the project root LICENSE-APACHE file.
"""

import datetime
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def printlog(info: str) -> None:
    """Print log information with datetime"""
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "==========" * 8 + "%s" % now_time)
    print(info + "...\n\n")


def seed_everything(seed: int = 42) -> int:
    """Set random seed for reproducibility.

    Args:
        seed (int, optional): The seed value. Defaults to 42.
    """
    print(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def delete_object(obj: object) -> None:
    """Delete an object and free up memory.

    Args:
        obj (object): The object to be deleted.
    """
    import gc
    obj = None
    gc.collect()
    del obj
    with torch.no_grad():
        torch.cuda.empty_cache()


def colorful(obj, color="red", display_type="plain"):
    # 彩色输出格式：
    # 设置颜色开始 ：\033[显示方式;前景色;背景色m
    # 说明：
    # 前景色            背景色           颜色
    # ---------------------------------------
    # 30                40              黑色
    # 31                41              红色
    # 32                42              绿色
    # 33                43              黃色
    # 34                44              蓝色
    # 35                45              紫红色
    # 36                46              青蓝色
    # 37                47              白色
    # 显示方式           意义
    # -------------------------
    # 0                终端默认设置
    # 1                高亮显示
    # 4                使用下划线
    # 5                闪烁
    # 7                反白显示
    # 8                不可见
    color_dict = {"black": "30", "red": "31", "green": "32", "yellow": "33",
                  "blue": "34", "purple": "35", "cyan": "36", "white": "37"}
    display_type_dict = {"plain": "0", "highlight": "1", "underline": "4",
                         "shine": "5", "inverse": "7", "invisible": "8"}
    s = str(obj)
    color_code = color_dict.get(color, "")
    display = display_type_dict.get(display_type, "")
    out = '\033[{};{}m'.format(display, color_code) + s + '\033[0m'
    return out


def namespace2dict(namespace: str) -> Dict:
    """Convert a string representation of a namespace to a dictionary.

    Args:
        namespace (str): The string representation of the namespace.

    Returns:
        Dict: The converted dictionary.
    """
    from argparse import Namespace
    result = {}
    for k, v in vars(namespace).items():
        if not isinstance(v, Namespace):
            result[k] = v
        else:
            v_dic = namespace2dict(v)
            for v_k, v_v in v_dic.items():
                result[f"{k}.{v_k}"] = v_v
    return result


def is_jupyter() -> bool:
    """Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter, False otherwise.
    """
    import contextlib
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False


def parse_args(parser, use_default = is_jupyter()):
    import argparse
    parser.add_help = not use_default
    if not use_default:
        return parser.parse_args()
    else:
        args_dict = {}
        for action in parser._actions:
            args_dict[action.dest] = action.default
        return argparse.Namespace(**args_dict)
