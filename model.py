from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32 # transformer block 堆叠数量
    n_heads: int = 32 # heads 中 Q 的数量，回顾MQA GQA
    n_kv_heads: Optional[int] = None # heads 中 K 和 V 的数量
    vocab_size: int = -1 # 这个值在加载分词器设置
    multiple_of: int = 256 # FFN 网络中隐藏神经元数量
    ffn_dim_multiplier: Optional[float] = None # 当使用GQA之后，K和V的数量会减少，但是增加FFN中的神经元数量
    norm_eps: float = 1e-5

    # 参数给 KV cache 所用
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

