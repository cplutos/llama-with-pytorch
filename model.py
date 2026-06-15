import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32  # transformer block 堆叠数量
    n_heads: int = 32  # heads 中 Q 的数量，回顾MQA GQA
    n_kv_heads: Optional[int] = None  # heads 中 K 和 V 的数量
    vocab_size: int = -1  # 这个值在加载分词器设置
    multiple_of: int = 256  # FFN 网络中隐藏神经元数量
    ffn_dim_multiplier: Optional[float] = None  # 当使用GQA之后，K和V的数量会减少，但是增加FFN中的神经元数量
    norm_eps: float = 1e-5

    # 参数给 KV cache 所用
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 预先计算Rope中需要的mθ
    assert head_dim % 2 == 0, "必须可以被2整除，因为公式中 d/2"

    # 构建theta 参数
    # 根据论文中的公式实现
    theta_numerator = torch.arange(0, head_dim, 2).float()  # 2(i - 1 )
    theta = 1.0 / (theta ** (theta_numerator / head_dim))  # 10000^(-2(i-1)/d)

    # 构建m参数，代表着position位置
    m = torch.arange(seq_len, device=device)

    # 接下来 mθ 两个序列内积， 这里我们要得到所有的排列组合，用torch.outer
    # 这样每个position 都有一组mθ 值
    freqs = torch.outer(m, theta).float()

    # 我们可以用极坐标形式计算复数
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 1. 将x token向量中的dimension个值分组， 2个值为一组
    # 2. 然后将其转换为复数形式
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # 3. 乘上我们准备好的矩阵
    x_rotated = x_complex * freqs_complex

    # 4. 将复数a+ib形式中的a和b 提取出来
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 公式中的g 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, Dim)
        # torch.rsqrt() 简单来说就是对每个开平方根，然后取倒数
        # -1 是对最后一个维度求平均
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self._norm(x.floor()).type_as(x) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        # MHA
        return x
    else:
        return (
            x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads

        # 指明一个query of head 对应多少个重复的repeated keys and values of head
        self.n_req = self.n_heads_q // self.n_kv_heads

        # 4096 / 32 = 128
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, 1, Dim)
        batch_size, seq_len, _ = x.shape

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用Rope
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)

        # 因为前面把 cache 全部初始化为 0 ，所以这里"append"其实就是将对应信息赋值
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # 为了后面去计算self attention
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # 重复 keys and values 以达到 queries 的数量
        keys = repeat_kv(keys, self.n_req)
        values = repeat_kv(values, self.n_req)

        # (B, 1, H_B, Head_Dim) -> (B, H_B, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_len_KV) --> (B, H_Q, 1, Seq_Len_KV) 做转置
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=1).type_as(xq)

        # (B, H_Q, 1, Seq_Len_KV) @ (B, H_Q,Seq_Len_KV, Head_Dim ) --> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q * Head_Dim) -> (B, 1, Dim)
        # contiguous()方法用于解决运行时出现错误：RuntimeError: Input is not contiguous
        output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)  # (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # hidden_size = 7；multiple = 5；现在是7但是想要比7大的第一个五的倍数
        # (7 + 5 - 1) // 5 = 2 --> 5 * 2 = 10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # SwiGLU 是Swish激活函数和GLU函数的结合
        # SwiGLU(A, B) = A * Swish(B)
        swish = F.silu(self.w1(x))
        # 升维
        x_v = self.w3(x)
        x = swish * x_v
        # 降维
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        # 4096 / 32 = 128
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Attention之前需要归一化
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # feedward 之前需要归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "必须设定词大小"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # input Embedding
        self.tok_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            # EncoderBlock 是之后要去实现的input Embedding block
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 稍后会展开去讲， 现在先写在这里
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // args.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 这里实现的是inference，所以每次传入一个token，那么 seq_len 一直都是 1
        # tokens 的形状是 （B, seq_len）
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "每次都是一个token"

        # （B，seq_len）-> (B, seq_len, dim)
        h = self.tok_embedding(tokens)
        # 先去计算positional encoding相关信息
        # 根据位置[start_pos, start_pos+seq_len] 获取 （m,theta）
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # 连续应用encoder layers / transformer block
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int , max_batch_size: int,
              device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert  len(checkpoints) > 0, f"checkpoint 文件没有在{checkpoints_dir}找到"
            chk_path = checkpoints[0]
            print(f'加载模型文件...{chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'模型文件加载完成，耗时{time.time() - prev_time:.2f}秒')
            prev_time = time.time()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        model = Transformer(model_args).to(device)

        if load_model:
            # 从checkpoints中删除repo.freqs, 因为我们前面通过precompute_theta_pos_frequencies计算了
            del checkpoint["repo.freqs"]
            model.load_state_dict(checkpoint)
