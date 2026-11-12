"""
KidLM-5M Model Definition (Llama-style Decoder-only)
Compatible with llama2.c / runq.c format for Seeed Studio XIAO ESP32-S3
"""

import math
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 4096
    hidden_dim: int = 688
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # xq: (bsz, seqlen, n_heads, head_dim)
    # reshape for complex multiplication
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim // 2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Causal mask
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)

        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        unique = len(set(p.data_ptr() for p in self.parameters()))
        return total


def export_llama2_c_q8(model: Transformer, filepath: str):
    """
    Exports weights in INT8 format compatible with llama2.c / runq.c
    Header:
      magic (0x616b3432), version (1), dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
    """
    args = model.args
    with open(filepath, "wb") as f:
        # Header (256 bytes)
        header = struct.pack(
            "iiiiiiii",
            args.dim,
            args.hidden_dim,
            args.n_layers,
            args.n_heads,
            args.n_heads, # n_kv_heads
            args.vocab_size,
            args.max_seq_len,
            0 # reserved
        )
        f.write(header)
        # Pad header to 256 bytes
        f.write(b"\x00" * (256 - len(header)))

        # Helper to quantize and write
        def write_weight(tensor, quantize=True):
            if quantize and tensor.dim() >= 2:
                scale = tensor.abs().max().item() / 127.0
                if scale == 0: scale = 1.0
                q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
                # Write scale as float32 first, then int8 bytes
                f.write(struct.pack("f", scale))
                f.write(q.cpu().numpy().tobytes())
            else:
                f.write(tensor.float().cpu().numpy().tobytes())

        # 1. Token embeddings
        write_weight(model.tok_embeddings.weight, quantize=True)

        # 2. Layers
        for l in model.layers:
            write_weight(l.attention_norm.weight, quantize=False)
            write_weight(l.attention.wq.weight, quantize=True)
            write_weight(l.attention.wk.weight, quantize=True)
            write_weight(l.attention.wv.weight, quantize=True)
            write_weight(l.attention.wo.weight, quantize=True)
            write_weight(l.ffn_norm.weight, quantize=False)
            write_weight(l.feed_forward.w1.weight, quantize=True)
            write_weight(l.feed_forward.w2.weight, quantize=True)
            write_weight(l.feed_forward.w3.weight, quantize=True)

        # 3. Final norm
        write_weight(model.norm.weight, quantize=False)


if __name__ == "__main__":
    args = ModelArgs()
    m = Transformer(args)
    params = m.count_parameters()
    print(f"KidLM-5M initialized!")
    print(f"Total Parameters: {params:,} (~{params * 4 / 1024 / 1024:.2f} MB in FP32, ~{params / 1024 / 1024:.2f} MB in INT8)")
