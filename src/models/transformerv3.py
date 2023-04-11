"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class MaskTransformerConfig:
    tokens_per_block: int
    max_blocks: int
    encoder_num_layers: int
    decoder_num_layers: int
    num_heads: int
    embed_dim: int
    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float
    num_iter: int

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks + 2  # reward token and end token


class SelfAttention(nn.Module):
    def __init__(self, config: MaskTransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: MaskTransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T1, C = x.size()
        _, T2, _ = context.size()

        q = self.query(x).view(B, T1, self.num_heads, C // self.num_heads).transpose(1, 2)         # (B, nh, T1, hs)
        k = self.key(context).view(B, T2, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T2, hs)
        v = self.value(context).view(B, T2, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T2, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')
        y = self.resid_drop(self.proj(y))
        return y


class EncoderBlock(nn.Module):
    def __init__(self, config: MaskTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: MaskTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ln2_context = nn.LayerNorm(config.embed_dim)
        self.ln3 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)         # self-attention
        self.cross_attn = CrossAttention(config)  # cross-attention
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), self.ln2_context(context))
        x = x + self.mlp(self.ln3(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: MaskTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_num_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # forward context
        # forward encoder
        # forward decoder

        x = self.drop(sequences)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = self.ln_f(x)
        return x