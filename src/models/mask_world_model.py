from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformerv3 import Transformer, MaskTransformerConfig

from utils import init_weights, LossWithIntermediateLosses


@dataclass
class MaskWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    mask: torch.bool    # 0 is keep, 1 is remove


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


class MaskWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: MaskTransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        # --------------------------------------------------------------------------
        # encoder specifics
        self.obs_embed = nn.Embedding(obs_vocab_size, config.embed_dim)
        self.act_embed = nn.Embedding(act_vocab_size, config.embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, config.tokens_per_block, config.embed_dim))
        self.pos_drop = nn.Dropout(config.embed_pdrop)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_num_layers)])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        self.act_encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, config.tokens_per_block + 2, config.embed_dim))
        # dropout? add position embedding for cross attention keys and values?
        self.reward_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.end_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_num_layers)])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # prediction heads
        self.head_observations = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, obs_vocab_size)
        )
        self.head_rewards = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 3)
        )
        self.head_ends = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 2)
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def __repr__(self) -> str:
        return "world_model"

    def initialize_weights(self):
        # TODO check init
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder_pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        torch.nn.init.normal_(self.reward_token, std=.02)
        torch.nn.init.normal_(self.end_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_context(self, prev_obs_tokens, prev_act_tokens):
        B, context_length, K = prev_obs_tokens.size()

        prev_obs_tokens = rearrange(prev_obs_tokens, 'b l k -> (b l) k')
        obs_sequences = self.obs_embed(prev_obs_tokens)
        obs_sequences = obs_sequences + self.encoder_pos_embed
        x = self.pos_drop(obs_sequences)

        # apply encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        x = rearrange(x, '(b l) k c -> b l k c', b=B, l=context_length)

        act_sequences = self.act_embed(prev_act_tokens)   # (B, L, C)
        act_sequences = self.act_encoder(act_sequences)   # (B, L, C)
        act_sequences = rearrange(act_sequences, 'b l c -> b l 1 c')

        prev_feat = rearrange(torch.cat((x, act_sequences), dim=2), 'b l k1 c-> b (l k1) c')
        # TODO: add position embedding?
        return prev_feat

    def forward_encoder(self, x, mask_ratio=0.75):
        # embed patches
        x = self.obs_embed(x)
        x = x + self.encoder_pos_embed
        x = self.pos_drop(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, prev_feat, ids_restore):
        # append mask tokens to sequence
        B, _, _ = x.size()
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # append reward token and env token
        reward_tokens = self.reward_token.expand(B, -1, -1)
        end_tokens = self.end_token.expand(B, -1, -1)
        x = torch.cat((x, reward_tokens, end_tokens), dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, prev_feat)
        x = self.decoder_norm(x)
        return x

    def forward(self, prev_obs_tokens, prev_act_tokens, curr_obs_tokens):

        prev_feat = self.forward_context(prev_obs_tokens, prev_act_tokens)

        curr_feat, mask, ids_restore = self.forward_encoder(curr_obs_tokens, mask_ratio=0.75)

        decoder_out = self.forward_decoder(curr_feat, prev_feat, ids_restore)

        logits_observations = self.head_observations(decoder_out[:, :-2])
        logits_rewards = self.head_rewards(decoder_out[:, -2])
        logits_ends = self.head_ends(decoder_out[:, -1])
        mask = mask.to(torch.bool)

        return MaskWorldModelOutput(decoder_out, logits_observations, logits_rewards, logits_ends, mask)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L+1, K)

        prev_obs_tokens = obs_tokens[:, :-1]        # (B, L, K)
        curr_obs_tokens = obs_tokens[:, -1]         # (B, K)
        prev_act_tokens = batch['actions'][:, :-1]  # (B, L)

        outputs = self(prev_obs_tokens, prev_act_tokens, curr_obs_tokens)

        # compute labels
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'],
                                                                                           obs_mask=outputs.mask)
        # compute loss

    def compute_labels_world_model(self, obs_tokens, rewards, ends, mask_padding, obs_mask):

        # TODO: add ignore index for mask obs

        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding[:, -1])
        labels_observations = obs_tokens[:, -1].masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens[:, -1]), -100)
        labels_rewards = (rewards[:, -2].sign() + 1).long()
        labels_ends = ends[:, -2]
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)

