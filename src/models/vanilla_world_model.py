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
from .transformerv2 import Transformer, VanillaTransformerConfig

from utils import init_weights, LossWithIntermediateLosses


@dataclass
class VanillaWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class VanillaWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: VanillaTransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim),
                                            nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.reward_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.end_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

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

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.reward_token, std=.02)
        torch.nn.init.normal_(self.end_token, std=0.02)
        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, tokens: torch.LongTensor):
        B, num_steps = tokens.size()
        prev_steps = 0

        sequences = self.embedder(tokens, num_steps, prev_steps)
        reward_tokens = self.reward_token.expand(B, -1, -1)
        end_tokens = self.end_token.expand(B, -1, -1)
        sequences = torch.cat((sequences, reward_tokens, end_tokens), dim=1)  # (B, num_steps+2, C)
        sequences = sequences + self.pos_emb(prev_steps + torch.arange(num_steps + 2, device=tokens.device))

        x = self.transformer(sequences)

        # -3: action token, -2: reward token, -1: end token
        logits_observations = self.head_observations(x[:, -self.config.tokens_per_block+1-3:-3])
        logits_rewards = self.head_rewards(x[:, -2])
        logits_ends = self.head_ends(x[:, -1])

        return VanillaWorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        prev_obs_tokens = obs_tokens[:, :-1]  # (B, l, k)
        prev_act_tokens = rearrange(batch['actions'][:, :-1], 'b l -> b l 1')
        prev_tokens = rearrange(torch.cat((prev_obs_tokens, prev_act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        outputs = self(prev_tokens)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        loss_obs = F.cross_entropy(rearrange(outputs.logits_observations, 'b t o -> (b t) o'), labels_observations)
        loss_rewards = F.cross_entropy(outputs.logits_rewards, labels_rewards)
        loss_ends = F.cross_entropy(outputs.logits_ends, labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens, rewards, ends, mask_padding):
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding[:, -1])
        labels_observations = obs_tokens[:, -1].masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens[:, -1]), -100)
        labels_rewards = (rewards[:, -2].sign() + 1).long()
        labels_ends = ends[:, -2]
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)