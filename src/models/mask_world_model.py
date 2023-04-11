from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from einops import rearrange
import math
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


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
        self.num_iter = config.num_iter

        # --------------------------------------------------------------------------
        # encoder specifics
        self.obs_embed = nn.Embedding(obs_vocab_size, config.embed_dim)
        self.act_embed = nn.Embedding(act_vocab_size, config.embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, config.tokens_per_block, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.embed_pdrop)

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
        self.context_pos_embed = nn.Parameter(torch.zeros(1, (config.tokens_per_block + 1) * 4, config.embed_dim))

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

        len_keep = round(L * (1 - mask_ratio))
        len_keep = min(len_keep, L - 1)

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

    def scores_masking(self, x, mask_ratio, scores, is_ranking_scores=True, is_confidence_scores=False):
        """
        Tokens with low "ranking scores" are kept, while tokens with high "ranking scores" are removed (masked)
        Tokens with high "confidence scores" are kept, while tokens with low "confidence scores" are removed (masked)
        """
        assert is_ranking_scores ^ is_confidence_scores
        N, L, D = x.shape  # batch, length, dim

        len_keep = round(L * (1 - mask_ratio))
        len_keep = min(len_keep, L - 1)  # mask at least 1 token

        if is_confidence_scores:
            scores = -scores
        ids_shuffle = torch.argsort(scores, dim=1)  # ascend: small is keep, large is remove (mask)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return x_masked, mask, ids_restore

    def forward_context(self, prev_obs_tokens, prev_act_tokens):
        assert prev_obs_tokens.size(1) == prev_act_tokens.size(1)

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
        return prev_feat

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.obs_embed(x)
        x = x + self.encoder_pos_embed
        x = self.pos_drop(x)

        # masking
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

        # append reward token and end token
        reward_tokens = self.reward_token.expand(B, -1, -1)
        end_tokens = self.end_token.expand(B, -1, -1)
        x = torch.cat((x, reward_tokens, end_tokens), dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed
        prev_feat = prev_feat + self.context_pos_embed

        # apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, prev_feat)
        x = self.decoder_norm(x)
        return x

    def forward(self, prev_obs_tokens, prev_act_tokens, curr_obs_tokens):

        prev_feat = self.forward_context(prev_obs_tokens, prev_act_tokens)

        t = random.uniform(0, 1)
        mask_ratio = math.cos(0.5 * math.pi * t)          # sample mask ratio
        curr_feat, mask, ids_restore = self.forward_encoder(curr_obs_tokens, mask_ratio)

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
        loss_obs = F.cross_entropy(rearrange(outputs.logits_observations, 'b t o -> (b t) o'), labels_observations)
        loss_rewards = F.cross_entropy(outputs.logits_rewards, labels_rewards)
        loss_ends = F.cross_entropy(outputs.logits_ends, labels_ends)
        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens, rewards, ends, mask_padding, obs_mask):
        assert torch.all(ends.sum(dim=1) <= 1)   # at most 1 done
        mask_fill = torch.logical_not(mask_padding[:, -1])
        labels_observations = obs_tokens[:, -1].masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens[:, -1]), -100)
        labels_observations = labels_observations.masked_fill(torch.logical_not(obs_mask), -100)   # compute the loss only on masked patches
        labels_rewards = (rewards[:, -2].sign() + 1).long()
        labels_ends = ends[:, -2]
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)

    @torch.no_grad()
    def generate_muse(self, prev_obs_tokens, prev_act_tokens):
        # See https://github.com/lucidrains/muse-maskgit-pytorch
        B, context_length, K = prev_obs_tokens.size()
        obs_tokens = torch.zeros((B, K), dtype=torch.long, device=prev_obs_tokens.device)
        scores = torch.zeros((B, K), device=prev_obs_tokens.device)  # "ranking scores"

        starting_temperature = 1.0  # TODO: set to ?

        prev_feat = self.forward_context(prev_obs_tokens, prev_act_tokens)

        ##################
        mask_list = []
        obs_list = []
        ##################

        for t, steps_until_x0 in zip(torch.linspace(0, 1, self.num_iter), reversed(range(self.num_iter))):

            x = self.obs_embed(obs_tokens)
            x = x + self.encoder_pos_embed
            x = self.pos_drop(x)

            # Step 1 (Mask Schedule): Compute the mask ratio according to the cosine mask scheduling function
            mask_ratio = math.cos(0.5 * math.pi * t)

            # Step 2 (Mask): Tokens with high "ranking scores" are removed (masked)
            x, mask, ids_restore = self.scores_masking(x, mask_ratio, scores=scores, is_ranking_scores=True, is_confidence_scores=False)
            mask_list.append(mask)

            # Step 3 (Predict): Given the unmasked tokens, predict the probability of the remaining masked tokens
            for blk in self.encoder_blocks:
                x = blk(x)
            x = self.encoder_norm(x)
            decoder_out = self.forward_decoder(x, prev_feat, ids_restore)
            logits_observations = self.head_observations(decoder_out[:, :-2])

            # Step 4 (Sample): At each masked location, sample a token based on the probability
            filtered_logits = top_k(logits_observations, thres=0.9)
            temperature = starting_temperature * (steps_until_x0 / self.num_iter)   # temperature is annealed
            pred_tokens = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            obs_tokens = torch.where(mask, pred_tokens, obs_tokens)   # (B, K)
            obs_list.append(obs_tokens)
            # Use prediction probability as "confidence scores", use 1 - probability as "ranking scores"
            # Tokens with low "ranking scores" are kept
            probs_without_temperature = logits_observations.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(dim=2, index=pred_tokens[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~mask, -1e5)    # For the unmasked position, set its ranking score to -1e5

            if steps_until_x0 == 0:
                logits_rewards = self.head_rewards(decoder_out[:, -2])
                logits_ends = self.head_ends(decoder_out[:, -1])
                reward = Categorical(logits=logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1  # (B,)
                done = Categorical(logits=logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)      # (B,)
                return obs_tokens, reward, done

            # if steps_until_x0 == 0:
            #     logits_rewards = self.head_rewards(decoder_out[:, -2])
            #     logits_ends = self.head_ends(decoder_out[:, -1])
            #     reward = Categorical(logits=logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1  # (B,)
            #     done = Categorical(logits=logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)      # (B,)
            #     return obs_tokens, reward, done, mask_list, obs_list

    @torch.no_grad()
    def generate_maskgit(self, prev_obs_tokens, prev_act_tokens):
        # See https://github.com/google-research/maskgit
        B, context_length, K = prev_obs_tokens.size()
        obs_tokens = torch.zeros((B, K), dtype=torch.long, device=prev_obs_tokens.device)
        confidence_scores = torch.zeros((B, K), device=prev_obs_tokens.device)  # "confidence scores"

        choice_temperature = 3.0

        prev_feat = self.forward_context(prev_obs_tokens, prev_act_tokens)

        # start from a blank canvas with all the tokens masked out
        x = self.obs_embed(obs_tokens)
        x = x + self.encoder_pos_embed
        x = self.pos_drop(x)
        x, mask, ids_restore = self.scores_masking(x, mask_ratio=1.0, scores=confidence_scores, is_ranking_scores=False, is_confidence_scores=True)

        for step in range(self.num_iter):
            # ------------------------------------------------------------------------------------
            # Step 1 (Predict): Given the unmasked tokens, predict the probability of the remaining masked tokens
            for blk in self.encoder_blocks:
                x = blk(x)
            x = self.encoder_norm(x)
            decoder_out = self.forward_decoder(x, prev_feat, ids_restore)
            logits_observations = self.head_observations(decoder_out[:, :-2])
            # ------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------
            # Step 2 (Sample): At each masked location, sample a token
            pred_tokens = Categorical(logits=logits_observations).sample()
            obs_tokens = torch.where(mask, pred_tokens, obs_tokens)
            # Use prediction scores plus a noise sampled from a Gumbel distribution multiplied by temperature as "confidence scores"
            probs = torch.nn.functional.softmax(logits_observations, dim=-1)
            selected_probs = torch.squeeze(torch.gather(probs, dim=-1, index=pred_tokens[..., None]), -1)
            selected_probs = torch.where(mask, selected_probs.double(), torch.inf).float()  # The "confidence scores" at the unmasked locations are set to Inf
            t = 1. * (step + 1) / self.num_iter
            temperature = choice_temperature * (1 - t)
            confidence_scores = torch.log(selected_probs) + torch.Tensor(temperature * np.random.gumbel(size=selected_probs.shape)).to(selected_probs.device)
            # ------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------
            # Step 3 (Mask Schedule): Compute the mask ratio according to the cosine mask scheduling function
            mask_ratio = math.cos(0.5 * math.pi * t)
            # ------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------
            # Step 4 (Mask): Tokens with low "confidence scores" are removed (masked)
            x = self.obs_embed(obs_tokens)
            x = x + self.encoder_pos_embed
            x = self.pos_drop(x)
            x, mask, ids_restore = self.scores_masking(x, mask_ratio, scores=confidence_scores, is_ranking_scores=False, is_confidence_scores=True)
            # ------------------------------------------------------------------------------------

        logits_rewards = self.head_rewards(decoder_out[:, -2])
        logits_ends = self.head_ends(decoder_out[:, -1])
        reward = Categorical(logits=logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1  # (B,)
        done = Categorical(logits=logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)      # (B,)
        return obs_tokens, reward, done

