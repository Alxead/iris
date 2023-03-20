import random
from collections import deque
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class VanillaWorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.context_length = self.world_model.config.max_blocks
        self.obs_stack = None
        self.act_stack = None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens
        self.obs_tokens = obs_tokens

        B, C, H, W = observations.shape
        pad_images = torch.zeros(B, self.context_length - 1, C, H, W, device=observations.device)
        pad_obs_tokens = self.tokenizer.encode(pad_images, should_preprocess=True).tokens

        self.obs_stack = deque([], maxlen=self.context_length)
        self.act_stack = deque([], maxlen=self.context_length)
        for i in range(self.context_length - 1):
            self.obs_stack.append(pad_obs_tokens[:, i].unsqueeze(1))
            self.act_stack.append(torch.zeros(B, 1, dtype=torch.long, device=observations.device))
        self.obs_stack.append(obs_tokens.unsqueeze(1))

        return self.decode_obs_tokens()

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)
        self.act_stack.append(token)

        prev_obs_tokens = torch.cat(list(self.obs_stack), dim=1)
        prev_act_tokens = torch.cat(list(self.act_stack), dim=1)
        prev_act_tokens = rearrange(prev_act_tokens, 'b l -> b l 1')
        prev_tokens = rearrange(torch.cat((prev_obs_tokens, prev_act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        outputs_wm = self.world_model(prev_tokens)

        reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1  # (B,)
        done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)  # (B,)
        self.obs_tokens = Categorical(logits=outputs_wm.logits_observations).sample()
        obs = self.decode_obs_tokens() if should_predict_next_obs else None

        self.obs_stack.append(self.obs_tokens.unsqueeze(1))

        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]