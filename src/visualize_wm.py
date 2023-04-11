from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from einops import rearrange
import shutil
import os, glob

from envs import SingleProcessEnv, MultiProcessEnv
from models.world_model import WorldModel
from models.vanilla_world_model import VanillaWorldModel
from models.mask_world_model import MaskWorldModel
from envs.world_model_env import WorldModelEnv
from envs.vanilla_world_model_env import VanillaWorldModelEnv
from envs.mask_world_model_env import MaskWorldModelEnv
from utils import extract_state_dict, RandomHeuristic
import imageio
import numpy as np
from models.transformer import TransformerConfig
from models.transformerv2 import VanillaTransformerConfig
from models.transformerv3 import MaskTransformerConfig

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):

    for f in glob.glob('/home/huyingdong/iris/visualization/*'):
        os.remove(f)

    device = torch.device(cfg.common.device)

    def create_env(cfg_env, num_envs):
        env_fn = partial(instantiate, config=cfg_env)
        return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

    env = create_env(cfg.env.train, cfg.collection.train.num_envs)

    # ---------------------------------------------------------
    iris_tokenizer = instantiate(cfg.tokenizer).to(device)
    iris_wm_config = TransformerConfig(tokens_per_block=17, max_blocks=20, attention='causal', num_layers=10, num_heads=4,
                                       embed_dim=256, embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
    iris_world_model = WorldModel(obs_vocab_size=iris_tokenizer.vocab_size, act_vocab_size=env.num_actions,
                                  config=iris_wm_config).to(device)
    path_to_checkpoint = '/home/huyingdong/iris/pretrained_models/025710_iris-efficient_MsPacmanNoFrameskip-v4_seed1/checkpoints/last.pt'
    agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
    iris_tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
    iris_world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
    iris_wm_env = WorldModelEnv(iris_tokenizer, iris_world_model, device)

    # ---------------------------------------------------------
    vanilla_tokenizer = instantiate(cfg.tokenizer).to(device)
    vanilla_wm_config = VanillaTransformerConfig(tokens_per_block=17, max_blocks=4, num_layers=12, num_heads=4,
                                                 embed_dim=256, embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
    vanilla_world_model = VanillaWorldModel(obs_vocab_size=vanilla_tokenizer.vocab_size, act_vocab_size=env.num_actions,
                                            config=vanilla_wm_config).to(device)
    path_to_checkpoint = '/home/huyingdong/iris/pretrained_models/030505_VanillaWM-efficient_MsPacmanNoFrameskip-v4_seed1/checkpoints/last.pt'
    agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
    vanilla_tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
    vanilla_world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
    vanilla_wm_env = VanillaWorldModelEnv(vanilla_tokenizer, vanilla_world_model, device)

    # ---------------------------------------------------------
    mask_tokenizer = instantiate(cfg.tokenizer).to(device)
    mask_wm_config = MaskTransformerConfig(tokens_per_block=16, max_blocks=4, encoder_num_layers=6, decoder_num_layers=6, num_heads=4,
                                           embed_dim=256, embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, num_iter=6)
    mask_world_model = MaskWorldModel(obs_vocab_size=mask_tokenizer.vocab_size, act_vocab_size=env.num_actions,
                                      config=mask_wm_config).to(device)
    path_to_checkpoint = '/home/huyingdong/iris/pretrained_models/134300_MaskWM-efficient_MsPacmanNoFrameskip-v4_seed1/checkpoints/last.pt'
    agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
    mask_tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
    mask_world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
    mask_wm_env = MaskWorldModelEnv(mask_tokenizer, mask_world_model, device)

    # Agent
    random_agent = RandomHeuristic(env.num_actions)

    num_episodes = 10
    imagine_horizon = 20
    for i in range(num_episodes):
        env_obs = env.reset()
        env_image_list = [env_obs[0]]
        env_obs = rearrange(torch.FloatTensor(env_obs).div(255), 'n h w c -> n c h w').to(device)

        _ = iris_wm_env.reset_from_initial_observations(env_obs)
        _ = vanilla_wm_env.reset_from_initial_observations(env_obs)
        _ = mask_wm_env.reset_from_initial_observations(env_obs)
        iris_wmenv_obs = np.array(iris_wm_env.render())
        iris_wmenv_image_list = [iris_wmenv_obs]
        vanilla_wmenv_obs = np.array(vanilla_wm_env.render())
        vanilla_wmenv_image_list = [vanilla_wmenv_obs]
        mask_wmenv_obs = np.array(mask_wm_env.render())
        mask_wmenv_image_list = [mask_wmenv_obs]

        for step in range(imagine_horizon):

            act = random_agent.act(env_obs).cpu().numpy()

            env_obs, reward, done, _ = env.step(act)
            env_image_list.append(env_obs[0])
            env_obs = rearrange(torch.FloatTensor(env_obs).div(255), 'n h w c -> n c h w').to(device)

            iris_wm_env.step(act)
            iris_wmenv_image_list.append(np.array(iris_wm_env.render()))
            vanilla_wm_env.step(act)
            vanilla_wmenv_image_list.append(np.array(vanilla_wm_env.render()))
            mask_wm_env.step(act)
            mask_wmenv_image_list.append(np.array(mask_wm_env.render()))

        concat_image_list = []
        for t in range(len(env_image_list)):
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs[0].imshow(env_image_list[t])
            axs[0].set_title("environment")
            axs[1].imshow(iris_wmenv_image_list[t])
            axs[1].set_title("IRIS")
            axs[2].imshow(vanilla_wmenv_image_list[t])
            axs[2].set_title("Vanilla Transformer")
            axs[3].imshow(mask_wmenv_image_list[t])
            axs[3].set_title("Masked Transformer")
            for ax in axs:
                ax.set_axis_off()
            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            concat_image_list.append(image_array)

        output_path = '/home/huyingdong/iris/visualization/episode{}.gif'.format(i)
        imageio.mimsave(output_path, concat_image_list, duration=0.5)


if __name__ == '__main__':
    main()




