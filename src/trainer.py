from collections import defaultdict
from functools import partial
from pathlib import Path
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import csv
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import ActorCritic
from models.world_model import WorldModel, BlockCausalWorldModel
from models.vanilla_world_model import VanillaWorldModel
from models.mask_world_model import MaskWorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.root_dir = Path.cwd()
        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        # VQVAE tokenizer
        tokenizer = instantiate(cfg.tokenizer)
        # World Model
        if cfg.common.world_model_type == 'default':
            world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        elif cfg.common.world_model_type == 'block':
            world_model = BlockCausalWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        elif cfg.common.world_model_type == 'vanilla':
            world_model = VanillaWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        elif cfg.common.world_model_type == 'mask':
            world_model = MaskWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        else:
            raise NotImplementedError
        self.world_model_type = cfg.common.world_model_type
        # Actor Critic
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)
        self.use_amp = cfg.training.use_amp
        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

        # # compile the model
        # if cfg.common.compile:
        #     print("compiling the model...")
        #     self.agent = torch.compile(self.agent)  # requires PyTorch 2.0

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                self.test_dataset.clear()
                to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                to_log += self.eval_agent(epoch)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

            # Final evaluation on 100 episodes
            if epoch == self.cfg.common.epochs:
                self.cfg.collection.test.config.num_episodes = 100
                self.test_dataset.clear()
                to_log = self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
                self.save_final_result(to_log=to_log)

        self.finish()

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        w = self.cfg.training.sampling_weights

        # train tokenizer
        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer, sequence_length=1, sample_from_start=True, sample_fixed_history=False, sampling_weights=w, **cfg_tokenizer)
        self.agent.tokenizer.eval()

        # train world model
        if epoch > cfg_world_model.start_after_epochs:
            sequence_length = self.cfg.common.sequence_length
            sample_fixed_history = False
            if self.world_model_type == 'vanilla' or self.world_model_type == 'mask':
                sample_fixed_history = True
                sequence_length = self.cfg.common.sequence_length + 1
            metrics_world_model = self.train_component(self.agent.world_model, self.optimizer_world_model,
                                                       sequence_length=sequence_length, sample_from_start=True, sample_fixed_history=sample_fixed_history,
                                                       sampling_weights=w, tokenizer=self.agent.tokenizer, **cfg_world_model)
        self.agent.world_model.eval()

        # train actor critic
        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic,
                                                        sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, sample_fixed_history=False,
                                                        sampling_weights=w, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
        self.agent.actor_critic.eval()

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int, batch_num_samples: int,
                        grad_acc_steps: int, max_grad_norm: Optional[float], sequence_length: int, sampling_weights: Optional[Tuple[float]],
                        sample_from_start: bool, sample_fixed_history: bool, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sampling_weights, sample_from_start, sample_fixed_history)
                batch = self._to_device(batch)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                    loss_total_step = losses.loss_total
                self.loss_scaler.scale(loss_total_step).backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                self.loss_scaler.unscale_(optimizer)  # Unscales the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)  # Since the gradients of optimizer's assigned params are unscaled, clips as usual:

            # optimizer.step()
            self.loss_scaler.step(optimizer)
            self.loss_scaler.update()

        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1)

        # eval world model
        if epoch > cfg_world_model.start_after_epochs:
            sequence_length = self.cfg.common.sequence_length
            if self.world_model_type == 'vanilla' or self.world_model_type == 'mask':
                sequence_length = self.cfg.common.sequence_length + 1
            metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples,
                                                      sequence_length=sequence_length, tokenizer=self.agent.tokenizer)

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def save_final_result(self, to_log):
        for metrics in to_log:
            metrics = {k.replace('test_dataset', 'final_eval'): v for k, v in metrics.items()}
            wandb.log({**metrics})

        avg_episode_return = to_log[-1]['test_dataset/return']
        eval_data = {
            'game': self.cfg.env.train.id,
            'seed': self.cfg.common.seed,
            'return': avg_episode_return
        }

        file_name = self.cfg.wandb.group.split('_')[0] + '.csv'
        csv_path = self.root_dir.parent / file_name
        should_write_header = False if csv_path.exists() else True
        csv_file = open(csv_path, 'a')
        csv_writer = csv.DictWriter(csv_file, fieldnames=eval_data.keys(), restval=0.0)
        if should_write_header:
            csv_writer.writeheader()
        csv_writer.writerow(eval_data)
        csv_file.flush()
        csv_file.close()

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            # optimizer and scaler
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            torch.save({
                "scaler": self.loss_scaler.state_dict()
            }, self.ckpt_dir / 'scaler.pt')
            # dataset
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        # optimizer and scaler
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        if os.path.exists(self.ckpt_dir / 'scaler.pt'):
            ckpt_scaler = torch.load(self.ckpt_dir / 'scaler.pt', map_location=self.device)
            self.loss_scaler.load_state_dict(ckpt_scaler['scaler'])

        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        if not self.cfg.common.save_train_set:
            # delete episodes in training dataset
            shutil.rmtree(self.ckpt_dir / 'dataset')
        if not self.cfg.common.save_media:
            # delete episodes in media
            shutil.rmtree(self.episode_dir)

        wandb.finish()