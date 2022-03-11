import os
from pathlib import Path
import sys
import time
from functools import partial
import math
import random
import copy
from collections import deque
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

import scipy
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.display import export_gif, Arrow3D
from py_diff_pd.common.rl_sim import DiffPDTask, make_water_snake_3d, tensor, MyGaussianActorCriticNet, get_logger, MeanStdNormalizer, AdaSim, IndSim

from deep_rl.utils import generate_tag, Config, set_one_thread
from deep_rl.agent import BaseNet, FCBody, PPOAgent
from deep_rl.network import GaussianActorCriticNet


def ppo_ada():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    set_one_thread()

    folder = Path('water_snake').resolve() / 'Ada_PPO'
    ckpt_folder = folder / 'checkpoints'
    video_folder = folder / 'videos'
    folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'game': 'water_snake'
    }

    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 4

    config.task_fn = lambda: DiffPDTask(make_water_snake_3d, AdaSim, seed, config.num_workers, False) # pylint: disable=no-member
    config.eval_env = DiffPDTask(make_water_snake_3d, AdaSim, seed, 1, True)

    config.network_fn = lambda: MyGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh),
        critic_body=FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 1000
    config.eval_interval = config.rollout_length * config.num_workers
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = config.rollout_length * config.num_workers
    config.save_interval = config.rollout_length * config.num_workers * 10
    config.max_steps = 1e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer(read_only=True)

    agent = PPOAgent(config)
    agent.logger = get_logger(folder)
    config = agent.config

    init_ckpt = torch.load(folder.parent / 'Ada' / 'checkpoints' / '0.pth', map_location='cpu')['state_dict']
    with torch.no_grad():
        for name, param in agent.network.named_parameters():
            if name == 'actor_body.layers.0.weight':
                param.copy_(init_ckpt['layers.0.linear.weight'])
            elif name == 'actor_body.layers.0.bias':
                param.copy_(init_ckpt['layers.0.linear.bias'])
            elif name == 'actor_body.layers.1.weight':
                param.copy_(init_ckpt['layers.1.linear.weight'])
            elif name == 'actor_body.layers.1.bias':
                param.copy_(init_ckpt['layers.1.linear.bias'])
            elif name == 'fc_action.weight':
                param.copy_(init_ckpt['layers.2.weight'])
            elif name == 'fc_action.bias':
                param.copy_(init_ckpt['layers.2.bias'])

    print(agent.network)

    log = []

    t0 = time.time()
    while True:
        last_step = config.max_steps and agent.total_steps >= config.max_steps

        if last_step or agent.total_steps % config.save_interval == 0:
            agent.save(ckpt_folder / f'{agent.total_steps}.pth')
        if last_step or agent.total_steps % config.log_interval == 0:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if last_step or agent.total_steps % config.eval_interval == 0:
            config.state_normalizer.set_read_only()
            state = config.eval_env.reset()
            total_reward = 0.0
            with torch.no_grad():
                while True:
                    state = config.state_normalizer(state)
                    action = agent.network(state)['mean'].cpu().detach().numpy()
                    state, reward, done, info = config.eval_env.step(action)
                    total_reward += reward
                    if done:
                        break
            agent.logger.add_scalar('episode_reward', total_reward, agent.total_steps)
            log.append([agent.total_steps, total_reward])
        if last_step:
            agent.close()
            break
        config.state_normalizer.set_read_only()
        agent.step()

    torch.save(log, folder / 'log.pth')


def ppo_ind():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    set_one_thread()

    folder = Path('water_snake').resolve() / 'Ind_PPO'
    ckpt_folder = folder / 'checkpoints'
    video_folder = folder / 'videos'
    folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'game': 'water_snake'
    }

    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 4

    config.task_fn = lambda: DiffPDTask(make_water_snake_3d, IndSim, seed, config.num_workers, False) # pylint: disable=no-member
    config.eval_env = DiffPDTask(make_water_snake_3d, IndSim, seed, 1, True)

    config.network_fn = lambda: MyGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh),
        critic_body=FCBody(config.state_dim, hidden_units=(64, 64), gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 1000
    config.eval_interval = config.rollout_length * config.num_workers
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = config.rollout_length * config.num_workers
    config.save_interval = config.rollout_length * config.num_workers * 10
    config.max_steps = 1e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer(read_only=True)

    agent = PPOAgent(config)
    agent.logger = get_logger(folder)
    config = agent.config

    init_ckpt = torch.load(folder.parent / 'Ind' / 'checkpoints' / '0.pth', map_location='cpu')['state_dict']
    with torch.no_grad():
        for name, param in agent.network.named_parameters():
            if name == 'actor_body.layers.0.weight':
                param.copy_(init_ckpt['layers.0.linear.weight'])
            elif name == 'actor_body.layers.0.bias':
                param.copy_(init_ckpt['layers.0.linear.bias'])
            elif name == 'actor_body.layers.1.weight':
                param.copy_(init_ckpt['layers.1.linear.weight'])
            elif name == 'actor_body.layers.1.bias':
                param.copy_(init_ckpt['layers.1.linear.bias'])
            elif name == 'fc_action.weight':
                param.copy_(init_ckpt['layers.2.weight'])
            elif name == 'fc_action.bias':
                param.copy_(init_ckpt['layers.2.bias'])

    print(agent.network)

    log = []

    t0 = time.time()
    while True:
        last_step = config.max_steps and agent.total_steps >= config.max_steps

        if last_step or agent.total_steps % config.save_interval == 0:
            agent.save(ckpt_folder / f'{agent.total_steps}.pth')
        if last_step or agent.total_steps % config.log_interval == 0:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if last_step or agent.total_steps % config.eval_interval == 0:
            config.state_normalizer.set_read_only()
            state = config.eval_env.reset()
            total_reward = 0.0
            with torch.no_grad():
                while True:
                    state = config.state_normalizer(state)
                    action = agent.network(state)['mean'].cpu().detach().numpy()
                    state, reward, done, info = config.eval_env.step(action)
                    total_reward += reward
                    if done:
                        break
            agent.logger.add_scalar('episode_reward', total_reward, agent.total_steps)
            log.append([agent.total_steps, total_reward])
        if last_step:
            agent.close()
            break
        config.state_normalizer.set_read_only()
        agent.step()

    torch.save(log, folder / 'log.pth')

if __name__ == "__main__":
    ppo_ada()
