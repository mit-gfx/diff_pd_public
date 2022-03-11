import os
from pathlib import Path
import sys
import time
from functools import partial
import itertools
import math
import random
import copy
from collections import deque
import argparse

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
from py_diff_pd.common.rl_sim import DiffPDTask, make_water_snake_3d, tensor, MyDeterministicActorCriticNet, get_logger, MeanStdNormalizer, AdaSim, IndSim

from sac_utils.sac import SAC
from sac_utils.replay_memory import ReplayMemory


def sac_ada():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    folder = Path('water_snake').resolve() / 'Ada_SAC'
    ckpt_folder = folder / 'checkpoints'
    video_folder = folder / 'videos'
    folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(Ï„) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter Î± determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust Î± (default: True)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                        help='hidden size (default: 64)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args()
    args.seed = seed

    env = make_water_snake_3d(AdaSim, seed, 0)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    init_ckpt = torch.load(folder.parent / 'Ada' / 'checkpoints' / '0.pth', map_location='cpu')['state_dict']
    with torch.no_grad():
        for name, param in agent.policy.named_parameters():
            if name == 'linear1.weight':
                param.copy_(init_ckpt['layers.0.linear.weight'])
            elif name == 'linear1.bias':
                param.copy_(init_ckpt['layers.0.linear.bias'])
            elif name == 'linear2.weight':
                param.copy_(init_ckpt['layers.1.linear.weight'])
            elif name == 'linear2.bias':
                param.copy_(init_ckpt['layers.1.linear.bias'])
            elif name == 'mean_linear.weight':
                param.copy_(init_ckpt['layers.2.weight'])
            elif name == 'mean_linear.bias':
                param.copy_(init_ckpt['layers.2.bias'])

    print(agent.policy)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    writer = SummaryWriter(folder, purge_step=0)
    log = []

    for i_episode in itertools.count(0):

        if i_episode % 10 == 0:
            with torch.no_grad():
                state = env.reset()
                episode_reward = 0
                eval_len = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    eval_len += 1

                    state = next_state

                writer.add_scalar('episode_reward', episode_reward, total_numsteps)

                print("----------------------------------------")
                print("Test Episodes: {}, Reward: {}, Length: {}".format(1, episode_reward, eval_len))
                print("----------------------------------------")

                log.append([total_numsteps, episode_reward])

                ckpt = {
                    'policy': agent.policy.state_dict(),
                    'critic': agent.critic.state_dict()
                }
                torch.save(ckpt, ckpt_folder / f'{total_numsteps}.pth')

        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == 200 else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    writer.flush()
    torch.save(log, folder / 'log.pth')

    env.close()


if __name__ == "__main__":
    sac_ada()
