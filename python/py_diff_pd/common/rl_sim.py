import random
from pathlib import Path
import math
import logging
import os

from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym import spaces
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

import numpy as np

from deep_rl import Task, Config, Logger, BaseNormalizer
from deep_rl.component.envs import DummyVecEnv, SubprocVecEnv, OriginalReturnWrapper
from deep_rl.network import GaussianActorCriticNet, DeterministicActorCriticNet, NoisyLinear, layer_init

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector, StdIntVector, QuadMesh2d
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.display import export_gif, Arrow3D

from baselines.common.running_mean_std import RunningMeanStd


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms and self.read_only:
            return np.clip(x, -self.clip, self.clip)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float64)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


class LayerNormFCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(LayerNormFCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [nn.Sequential(*[
                    nn.Linear(dim_in, dim_out, bias=False),
                    nn.LayerNorm(dim_out, elementwise_affine=True)
                    ]) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class MyDeterministicActorCriticNet(DeterministicActorCriticNet):
    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def save(self, path):
        torch.save({
            'checkpoints': self.network.state_dict(),
            'normalizer': self.config.state_normalizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self.network.load_state_dict(state_dict['checkpoints'])
        self.config.state_normalizer.load_state_dict(state_dict['normalizer'])


class MyGaussianActorCriticNet(GaussianActorCriticNet):
    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean,
                'v': v}

    def save(self, path):
        torch.save({
            'checkpoints': self.network.state_dict(),
            'normalizer': self.config.state_normalizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self.network.load_state_dict(state_dict['checkpoints'])
        self.config.state_normalizer.load_state_dict(state_dict['normalizer'])


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

def get_logger(path, log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return Logger(logger, str(path), log_level)


class DiffPDTask(Task):
    def __init__(
            self,
            env_fn,
            sim_class,
            seed,
            num_envs=1,
            single_process=True,
            episode_life=True,
        ):

        envs = [make_env(env_fn, sim_class, seed, i, episode_life) for i in range(num_envs)]
        if num_envs == 1 or single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = 'diffpd'
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


def make_env(env_fn, *args, **kwargs):
    def _thunk():
        env = env_fn(*args, **kwargs)
        env = OriginalReturnWrapper(env)

        return env
    return _thunk


def make_water_snake_3d(sim_class, seed, rank, *args, **kwargs):

    os.system("taskset -p 0xff %d" % os.getpid())

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    folder = Path('water_snake').resolve()
    folder.mkdir(parents=True, exist_ok=True)

    # Mesh parameters.
    cell_nums = [20, 2, 2]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.zeros((3,))
    bin_file_name = str(folder / 'water_snake.bin')
    voxels = np.ones(cell_nums)

    voxel_indices, vertex_indices = generate_hex_mesh(voxels, dx, origin, bin_file_name, write=False)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'pd_eigen'
    opt = {
        'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0,
        'thread_ct': 1, 'use_bfgs': 1, 'bfgs_history_size': 10
    }

    deformable = HexDeformable()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0, 0]   # Velocity of the water.
    # # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_face(mesh)
    deformable.AddStateForce(
        'hydrodynamics', np.concatenate(
            [[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), ndarray(surface_faces).ravel()]))

    # Add actuation.
    # ******************** <- muscle
    # |                  | <- body
    # |                  | <- body
    # ******************** <- muscle

    all_muscles = []
    shared_muscles = []
    for i in [0, cell_nums[2] - 1]:
        muscle_pair = []
        for j in [0, cell_nums[1] - 1]:
            indices = voxel_indices[:, j, i].tolist()
            deformable.AddActuation(1e5, [1.0, 0.0, 0.0], indices)
            muscle_pair.append(indices)
        shared_muscles.append(muscle_pair)
    all_muscles.append(shared_muscles)
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    num_frames = 200
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    arrow_target_data = np.array([-1, 0, 0], dtype=np.float64)

    w_sideward = 10.0
    w_face = 0.0

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    mid_line = vertex_indices[:, mid_y, mid_z]
    center = vertex_indices[mid_x, mid_y, mid_z]

    face_head = vertex_indices[0, mid_y, mid_z]
    face_tail = vertex_indices[2, mid_y, mid_z]

    def get_state_(sim, q_, v_, a_=None, f_ext_=None):
        q_center = q_.reshape((-1, 3))[center]
        v_center = v_.reshape((-1, 3))[center]

        q_mid_line_rel = q_.reshape((-1, 3))[mid_line] - q_center
        v_mid_line = v_.reshape((-1, 3))[mid_line]
        state = [
            v_center,
            q_mid_line_rel.ravel(),
            v_mid_line.ravel(),
        ]
        return np.concatenate(state).copy()

    def get_reward_(sim, q_, v_, a_=None, f_ext_=None):

        v_center = np.mean(v_.reshape((-1, 3))[mid_line], axis=0)
        face_dir = q_.reshape((-1, 3))[face_head] - q_.reshape((-1, 3))[face_tail]
        face_dir = face_dir / np.linalg.norm(face_dir)

        # forward loss
        forward_reward = np.dot(v_center, arrow_target_data)

        # sideward loss
        cross = np.cross(v_center, arrow_target_data)
        sideward_reward = -np.dot(cross, cross)

        # face loss
        face_reward = np.dot(face_dir, arrow_target_data)

        return forward_reward + w_sideward * sideward_reward + w_face * face_reward

    def get_done_(sim, q_, v_, a_, f_ext_):
        if sim.frame >= sim.num_frames:
            return True
        return False

    setattr(sim_class, 'get_state_', get_state_)
    setattr(sim_class, 'get_reward_', get_reward_)
    setattr(sim_class, 'get_done_', get_done_)

    sim = sim_class(
        deformable, mesh, center, dofs, act_dofs, method, dt, opt, num_frames)

    if sim_class is AdaSim:
        action_shape = (len(all_muscles),)
    elif sim_class is IndSim:
        muscle_dofs = 0
        for shared_muscles in all_muscles:
            muscle_dofs += len(shared_muscles[0][0])
        action_shape = (muscle_dofs,)
    else:
        raise ValueError('invalid simulation class')

    sim.set_action_space(action_shape)

    sim.observation_space.seed(seed + rank)
    sim.action_space.seed(seed + rank)

    return sim



def make_starfish_3d(sim_class, seed, rank, *args, **kwargs):

    os.system("taskset -p 0xff %d" % os.getpid())

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    folder = Path('starfish').resolve()
    folder.mkdir(parents=True, exist_ok=True)

    # Mesh parameters
    limb_width = 2
    limb_length = 10
    limb_depth = 2

    cell_nums = [limb_length * 2 + limb_width, limb_length * 2 + limb_width, limb_depth]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.zeros((3,))
    bin_file_name = str(folder / 'starfish.bin')

    voxels = np.ones(cell_nums)
    voxels[:limb_length, :limb_length] = 0
    voxels[:limb_length, -limb_length:] = 0
    voxels[-limb_length:, :limb_length] = 0
    voxels[-limb_length:, -limb_length:] = 0

    voxel_indices, vertex_indices = generate_hex_mesh(
        voxels, dx, origin, bin_file_name, write=False)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'pd_eigen'
    opt = {
        'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0,
        'thread_ct': 1, 'use_bfgs': 1, 'bfgs_history_size': 10
    }

    deformable = HexDeformable()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0, 0]   # Velocity of the water.
    # # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_face(mesh)
    deformable.AddStateForce(
        'hydrodynamics', np.concatenate(
            [[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), ndarray(surface_faces).ravel()]))

    # Add actuation.
    all_muscles = []
    muscle_pairs = []

    muscle_stiffness = 1e5

    for move in [range(limb_length - 1, -1, -1), range(-limb_length, 0)]:
        for fix in [limb_length, limb_length + limb_width - 1]:

            muscle_pair = []
            for depth in [0, limb_depth - 1]:
                indices = [int(voxel_indices[fix, m, depth]) for m in move]
                deformable.AddActuation(muscle_stiffness, [0.0, 1.0, 0.0], indices)
                muscle_pair.append(indices)
            muscle_pairs.append(muscle_pair)

            muscle_pair = []
            for depth in [0, limb_depth - 1]:
                indices = [int(voxel_indices[m, fix, depth]) for m in move]
                deformable.AddActuation(muscle_stiffness, [1.0, 0.0, 0.0], indices)
                muscle_pair.append(indices)
            muscle_pairs.append(muscle_pair)

    all_muscles = [
        [muscle_pairs[0], muscle_pairs[2]],
        [muscle_pairs[1], muscle_pairs[3]],
        [muscle_pairs[4], muscle_pairs[6]],
        [muscle_pairs[5], muscle_pairs[7]],
    ]
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    num_frames = 200
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    arrow_target_data = np.array([0, 0, 1], dtype=np.float64)

    w_sideward = 1.0
    w_face = 0.0

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    center = vertex_indices[mid_x, mid_y, mid_z]

    face_head = vertex_indices[mid_x, mid_y, -1]
    face_tail = vertex_indices[mid_x, mid_y, 0]

    mid_plane = np.array([
        vertex_indices[mid_x, :limb_length, mid_z],
        vertex_indices[mid_x, -limb_length:, mid_z],
        vertex_indices[:limb_length, mid_y, mid_z],
        vertex_indices[-limb_length:, mid_y, mid_z],
    ]).ravel()

    def get_state_(sim, q_, v_, a_=None, f_ext_=None):
        q_center = q_.reshape((-1, 3))[center]
        v_center = v_.reshape((-1, 3))[center]

        q_mid_line_rel = q_.reshape((-1, 3))[mid_plane] - q_center
        v_mid_line = v_.reshape((-1, 3))[mid_plane]
        state = [
            v_center,
            q_mid_line_rel.ravel(),
            v_mid_line.ravel(),
        ]
        return np.concatenate(state).copy()

    def get_reward_(sim, q_, v_, a_=None, f_ext_=None):

        v_center = v_.reshape((-1, 3))[center]
        face_dir = q_.reshape((-1, 3))[face_head] - q_.reshape((-1, 3))[face_tail]
        face_dir = face_dir / np.linalg.norm(face_dir)

        # forward loss
        forward_reward = np.dot(v_center, arrow_target_data)

        # sideward loss
        cross = np.cross(v_center, arrow_target_data)
        sideward_reward = -np.dot(cross, cross)

        # face loss
        face_reward = np.dot(face_dir, arrow_target_data)

        return forward_reward + w_sideward * sideward_reward + w_face * face_reward

    def get_done_(sim, q_, v_, a_, f_ext_):
        if sim.frame >= sim.num_frames:
            return True
        return False

    setattr(sim_class, 'get_state_', get_state_)
    setattr(sim_class, 'get_reward_', get_reward_)
    setattr(sim_class, 'get_done_', get_done_)

    sim = sim_class(
        deformable, mesh, center, dofs, act_dofs, method, dt, opt, num_frames)

    if sim_class is AdaSim:
        action_shape = (len(all_muscles),)
    elif sim_class is IndSim:
        muscle_dofs = 0
        for shared_muscles in all_muscles:
            muscle_dofs += len(shared_muscles[0][0])
        action_shape = (muscle_dofs,)
    else:
        raise ValueError('invalid simulation class')

    sim.set_action_space(action_shape)

    sim.observation_space.seed(seed + rank)
    sim.action_space.seed(seed + rank)

    return sim


class AdaSim(gym.Env):

    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self, deformable, mesh, center,
            dofs, act_dofs, method, dt, option, num_frames
        ):

        super(AdaSim, self).__init__()
        self.deformable = deformable
        self.mesh = mesh
        self.center = center
        self.dofs = dofs
        self.act_dofs = act_dofs
        self.method = method
        self.dt = dt
        self.option = option
        self.num_frames = num_frames

        self.a_init = np.zeros(self.act_dofs)
        self.prev_a = None

        self.frame = 0

        self.q = None
        self.v = None

        if isinstance(mesh, Mesh2d):
            dim = 2
        elif isinstance(mesh, HexMesh3d):
            dim = 3
        else:
            raise ValueError(f'invlaid mesh type: {type(mesh)}')

        q0 = ndarray(self.mesh.py_vertices())
        q0_center = q0.reshape((-1, dim))[center]
        self.q0 = (q0.reshape((-1, dim)) - q0_center).ravel()
        self.v0 = np.zeros_like(q0, dtype=np.float64)
        self.f_ext = np.zeros_like(q0, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=self.get_state_(self.q0, self.v0).shape, dtype=np.float64) # pylint: disable=no-member
        self.action_space = None
        self.reset()

    def set_action_space(self, action_shape):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float64)

    def get_action(self, action):

        if self.prev_a is None:
            prev_a = self.a_init.copy()
        else:
            prev_a = self.prev_a

        a = []
        pointer = 0

        for w, shared_muscles in zip(action, self.deformable.all_muscles):
            mu_pair = [0.5 * (np.abs(w) - w), 0.5 * (np.abs(w) + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                    pointer += len(muscle)
                    a_cord = np.concatenate([mu.reshape((1,)), prev_a_cord[:-1]])
                    a.append(a_cord)

        a = np.array(a).ravel()
        self.prev_a = a.copy()
        return 1 - a

    def step(self, a):

        self.frame += 1

        a = self.get_action(a)

        q_array = StdRealVector(self.q)
        v_array = StdRealVector(self.v)
        a_array = StdRealVector(a)
        f_ext_array = StdRealVector(self.f_ext)
        q_next_array = StdRealVector(self.dofs)
        v_next_array = StdRealVector(self.dofs)

        self.deformable.PyForward(
            self.method, q_array, v_array, a_array, f_ext_array,
            self.dt, self.option, q_next_array, v_next_array, StdIntVector(0))

        q = ndarray(q_next_array)
        v = ndarray(v_next_array)

        self.q, self.v = q.copy(), v.copy()

        state = self.get_state_(q, v, a, self.f_ext) # pylint: disable=no-member
        reward = self.get_reward_(q, v, a, self.f_ext) # pylint: disable=no-member
        done = self.get_done_(q, v, a, self.f_ext) # pylint: disable=no-member

        if done:
            self.reset()

        return state, reward, done, dict()

    def reset(self):
        self.frame = 0
        self.q = self.q0.copy()
        self.v = self.v0.copy()
        self.prev_a = None

        return self.get_state_(self.q, self.v) # pylint: disable=no-member


class IndSim(AdaSim):
    def get_action(self, action):

        a_shared_muscles = []
        a = []

        pointer = 0
        for shared_muscles in self.deformable.all_muscles:
            a_shared_muscles.append(action[pointer:pointer + len(shared_muscles[0][0])])

        for w, shared_muscles in zip(a_shared_muscles, self.deformable.all_muscles):
            mu_pair = [0.5 * (np.abs(w) - w), 0.5 * (np.abs(w) + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    a.append(mu)

        a = np.array(a).ravel()
        return 1 - a
