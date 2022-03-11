import os
from pathlib import Path
import sys
import time
from functools import partial
import math
import random
import copy
import pprint
import argparse
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent))

import scipy
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.display import display_hex_mesh, export_gif
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.controller import AdaNNController, SnakeAdaNNController, IndNNController


class Display(object):
    def __init__(self, sim, controller, get_state, get_plot, init_qv, num_frames):
        self.sim = sim
        self.controller = controller
        self.get_state = get_state
        self.get_plot = get_plot
        self.init_qv = init_qv
        self.num_frames = num_frames

        self.stream = None
        self.fig, self.ax = None, None

        self.scatter = None
        self.arrow_target = None
        self.arrow_vel = None
        self.arrow_face = None
        self.title = None

        self.fpss = deque(maxlen=100)

    def save(self, filename):
        fps = 10
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d'))
        ani = animation.FuncAnimation(
            self.fig, self.update, self.num_frames, init_func=self.setup_plot, interval=1 / fps, repeat=False, blit=False)
        try:
            writer = animation.writers['avconv']
        except KeyError:
            writer = animation.writers['ffmpeg']
        writer = writer(fps=fps)
        ani.save(filename, writer=writer)
        plt.close(self.fig)


    def setup_plot(self):
        self.fpss = deque(maxlen=10)
        self.stream = self.data_stream()
        q, v_center, arrow_target_data, face_base, face_dir = next(self.stream)
        xs, ys, zs = q.numpy().T

        self.scatter = self.ax.scatter(xs, ys, zs, c='tab:blue')

        self.arrow_target = self.ax.arrow3D(
            (1, 1, 0),
            arrow_target_data.numpy(),
            mutation_scale=10,
            ec='tab:red', fc='tab:red')

        self.arrow_vel = self.ax.arrow3D(
            (1, 1, 0),
            v_center.numpy(),
            mutation_scale=10,
            ec='tab:green', fc='tab:green')

        self.arrow_face = self.ax.arrow3D(
            face_base.numpy(),
            face_dir.numpy(),
            mutation_scale=10,
            ec='tab:orange', fc='tab:orange')

        radius = 3
        self.ax.set_xlim([-radius, radius])
        self.ax.set_ylim([-radius, radius])
        self.ax.set_zlim([-radius, radius])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.title = self.ax.set_title('Interactive Animation step=0 fps=?', animated=True)

        return self.scatter, self.arrow_target, self.arrow_vel, self.arrow_face

    def update(self, i):
        tstart = time.time()
        q, v_center, arrow_target_data, face_base, face_dir = next(self.stream)
        tend = time.time()
        xs, ys, zs = q.numpy().T
        self.fpss.append(1 / (tend - tstart))

        self.scatter._offsets3d = (xs, ys, zs)  # pylint: disable=protected-access

        self.arrow_target.set_positions((1, 1, 0), arrow_target_data.numpy())
        self.arrow_vel.set_positions((1, 1, 0), v_center.numpy())
        self.arrow_face.set_positions(face_base.numpy(), face_dir.numpy())

        self.title.set_text(f'Interactive Animation step={i} fps={sum(self.fpss) / len(self.fpss):.2f}')

        return self.scatter, self.arrow_target, self.arrow_vel, self.arrow_face

    @torch.no_grad()
    def data_stream(self):

        self.controller.train(False)

        q, v = self.init_qv
        get_plot = self.get_plot
        get_state = self.get_state
        controller = self.controller
        sim = self.sim

        a = None

        for frame in range(1, self.num_frames + 1):

            yield get_plot(q, v)

            state = get_state(q, v)
            a = controller(state, a)
            q, v = sim(q=q, v=v, a=a)

        yield get_plot(q, v)


def main():

    num_epochs = 1000

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    folder = Path('starfish').resolve()
    sub_folder = folder / 'Ind'
    ckpt_folder = sub_folder / 'checkpoints'
    video_folder = sub_folder / 'videos'
    folder.mkdir(parents=True, exist_ok=True)
    sub_folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    video_folder.mkdir(parents=True, exist_ok=True)

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
        voxels, dx, origin, bin_file_name)
    mesh = HexMesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'pd_eigen'
    opt = {
        'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0,
        'thread_ct': 4, 'use_bfgs': 1, 'bfgs_history_size': 10
    }

    deformable = HexDeformable()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0, 0]   # Velocity of the water.
    # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
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
    f_ext = torch.zeros(dofs).detach()
    arrow_target_data = torch.Tensor([0, 0, 1]).detach()

    w_sideward = 1.0
    w_face = 0.0

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    center = vertex_indices[mid_x, mid_y, mid_z]

    face_head = vertex_indices[mid_x, mid_y, -1]
    face_tail = vertex_indices[mid_x, mid_y, 0]

    q0 = torch.as_tensor(ndarray(mesh.py_vertices())).detach()
    q0_center = q0.view(-1, 3)[center].detach()
    q0 = q0.view(-1, 3).sub(q0_center).view(-1).detach()

    v0 = torch.zeros(dofs).detach()

    mid_plane = np.array([
        vertex_indices[mid_x, :limb_length, mid_z],
        vertex_indices[mid_x, -limb_length:, mid_z],
        vertex_indices[:limb_length, mid_y, mid_z],
        vertex_indices[-limb_length:, mid_y, mid_z],
    ]).ravel()

    def get_state(q, v):
        q_center = q.view(-1, 3)[center]
        v_center = v.view(-1, 3)[center]

        q_mid_plane_rel = q.view(-1, 3)[mid_plane] - q_center.detach()
        v_mid_plane = v.view(-1, 3)[mid_plane]
        state = [
            v_center,
            q_mid_plane_rel.view(-1),
            v_mid_plane.view(-1),
        ]
        return torch.cat(state).unsqueeze(0)


    def get_plot(q, v):
        face_dir = q.view(-1, 3)[face_head] - q.view(-1, 3)[face_tail]
        face_dir = face_dir / face_dir.norm()
        return (
            (q.view(-1, 3)).clone().detach(),
            v.view(-1, 3)[center].clone().detach(),
            arrow_target_data.clone().detach(),
            q.view(-1, 3)[face_head].clone().detach(),
            face_dir.clone().detach())

    sim = Sim(deformable)

    controller = IndNNController(
        deformable, [get_state(q0, v0).size(1), 64, 64, len(all_muscles)], None, dropout=0.0)
    controller.reset_parameters()

    optimizer = optim.Adam(controller.parameters(), lr=0.001, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / num_epochs)

    q, v = q0, v0

    display = Display(
        partial(sim, dofs=dofs, act_dofs=act_dofs, method=method, f_ext=f_ext, dt=dt, option=opt),
        controller, get_state, get_plot, (q0, v0), num_frames
    )

    ckpt = {
        'state_dict': controller.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_folder / '0.pth')

    writer = SummaryWriter(sub_folder, purge_step=0)
    log = []

    for epoch in range(num_epochs):
        controller.train(True)

        forward_loss = 0
        sideward_loss = 0
        face_loss = 0

        a = None
        q, v = q0, v0

        for frame in range(1, num_frames + 1):
            state = get_state(q, v)
            a = controller(state, a)
            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)
            v_center = v.view(-1, 3)[center]
            q_center = q.view(-1, 3)[center].detach()
            face_dir = q.view(-1, 3)[face_head] - q.view(-1, 3)[face_tail]
            face_dir = face_dir / face_dir.norm()

            # forward loss
            dot = torch.dot(v_center, arrow_target_data)
            forward_loss += -dot

            # sideward loss
            cross = torch.cross(v_center, arrow_target_data)
            sideward_loss += torch.dot(cross, cross)

            # face loss
            face_loss += -torch.dot(face_dir, arrow_target_data)

        loss = forward_loss + w_sideward * sideward_loss + w_face * face_loss

        optimizer.zero_grad()

        loss.backward() # pylint: disable=no-member

        norm = nn.utils.clip_grad_norm_(controller.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()

        loss = loss.clone().detach()
        q_center = q_center.clone().detach()

        ckpt = {
            'state_dict': controller.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpt, ckpt_folder / f'{epoch + 1}.pth')

        writer.add_scalar('episode_reward', -loss.item(), epoch * num_frames)
        writer.add_scalar('loss/total', loss.item(), epoch)
        writer.add_scalar('loss/forward', forward_loss.item(), epoch)
        writer.add_scalar('loss/sideward', sideward_loss.item(), epoch)
        writer.add_scalar('loss/face', face_loss.item(), epoch)
        writer.add_scalar('q_center/x', q_center[0].item(), epoch)
        writer.add_scalar('q_center/y', q_center[1].item(), epoch)
        writer.add_scalar('q_center/z', q_center[2].item(), epoch)
        log.append([epoch * num_frames, -loss.item()])

        if (epoch + 1) % 10 == 0:
            display.save(str(video_folder / f'{epoch + 1}.mp4'))

        print(f'{epoch + 1}/{num_epochs} loss: {loss.item():.6e} center: {q_center.numpy()} norm: {norm:.3f}') # pylint: disable=no-member

    with torch.no_grad():
        forward_loss = 0
        sideward_loss = 0
        face_loss = 0

        a = None
        q, v = q0, v0
        for frame in range(1, num_frames + 1):
            state = get_state(q, v)
            a = controller(state, a)
            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)
            v_center = v.view(-1, 3)[center]
            q_center = q.view(-1, 3)[center].detach()
            face_dir = q.view(-1, 3)[face_head] - q.view(-1, 3)[face_tail]
            face_dir = face_dir / face_dir.norm()

            # forward loss
            dot = torch.dot(v_center, arrow_target_data)
            forward_loss += -dot

            # sideward loss
            cross = torch.cross(v_center, arrow_target_data)
            sideward_loss += torch.dot(cross, cross)

            # face loss
            face_loss += -torch.dot(face_dir, arrow_target_data)
        loss = forward_loss + w_sideward * sideward_loss + w_face * face_loss

    writer.add_scalar('episode_reward', -loss.item(), num_epochs * num_frames)
    writer.add_scalar('loss/total', loss.item(), num_epochs)
    writer.add_scalar('loss/forward', forward_loss.item(), num_epochs)
    writer.add_scalar('loss/sideward', sideward_loss.item(), num_epochs)
    writer.add_scalar('loss/face', face_loss.item(), num_epochs)
    writer.add_scalar('q_center/x', q_center[0].item(), num_epochs)
    writer.add_scalar('q_center/y', q_center[1].item(), num_epochs)
    writer.add_scalar('q_center/z', q_center[2].item(), num_epochs)
    log.append([epoch * num_frames, -loss.item()])
    writer.flush()

    torch.save(log, sub_folder / 'log.pth')


if __name__ == "__main__":
    main()
