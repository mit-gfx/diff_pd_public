import sys
sys.path.append('../')

import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info, ndarray
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.hex_mesh import hex2obj

def render_shark_3d(mesh_folder, img_name):
    # Read mesh.
    mesh = HexMesh3d()
    mesh.Initialize(str(mesh_folder / 'body.bin'))

    options = {
        'file_name': img_name,
        'resolution': (800, 600),
        'sample': 256,
        'max_depth': 3,
        'light_map': 'uffizi-large.exr',
        'camera_pos': (-0.07, -1.4, 0.5),
        'camera_lookat': (-0.07, 0, 0.15),
    }
    renderer = PbrtRenderer(options)
    renderer.add_hex_mesh(mesh, render_voxel_edge=True, color='e6ab45',
        transforms=[
            ('r', (np.pi, 0, 0, 1)),
            ('s', 0.4),
            ('t', (-0.2, -0.1, 0.2)),
        ])
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7', color='064273')
    renderer.render(verbose=True, nproc=None)

def render_shark_actuator(mesh_folder, img_name):
    options = {
        'file_name': img_name,
        'resolution': (800, 600),
        'sample': 256,
        'max_depth': 3,
        'light_map': 'uffizi-large.exr',
        'camera_pos': (-0.07, -1.4, 0.5),
        'camera_lookat': (-0.07, 0, 0.15),
    }
    renderer = PbrtRenderer(options)

    # Peek muscle numbers.
    muscle_num = 0
    while True:
        f = mesh_folder / 'muscle/{}.obj'.format(muscle_num)
        if not f.exists(): break
        muscle_num += 1
    assert muscle_num >= 0
    for i in range(muscle_num):
        f = mesh_folder / 'muscle/{}.obj'.format(i)
        renderer.add_hex_mesh(str(f), render_voxel_edge=True, texture_img='act.png',
            color=(1, 1, 1),
            transforms=[
                ('r', (np.pi, 0, 0, 1)),
                ('s', 0.4),
                ('t', (-0.2, -0.1, 0.2)),
            ])
    # Draw wireframe of the body.
    mesh = HexMesh3d()
    mesh.Initialize(str(mesh_folder / 'body.bin'))
    vertices, faces = hex2obj(mesh)
    for f in faces:
        for i in range(4):
            vi = vertices[f[i]]
            vj = vertices[f[(i + 1) % 4]]
            # Draw line vi to vj.
            renderer.add_shape_mesh({
                    'name': 'curve',
                    'point': ndarray([vi, (2 * vi + vj) / 3, (vi + 2 * vj) / 3, vj]),
                    'width': 0.002
                },
                color='e6ab45',
                transforms=[
                    ('r', (np.pi, 0, 0, 1)),
                    ('s', 0.4),
                    ('t', (-0.2, -0.1, 0.2)),
                ])

    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7', color='064273')
    renderer.render(verbose=True, nproc=None)

if __name__ == '__main__':
    # Download the mesh data from Dropbox and put them in a folder as follows:
    # - shark_3d
    #   - init
    #   - ppo
    #   - diffpd
    folder = Path('shark_3d')

    for mesh_folder in ['init', 'ppo', 'diffpd']:
        print_info('Processing {}...'.format(mesh_folder))
        render_folder = folder / '{}_render'.format(mesh_folder)
        create_folder(render_folder)
        render_act_folder = folder / '{}_render_act'.format(mesh_folder)
        create_folder(render_act_folder)

        # Peek the frame number.
        frame_num = 0
        while True:
            f = folder / mesh_folder / '{}'.format(frame_num)
            if not f.exists(): break
            frame_num += 1
        assert frame_num >= 0
        print_info('{} frames in total.'.format(frame_num))

        # Loop over all frames.
        for i in range(frame_num):
            render_shark_3d(folder / mesh_folder / '{}'.format(i), render_folder / '{:04d}.png'.format(i))
            render_shark_actuator(folder / mesh_folder / '{}'.format(i), render_act_folder / '{:04d}.png'.format(i))
