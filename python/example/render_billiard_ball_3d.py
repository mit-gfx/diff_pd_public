import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.cluster.vq import vq, kmeans2

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize
from py_diff_pd.core.py_diff_pd_core import TetMesh3d
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.env.billiard_ball_env_3d import BilliardBallEnv3d
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

def load_image(image_file):
    with cbook.get_sample_data(image_file) as f:
        img = plt.imread(f)
    return ndarray(img)

img_height, img_width = 720, 1280
def pxl_to_cal(pxl):
    pxl = ndarray(pxl).copy()
    pxl[:, 1] *= -1
    pxl[:, 1] += img_height
    return pxl
def cal_to_pxl(cal):
    cal = ndarray(cal).copy()
    cal[:, 1] -= img_height
    cal[:, 1] *= -1
    return cal

def q_to_obj(q, obj_file_name):
    q = ndarray(q).copy()
    # Generate the mesh from q.
    tmp_bin_file_name = '.tmp.bin'
    sphere_file_name = Path(root_path) / 'asset' / 'mesh' / 'sphere.obj'
    _, eles = tetrahedralize(sphere_file_name, normalize_input=False)
    all_verts = q.reshape((2, -1, 3))
    # Shift the z axis by radius.
    all_verts[:, :, 2] += ball_radius
    num_balls = 2
    all_eles = [eles + i * all_verts.shape[1] for i in range(num_balls)]
    all_verts = np.vstack(all_verts)
    all_eles = np.vstack(all_eles)
    generate_tet_mesh(all_verts, all_eles, tmp_bin_file_name)
    mesh = TetMesh3d()
    mesh.Initialize(str(tmp_bin_file_name))
    tet2obj(mesh, obj_file_name=obj_file_name)
    os.remove(tmp_bin_file_name)

def get_camera_info(R, T, alpha):
    # Compute camera_pos, camera_lookat, camera_up, and fov.
    # R.T indicate the camera coordinates.
    camera_pos = -R.T @ T
    camera_x = R[0]
    camera_y = R[1]
    camera_z = R[2]
    # Do we want to flip the directions?
    if camera_x[0] < 0:
        camera_x = -camera_x
        camera_y = -camera_y
    # Now x is left to right and y is bottom to up.
    camera_up = camera_y
    camera_lookat = camera_pos - camera_z
    # Compute fov from alpha.
    # np.tan(half_fov) * alpha = cy
    fov = np.rad2deg(np.arctan(img_height / 2 / alpha) * 2)
    return camera_pos, camera_lookat, camera_up, fov

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('render_billiard_ball_3d')

    # Simulation parameters.
    num_balls = 2
    substeps = 3
    dt = (1 / 60) / substeps
    start_frame = 150
    end_frame = 200

    # Extract the initial information of the balls.
    ball_radius = 0.06858 / 2   # In meters and from measurement/googling the diameter of a tennis ball.
    experiment_data_folder = Path(root_path) / 'python/example/billiard_ball_calibration/experiment_video'
    optimization_data_folder = Path(root_path) / 'python/example/billiard_ball_3d'
    opt_data = pickle.load(open(optimization_data_folder / 'data_0008_threads.bin', 'rb'))

    # Generate original video sequence and overlay video sequence.
    create_folder(folder / 'video', exist_ok=True)
    for i in range(start_frame, end_frame):
        img_name = folder / 'video' / '{:04d}.png'.format(i - start_frame)
        if img_name.is_file(): continue
        img = load_image(Path(root_path) / 'python/example/billiard_ball_calibration/experiment_video/{:04d}.png'.format(i))
        plt.imsave(img_name, img)

    create_folder(folder / 'video_overlay', exist_ok=True)
    for i in range(end_frame - start_frame):
        img_name = folder / 'video_overlay' / '{:04d}.png'.format(i)
        if img_name.is_file(): continue
        img = load_image(Path(root_path) / 'python/example' / folder / 'video' / '{:04d}.png'.format(i))
        img = img[:, :, :3]
        for j in range(0, i, 10):
            img_j = load_image(Path(root_path) / 'python/example/billiard_ball_calibration'
                / 'experiment/{:04d}_filtered.png'.format(j + start_frame))
            img_j = img_j[:, :, :3]
            img += img_j * 0.3
        img = np.clip(img, 0, 1)
        plt.imsave(img_name, img)

    # Render initial and optimized results.
    for name in ('init', 'pd_eigen'):
        sim_data = pickle.load(open(optimization_data_folder / name / 'info.data', 'rb'))
        _, info = sim_data
        create_folder(folder / '{}_normal'.format(name), exist_ok=True)
        create_folder(folder / '{}_black'.format(name), exist_ok=True)

        # Extract the camera information.
        camera_data = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_calibration/experiment/intrinsic.data', 'rb'))
        R = camera_data['R']
        T = camera_data['T']
        alpha = camera_data['alpha']
        x_sol, _ = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_3d' / name / 'info.data', 'rb'))
        roll, pitch, yaw = x_sol[12:15]
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        R_yaw = ndarray([[c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]])
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        R_pitch = ndarray([[c_pitch, 0, s_pitch],
            [0, 1, 0],
            [-s_pitch, 0, c_pitch]])
        c_roll, s_roll = np.cos(roll), np.sin(roll)
        R_roll = ndarray([[1, 0, 0],
            [0, c_roll, -s_roll],
            [0, s_roll, c_roll]])
        R_final = R_yaw @ R_pitch @ R_roll @ R
        T_final = x_sol[15:18] + T
        alpha_final = x_sol[18] * alpha
        camera_pos, camera_lookat, camera_up, fov = get_camera_info(R_final, T_final, alpha_final)

        # Render frames.
        for i, qi in enumerate(info['q']):
            if i % substeps != 0: continue
            for ext in ('normal', 'black'):
                img_name = folder / '{}_{}'.format(name, ext) / '{:04d}.png'.format(int(i // substeps))
                if img_name.is_file(): continue

                options = {
                    'file_name': img_name,
                    'light_map': 'uffizi-large.exr',
                    'sample': 4,
                    'max_depth': 2,
                    'camera_pos': camera_pos,
                    'camera_lookat': camera_lookat,
                    'camera_up': camera_up,
                    'resolution': (img_width, img_height),
                    'fov': fov,
                }
                renderer = PbrtRenderer(options)
                obj_file_name = folder / '.tmp.obj'
                q_to_obj(qi, obj_file_name)
                renderer.add_tri_mesh(obj_file_name, color=ndarray([150 / 255, 30 / 255, 20 / 255]), render_tet_edge=False)

                # Rotate the ground so that it is horizontal.
                camera_side = np.cross(camera_lookat - camera_pos, camera_up)
                ground_angle = np.arctan2(camera_side[1], camera_side[0])
                renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/flat_ground.obj', texture_img='chkbd_24_0.7',
                    transforms=[('r', (ground_angle, 0, 0, 1)), ('t', (0.5, 0.5, 0))], color=[0, 0, 0] if ext == 'black' else [.5, .5, .5])
                renderer.render(light_rgb=(.5, .5, .5), verbose=True)
                os.remove(folder / '.tmp.obj')
        # Overlay.
        create_folder(folder / '{}_overlay'.format(name), exist_ok=True)
        for i in range(end_frame - start_frame):
            img_name = folder / '{}_overlay'.format(name) / '{:04d}.png'.format(i)
            if img_name.is_file(): continue
            img = load_image(Path(root_path) / 'python/example' / folder / '{}_normal'.format(name) / '{:04d}.png'.format(i))
            img = img[:, :, :3]
            for j in range(0, i, 10):
                img_j = load_image(Path(root_path) / 'python/example' / folder / '{}_black'.format(name) / '{:04d}.png'.format(j))
                img_j = img_j[:, :, :3]
                # Extract the ball center.
                alpha_map = np.sum(np.abs(img_j), axis=2) > 0
                pixels = ndarray([(j, i) for i in range(img_height) for j in range(img_width) if alpha_map[i, j]])
                centroid, label = kmeans2(pixels, num_balls, minit='points')
                assert centroid.shape == (num_balls, 2)
                assert label.shape == (pixels.shape[0],)
                # Estimate the radius of each ball.
                alpha_map = np.full((img_height, img_width), False)
                pixel_jj, pixel_ii = np.meshgrid(np.arange(img_width), np.arange(img_height))
                for i in range(num_balls):
                    ball_i = ndarray(pixels[label == i]).copy()
                    r = np.mean(np.max(ball_i, axis=0) - np.min(ball_i, axis=0)) / 2
                    # Scale down r a bit to get rid of the black ring.
                    r *= 0.95
                    alpha_map = np.logical_or(alpha_map, (pixel_jj - centroid[i][0]) ** 2 + (pixel_ii - centroid[i][1]) ** 2 < r ** 2)
                # Overlay images.
                alpha_map = alpha_map[:, :, None]
                img = ndarray(alpha_map) * img_j * 0.3 + ndarray(~alpha_map) * img + ndarray(alpha_map) * img * 0.7
            img = np.clip(img, 0, 1)
            plt.imsave(img_name, img)

    # Generate videos.
    video_folder = Path(root_path) / 'python/example' / folder / 'video'
    video_overlay_folder = Path(root_path) / 'python/example' / folder / 'video_overlay'
    init_folder = Path(root_path) / 'python/example' / folder / 'init_normal'
    init_overlay_folder = Path(root_path) / 'python/example' / folder / 'init_overlay'
    pd_eigen_folder = Path(root_path) / 'python/example' / folder / 'pd_eigen_normal'
    pd_eigen_overlay_folder = Path(root_path) / 'python/example' / folder / 'pd_eigen_overlay'
    create_folder(folder / 'video_init_normal', exist_ok=True)
    create_folder(folder / 'video_init_overlay', exist_ok=True)
    create_folder(folder / 'video_pd_eigen_normal', exist_ok=True)
    create_folder(folder / 'video_pd_eigen_overlay', exist_ok=True)
    create_folder(folder / 'video_init_overlay_circle', exist_ok=True)
    create_folder(folder / 'video_pd_eigen_overlay_circle', exist_ok=True)
    centroids_so_far = {}
    for i in range(end_frame - start_frame):
        video_img = load_image(video_folder / '{:04d}.png'.format(i))[:, :, :3]
        video_overlay_img = load_image(video_overlay_folder / '{:04d}.png'.format(i))[:, :, :3]
        init_img = load_image(init_folder / '{:04d}.png'.format(i))[:, :, :3]
        init_overlay_img = load_image(init_overlay_folder / '{:04d}.png'.format(i))[:, :, :3]
        pd_eigen_img = load_image(pd_eigen_folder / '{:04d}.png'.format(i))[:, :, :3]
        pd_eigen_overlay_img = load_image(pd_eigen_overlay_folder / '{:04d}.png'.format(i))[:, :, :3]
        video_init_img = np.concatenate([video_img, init_img], axis=1)
        if not (folder / 'video_init_normal' / '{:04d}.png'.format(i)).is_file():
            plt.imsave(folder / 'video_init_normal' / '{:04d}.png'.format(i), video_init_img)
        video_pd_eigen_img = np.concatenate([video_img, pd_eigen_img], axis=1)
        if not (folder / 'video_pd_eigen_normal' / '{:04d}.png'.format(i)).is_file():
            plt.imsave(folder / 'video_pd_eigen_normal' / '{:04d}.png'.format(i), video_pd_eigen_img)

        # Generate overlay image.
        init_overlay_img_original = np.copy(init_overlay_img)
        pd_eigen_overlay_img_original = np.copy(pd_eigen_overlay_img)
        for j in list(range(0, i, 10)) + [end_frame - start_frame - 1,]:
            if j > i: continue
            img_j = load_image(Path(root_path) / 'python/example/billiard_ball_calibration'
                / 'experiment/{:04d}_filtered.png'.format(j + start_frame))[:, :, :3]
            init_overlay_img += img_j * 0.3
            pd_eigen_overlay_img += img_j * 0.3
            centroid = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_calibration/experiment'
                / '{:04d}_centroid.data'.format(j + start_frame), 'rb'))
            centroids_so_far[j] = centroid

        # Generate overlay image with circles.
        if not (folder / 'video_init_overlay_circle' / '{:04d}.png'.format(i)).is_file():
            for _, centroid in centroids_so_far.items():
                for px, py in centroid:
                    px = int(px)
                    py = int(py)
                    init_overlay_img_original[py - 4:py + 5, px - 4:px + 5] = ndarray([1, 1, 0])
            init_overlay_img_original = np.clip(init_overlay_img_original, 0, 1)
            plt.imsave(folder / 'video_init_overlay_circle' / '{:04d}.png'.format(i), init_overlay_img_original)
        if not (folder / 'video_pd_eigen_overlay_circle' / '{:04d}.png'.format(i)).is_file():
            for _, centroid in centroids_so_far.items():
                for px, py in centroid:
                    px = int(px)
                    py = int(py)
                    pd_eigen_overlay_img_original[py - 4:py + 5, px - 4:px + 5] = ndarray([1, 1, 0])
            pd_eigen_overlay_img_original = np.clip(pd_eigen_overlay_img_original, 0, 1)
            plt.imsave(folder / 'video_pd_eigen_overlay_circle' / '{:04d}.png'.format(i), pd_eigen_overlay_img_original)
        init_overlay_img = np.clip(init_overlay_img, 0, 1)
        pd_eigen_overlay_img = np.clip(pd_eigen_overlay_img, 0, 1)
        video_init_overlay_img = np.concatenate([video_overlay_img, init_overlay_img], axis=1)
        if not (folder / 'video_init_overlay' / '{:04d}.png'.format(i)).is_file():
            plt.imsave(folder / 'video_init_overlay' / '{:04d}.png'.format(i), video_init_overlay_img)
        video_pd_eigen_overlay_img = np.concatenate([video_overlay_img, pd_eigen_overlay_img], axis=1)
        if not (folder / 'video_pd_eigen_overlay' / '{:04d}.png'.format(i)).is_file():
            plt.imsave(folder / 'video_pd_eigen_overlay' / '{:04d}.png'.format(i), video_pd_eigen_overlay_img)
    export_mp4(folder / 'video_init_normal', folder / 'video_init_normal.mp4', fps=30)
    export_mp4(folder / 'video_init_overlay', folder / 'video_init_overlay.mp4', fps=15)
    export_mp4(folder / 'video_pd_eigen_normal', folder / 'video_pd_eigen_normal.mp4', fps=30)
    export_mp4(folder / 'video_pd_eigen_overlay', folder / 'video_pd_eigen_overlay.mp4', fps=15)