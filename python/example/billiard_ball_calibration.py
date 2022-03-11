import sys
sys.path.append('../')

from collections import deque
import os
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.backend_bases import MouseButton
import pickle
from re import split
from scipy.cluster.vq import vq, kmeans2

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import ndarray, create_folder, print_info, print_warning

def extract_intrinsic_parameters(K):
    K = ndarray(K).copy()
    cx = K[0, 2]
    cy = K[1, 2]
    alpha = K[0, 0]
    cot_theta = K[0, 1] / -alpha
    tan_theta = 1 / cot_theta
    theta = np.arctan(tan_theta)
    if theta < 0:
        theta += np.pi
    beta = K[1, 1] * np.sin(theta)
    return { 'alpha': alpha, 'beta': beta, 'theta': theta, 'cx': cx, 'cy': cy }

def assemble_intrinsic_parameters(alpha, beta, theta, cx, cy):
    K = np.zeros((3, 3))
    K[0, 0] = alpha
    K[0, 1] = -alpha / np.tan(theta)
    K[0, 2] = cx
    K[1, 1] = beta / np.sin(theta)
    K[1, 2] = cy
    K[2, 2] = 1
    return ndarray(K).copy()

def solve_camera(points_in_pixel, points_in_world):
    # This is a better reference: https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf
    #
    # The pixel space is:
    # - Origin: lower left.
    # - x: left to right.
    # - y: bottom to top.
    # Let p and P be points_in_pixel (2D) and points_in_world (3D), respectively.
    # Let R and t be the orientation and location of the world frame in the camera frame.
    # T = [R, t]
    #     [0, 1]
    # K = [alpha, -alpha * cot theta, cx, 0]
    #     [0, beta / sin theta, cy, 0]
    #     [0, 0, 1, 0]
    # Pixels: alpha * (x - cot theta * y) / z + cx
    #         beta / sin theta * y / z + cy
    # which makes sense if the image is skewed to its right.

    # [p, 1] = Homogenous(KT[P, 1]).
    # Let M = KT \in R^{3 x 4} = [m1, m2, m3]
    # p.x = <m1, [P, 1]> / <m3, [P, 1]>.
    # p.y = <m2, [P, 1]> / <m3, [P, 1]>.
    # p.x * <m3, [P, 1]> - <m1, [P, 1]> = 0.
    # p.y * <m3, [P, 1]> - <m2, [P, 1]> = 0.
    # Let's flatten them into a linear system.
    points_in_pixel = ndarray(points_in_pixel).copy()
    points_in_world = ndarray(points_in_world).copy()
    num_points = points_in_pixel.shape[0]
    assert (num_points, 2) == points_in_pixel.shape
    assert (num_points, 3) == points_in_world.shape
    P = ndarray(np.zeros((2 * num_points, 12)))
    for i in range(num_points):
        # Assemble the x equation.
        # m1:
        P[2 * i, :3] = -points_in_world[i]
        P[2 * i, 3] = -1
        # m3:
        P[2 * i, 8:11] = points_in_world[i] * points_in_pixel[i, 0]
        P[2 * i, 11] = points_in_pixel[i, 0]
        # Assemble the y equation.
        # m2:
        P[2 * i + 1, 4:7] = -points_in_world[i]
        P[2 * i + 1, 7] = -1
        # m3:
        P[2 * i + 1, 8:11] = points_in_world[i] * points_in_pixel[i, 1]
        P[2 * i + 1, 11] = points_in_pixel[i, 1]
    # Now m can be obtained from P * m = 0.
    # We solve this by minimizing \|P * m\|^2 s.t. \|m\|^2 = 1.
    # Consider SVD of P: P = U * Sigma * V.T.
    U, Sigma, Vt = np.linalg.svd(P)
    # U @ np.diag(Sigma) @ Vt = P.
    # So, Vt * m = [0, 0, 0, ..., 1], or m = V * [0, 0, 0, ..., 1].
    m = Vt[-1]
    M = ndarray(np.reshape(m, (3, 4)))
    # Now we know M = 1 / rho * KT. Let's extract camera parameters.
    a1 = M[0, :3]
    a2 = M[1, :3]
    a3 = M[2, :3]
    rho = 1 / np.linalg.norm(a3)
    cx = rho * rho * (a1.dot(a3))
    cy = rho * rho * (a2.dot(a3))
    a1_cross_a3 = np.cross(a1, a3)
    a2_cross_a3 = np.cross(a2, a3)
    cos_theta = -a1_cross_a3.dot(a2_cross_a3) / (np.linalg.norm(a1_cross_a3) * np.linalg.norm(a2_cross_a3))
    theta = np.arccos(cos_theta)
    alpha = rho * rho * np.linalg.norm(a1_cross_a3) * np.sin(theta)
    beta = rho * rho * np.linalg.norm(a2_cross_a3) * np.sin(theta)
    K = ndarray([[alpha, -alpha / np.tan(theta), cx],
        [0, beta / np.sin(theta), cy],
        [0, 0, 1]])

    # Extrinsic camera info:
    r1 = a2_cross_a3 / np.linalg.norm(a2_cross_a3)
    # r3 has two possibilities. We need to figure out which one is better.
    r3_pos = rho * a3
    r2_pos = np.cross(r3_pos, r1)
    R_pos = np.vstack([r1, r2_pos, r3_pos])
    r3_neg = -rho * a3
    r2_neg = np.cross(r3_neg, r1)
    R_neg = np.vstack([r1, r2_neg, r3_neg])
    # Compare K @ R and rho M[:, :3].
    if np.linalg.norm(K @ R_pos - rho * M[:, :3]) < np.linalg.norm(K @ R_neg + rho * M[:, :3]):
        R = R_pos
    else:
        R = R_neg
        rho = -rho
    T = rho * np.linalg.inv(K) @ M[:, 3]
    info = {
        'K': ndarray(K).copy(),
        'R': ndarray(R).copy(),
        'T': ndarray(T).copy(),
        'alpha': alpha,
        'beta': beta,
        'theta': theta,
        'cx': cx,
        'cy': cy,
    }
    return info

def solve_simple_camera(points_in_pixel, points_in_world):
    # The pixel space is:
    # - Origin: lower left.
    # - x: left to right.
    # - y: bottom to top.
    # Let p and P be points_in_pixel (2D) and points_in_world (3D), respectively.
    # Let R and t be the orientation and location of the world frame in the camera frame.
    # T = [R, t]
    #     [0, 1]
    # K = [alpha, 0, img_width / 2, 0]
    #     [0, alpha, img_height / 2, 0]
    #     [0, 0, 1, 0]
    # Pixels: alpha * x / z + cx
    #         alpha * y / z + cy

    # [p, 1] = Homogenous(KT[P, 1]).
    # Let M = KT \in R^{3 x 4} = [m1, m2, m3]
    # p.x = <m1, [P, 1]> / <m3, [P, 1]>.
    # p.y = <m2, [P, 1]> / <m3, [P, 1]>.
    # p.x * <m3, [P, 1]> - <m1, [P, 1]> = 0.
    # p.y * <m3, [P, 1]> - <m2, [P, 1]> = 0.
    # Let's flatten them into a linear system.
    points_in_pixel = ndarray(points_in_pixel).copy()
    points_in_pixel[:, 0] -= img_width / 2
    points_in_pixel[:, 1] -= img_height / 2
    points_in_world = ndarray(points_in_world).copy()
    num_points = points_in_pixel.shape[0]
    assert (num_points, 2) == points_in_pixel.shape
    assert (num_points, 3) == points_in_world.shape
    P = ndarray(np.zeros((2 * num_points, 12)))
    for i in range(num_points):
        # Assemble the x equation.
        # m1:
        P[2 * i, :3] = -points_in_world[i]
        P[2 * i, 3] = -1
        # m3:
        P[2 * i, 8:11] = points_in_world[i] * points_in_pixel[i, 0]
        P[2 * i, 11] = points_in_pixel[i, 0]
        # Assemble the y equation.
        # m2:
        P[2 * i + 1, 4:7] = -points_in_world[i]
        P[2 * i + 1, 7] = -1
        # m3:
        P[2 * i + 1, 8:11] = points_in_world[i] * points_in_pixel[i, 1]
        P[2 * i + 1, 11] = points_in_pixel[i, 1]
    # Now m can be obtained from P * m = 0.
    # We solve this by minimizing \|P * m\|^2 s.t. \|m\|^2 = 1.
    # Consider SVD of P: P = U * Sigma * V.T.
    U, Sigma, Vt = np.linalg.svd(P)
    # U @ np.diag(Sigma) @ Vt = P.
    # So, Vt * m = [0, 0, 0, ..., 1], or m = V * [0, 0, 0, ..., 1].
    m = Vt[-1]
    M = ndarray(np.reshape(m, (3, 4)))
    # Now we know M = 1 / rho * KT. Let's extract camera parameters.
    # M = 1 / rho * [alpha, 0, 0] * [R, t]
    #               [0, alpha, 0]
    #               [0,     0, 1]
    a1 = M[0, :3]
    a2 = M[1, :3]
    a3 = M[2, :3]
    # |rho| * |a3| = 1.
    rho_pos = 1 / np.linalg.norm(a3)
    rho_neg = -rho_pos
    info = None
    error = np.inf
    for rho in (rho_pos, rho_neg):
        KR = rho * M[:, :3]
        alpha0 = np.linalg.norm(KR[0])
        alpha1 = np.linalg.norm(KR[1])
        assert np.isclose(alpha0, alpha1, rtol=0.1)
        alpha = (alpha0 + alpha1) / 2
        R_est = np.copy(KR)
        R_est[0] /= alpha
        R_est[1] /= alpha
        U, Sig, Vt = np.linalg.svd(R_est)
        assert np.allclose(U @ np.diag(Sig) @ Vt, R_est)
        assert np.allclose(Sig, [1, 1, 1], rtol=0.5)
        R = U @ Vt
        K = np.diag([alpha, alpha, 1])
        t = np.linalg.inv(K) @ M[:, 3] * rho
        e = np.linalg.norm(np.hstack([K @ R, (K @ t)[:, None]]) / rho - M)
        if e < error:
            info = {
                'K': ndarray([[alpha, 0, img_width / 2],
                    [0, alpha, img_height / 2],
                    [0, 0, 1]]),
                'R': ndarray(R).copy(),
                'T': ndarray(t).copy(),
                'alpha': alpha,
                'beta': alpha,
                'theta': np.pi / 2,
                'cx': img_width / 2,
                'cy': img_height / 2
            }
            error = e
    return info

# Input:
# - image_data: H x W x 3 ndarray.
# Output:
# - M x 2 pixel coordinates and M x 3 3D coordinates in the world space.
# The world frame is defined as follows:
# - origin: lower left corner of the table.
# - x: left to right.
# - y: bottom to top.
# - z: pointing up from the table surface.
points_in_pixel = []
points_in_world_space = []
last_img_x = None
last_img_y = None
def select_corners(image_data):
    global points_in_pixel
    global points_in_world_space
    global last_img_x
    global last_img_y
    points_in_pixel = []
    points_in_world_space = []
    last_img_x = -1
    last_img_y = -1

    fig = plt.figure()
    ax_img = fig.add_subplot(211)
    ax_img.imshow(image_data)
    # The flat sheet.
    ax_table = fig.add_subplot(212)
    ax_table.set_xlabel('x')
    ax_table.set_ylabel('y')
    # We know the 3D coordinates of the table and the billiard box.
    table_corners = ndarray([
        [0, 0, 0],
        [1.10, 0, 0],
        [1.10, 0.67, 0],
        [0, 0.67, 0]
    ])
    billiard_box_top_corners = ndarray([
        [0, 0.67 - 0.056, 0.245],
        [0.245, 0.67 - 0.056, 0.245],
        [0.245, 0.67, 0.245],
        [0, 0.67, 0.245]
    ])
    billiard_box_bottom_corners = ndarray([
        [0, 0.67 - 0.056, 0],
        [0.245, 0.67 - 0.056, 0],
        [0.245, 0.67, 0],
        [0, 0.67, 0]
    ])
    billiard_box_top_corners_proxy = np.copy(billiard_box_top_corners)
    billiard_box_bottom_corners_proxy = np.copy(billiard_box_bottom_corners)
    # We shifted billiard box corners so that they don't overlap.
    billiard_box_top_corners_proxy[:, 0] += 0.1
    billiard_box_top_corners_proxy[:, 1] += 0.3
    billiard_box_bottom_corners_proxy[:, 0] += 0.03
    billiard_box_bottom_corners_proxy[:, 1] += 0.04
    # Plot the table and the corners.
    table_corners_aug = np.vstack([table_corners, table_corners[0]])
    ax_table.plot(table_corners_aug[:, 0], table_corners_aug[:, 1], 'k')
    billiard_box_top_corners_proxy_aug = np.vstack([billiard_box_top_corners_proxy, billiard_box_top_corners_proxy[0]])
    ax_table.plot(billiard_box_top_corners_proxy_aug[:, 0], billiard_box_top_corners_proxy_aug[:, 1], 'tab:blue')
    billiard_box_bottom_corners_proxy_aug = np.vstack([billiard_box_bottom_corners_proxy, billiard_box_bottom_corners_proxy[0]])
    ax_table.plot(billiard_box_bottom_corners_proxy_aug[:, 0], billiard_box_bottom_corners_proxy_aug[:, 1], 'tab:blue')
    ax_table.set_aspect('equal')

    def on_click(event):
        global points_in_pixel
        global points_in_world_space
        global last_img_x
        global last_img_y
        if event.button == MouseButton.LEFT:
            return
        if event.inaxes == ax_img:
            last_img_x, last_img_y = event.xdata, event.ydata
            # Plot the selected corner.
            ax_img.plot(last_img_x, last_img_y, 'y+')
        elif event.inaxes == ax_table:
            ix, iy = event.xdata, event.ydata
            xy = ndarray([ix, iy])

            all_corners = np.vstack([table_corners, billiard_box_top_corners, billiard_box_bottom_corners])
            all_proxy = np.vstack([table_corners, billiard_box_top_corners_proxy, billiard_box_bottom_corners_proxy])
            selected_id = np.argmin(np.sum((all_proxy[:, :2] - xy) ** 2, axis=1))
            # Plot the selected corner.
            ax_table.plot(all_proxy[selected_id, 0], all_proxy[selected_id, 1], 'y+')
            points_in_world_space.append(all_corners[selected_id])
            points_in_pixel.append([last_img_x, last_img_y])

        if len(points_in_world_space) == 8:
            fig.canvas.mpl_disconnect(cid)
        plt.gcf().canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return ndarray(points_in_pixel).copy(), ndarray(points_in_world_space).copy()

def convert_video_to_images(video_name, folder):
    os.system('ffmpeg -i {} -f image2 "{}/%04d.png"'.format(str(video_name), str(folder)))
    # Get the frame rate.
    os.system('ffprobe -v quiet -show_streams -select_streams v:0 {} | grep "r_frame_rate" > {}/frame_rate.txt'.format(video_name, folder))
    with open(folder / 'frame_rate.txt', 'r') as f:
        line = f.readline()
    os.remove(folder / 'frame_rate.txt')
    _, fps_str = line.strip().split('=')
    a, b = fps_str.split('/')
    fps = float(a) / float(b)
    dt = 1 / fps
    print_info('Loading video {}...'.format(video_name))
    print('fps: {:2.2f}'.format(fps))
    print('dt: {:2.4f}s'.format(dt))

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

def load_image(image_file):
    with cbook.get_sample_data(image_file) as f:
        img = plt.imread(f)
    return ndarray(img)

if __name__ == '__main__':
    # This is a script for calibrating the intrinsic camera parameters as well as the states of cube.
    np.random.seed(42)

    folder = Path(root_path) / 'python/example/billiard_ball_calibration'
    create_folder(folder, exist_ok=True)

    # Step 1: extract video information.
    experiment_video_name = Path(root_path) / 'asset/video/tennis_09.mov'
    experiment_video_data_folder = folder / 'experiment_video'
    create_folder(experiment_video_data_folder, exist_ok=True)
    if not (experiment_video_data_folder / '0001.png').is_file():
        convert_video_to_images(experiment_video_name, experiment_video_data_folder)

    # Step 2.1: calibrate the camera parameters.
    experiment_folder = folder / 'experiment'
    create_folder(experiment_folder, exist_ok=True)
    # Display the first frame.
    img = load_image(experiment_video_data_folder / '0001.png')
    # Dimension of img: (height, width, channel).
    # img_height, img_width, num_channels = img.shape
    assert img.shape[0] == img_height and img.shape[1] == img_width and img.shape[2] == 3
    # Call the label program.
    f = experiment_folder / 'intrinsic.data'
    if not f.is_file():
        print('Labeling intrinsic image 0001.png.')
        samples = select_corners(img)
        # Calibrate the camera system.
        pixels, coordinates = samples
        info = {}
        info['pts_pixel'] = ndarray(pixels).copy()
        info['pts_world'] = ndarray(coordinates).copy()
        # Save data.
        pickle.dump(info, open(f, 'wb'))
    else:
        info_loaded = pickle.load(open(f, 'rb'))
        pixels = info_loaded['pts_pixel']
        coordinates = info_loaded['pts_world']
        info = {}
        info['pts_pixel'] = ndarray(pixels).copy()
        info['pts_world'] = ndarray(coordinates).copy()

    # The pixel space in matplotlib is different from the pixel space in the calibration algorithm.
    camera_info = solve_simple_camera(pxl_to_cal(pixels), coordinates)
    K = camera_info['K']
    R = camera_info['R']
    T = camera_info['T']
    alpha = camera_info['alpha']
    beta = camera_info['beta']
    cx = camera_info['cx']
    cy = camera_info['cy']
    print('Camera information: alpha: {:2.2f}, beta: {:2.2f}, theta: {:2.2f}, cx: {:4.1f}, cy: {:4.1f}'.format(
        camera_info['alpha'], camera_info['beta'], np.rad2deg(camera_info['theta']), camera_info['cx'], camera_info['cy']
    ))
    # Now R and t are the orientation and location of the world frame in the camera space.
    # Verification:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    # Now plot the predicted object location.
    points_predicted_camera = (coordinates @ R.T + T) @ K.T
    points_predicted_calib = points_predicted_camera[:, :2] / points_predicted_camera[:, 2][:, None]
    points_predicted_pixl = cal_to_pxl(points_predicted_calib)
    ax.plot(points_predicted_pixl[:, 0], points_predicted_pixl[:, 1], 'y+')
    plt.show()
    fig.savefig(experiment_folder / '0001.png')
    plt.close('all')
    for k, v in camera_info.items():
        info[k] = v
    # Save data.
    pickle.dump(info, open(f, 'wb'))
    print('Data saved to', f)

    # Step 2.2: filter out the billiard ball positions.
    # The following start and end frames are chosen manually by looking at each frame.
    start_frame = 150
    end_frame = 200
    x_range = [350, 900]
    y_range = [100, 470]
    ball_rgb = ndarray([150, 150, 20]) / 255
    num_balls = 2
    num_ball_colors = np.random.rand(num_balls, 3)
    for idx in range(start_frame, end_frame):
        if (experiment_folder / '{:04d}_centroid.data'.format(idx)).is_file(): continue
        # Load images.
        img = load_image(experiment_video_data_folder / '{:04d}.png'.format(idx))

        # Extract the billiard balls.
        img_flag = np.full((img_height, img_width), False)
        img_flag[y_range[0]:y_range[1], x_range[0]:x_range[1]] = True
        img_flag = np.logical_and(img_flag, np.sum((img - ball_rgb) ** 2, axis=2) < 0.02)

        # Grow the billiard balls.
        q = deque([(i, j) for i in range(img_height) for j in range(img_width) if img_flag[i, j]])
        while q:
            i, j = q.popleft()
            # Check its four neighbors.
            for i2, j2 in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if not (y_range[0] <= i2 < y_range[1] and x_range[0] <= j2 < x_range[1]): continue
                if img_flag[i2, j2]: continue
                c = img[i2, j2]
                c_mag = np.linalg.norm(c)
                # Filter out dark area.
                if c_mag < 0.5: continue
                # Skip if the color is too different from the parent.
                if np.linalg.norm(c - img[i, j]) > 0.1: continue
                cos_angle = ndarray(c).dot(ball_rgb) / c_mag / np.linalg.norm(ball_rgb)
                if cos_angle > 0.92:
                    img_flag[i2, j2] = True
                    q.append((i2, j2))

        # Use k-means clustering to figure out the ball location (we only need the center.)
        pixels = ndarray([(j, i) for i in range(img_height) for j in range(img_width) if img_flag[i, j]])
        centroid, label = kmeans2(pixels, num_balls, minit='points')
        assert centroid.shape == (num_balls, 2)
        assert label.shape == (pixels.shape[0],)
        # Estimate the radius of each ball.
        img_flag = np.full((img_height, img_width), False)
        pixel_jj, pixel_ii = np.meshgrid(np.arange(img_width), np.arange(img_height))
        for i in range(num_balls):
            ball_i = ndarray(pixels[label == i]).copy()
            r = np.mean(np.max(ball_i, axis=0) - np.min(ball_i, axis=0)) / 2
            img_flag = np.logical_or(img_flag, (pixel_jj - centroid[i][0]) ** 2 + (pixel_ii - centroid[i][1]) ** 2 < r ** 2)

        # Remap centroids so that it is consistent with the last frame.
        if idx > start_frame:
            last_centroid = pickle.load(open(experiment_folder / '{:04d}_centroid.data'.format(idx - 1), 'rb'))
            new_idx = []
            new_centroid = np.zeros(centroid.shape)
            for c in centroid:
                new_j = np.argmin(np.sum((last_centroid - c) ** 2, axis=1))
                # c should be placed at centroid[new_j]
                new_centroid[new_j] = c
            centroid = new_centroid
        # Write filtered images.
        img_filtered = np.copy(img) * ndarray(img_flag)[:, :, None]
        plt.imsave(experiment_folder / '{:04d}_filtered.png'.format(idx), img_filtered)
        for c, cl in zip(centroid, num_ball_colors):
            ci, cj = int(c[0]), int(c[1])
            img[cj - 3 : cj + 4, ci - 3 : ci + 4] = cl
        # Write filtered images.
        img_filtered = np.copy(img) * ndarray(img_flag)[:, :, None]
        plt.imsave(experiment_folder / '{:04d}_filtered_with_center.png'.format(idx), img_filtered)

        # Save data.
        pickle.dump(centroid, open(experiment_folder / '{:04d}_centroid.data'.format(idx), 'wb'))

    # Step 2.3: reconstruct the 3D motion of the two billiard balls.
    ball_xy_positions = []
    ball_radius = 0.06858 / 2   # In meters and from measurement/googling the diameter of a tennis ball.
    for i in range(start_frame, end_frame):
        centroid = pickle.load(open(experiment_folder / '{:04d}_centroid.data'.format(i), 'rb'))
        positions = []
        assert len(centroid) == num_balls
        centroid = pxl_to_cal(centroid)
        for c in centroid:
            # K @ (R @ loc + T) = rho * [centroid, 1]
            # Now that we know loc[2] = ball_radius, we have 3 equations and 3 unknowns.
            # KR[:, :2] @ loc[:2] + KR[:, 2] @ loc[2] - [centroid, 1] * rho = -KT.
            # The sign of rho does not matter...
            # KR[:, :2] @ loc[:2] + [centroid, 1] * rho = -KT - KR[:, 2] @ loc[2].
            A = np.zeros((3, 3))
            A[:, :2] = K @ R[:, :2]
            A[:2, 2] = c
            A[2, 2] = 1
            b = -K @ T - K @ R[:, 2] * ball_radius
            x = np.linalg.inv(A) @ b
            pos = ndarray([x[0], x[1], ball_radius])
            # Sanity check.
            predicted = K @ (R @ pos + T)
            predicted = predicted[:2] / predicted[2]
            assert np.allclose(predicted, c)
            positions.append(pos)
        ball_xy_positions.append(('{:04d}'.format(i), ndarray(positions).copy()))
    # Plot them in 2D.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_balls):
        p = [pos[i] for _, pos in ball_xy_positions]
        p = ndarray(p)
        ax.plot(p[:, 0], p[:, 1], label='ball_{:d}'.format(i + 1), color=num_ball_colors[i])
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 0.67])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    # Save the position data.
    pickle.dump(ball_xy_positions, open(experiment_folder / 'ball_xy_positions.data', 'wb'))