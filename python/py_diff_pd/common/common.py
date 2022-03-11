import numpy as np

def ndarray(val):
    return np.asarray(val, dtype=np.float64)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

class PrettyTabular(object):
    def __init__(self, head):
        self.head = head

    def head_string(self):
        line = ''
        for key, value in self.head.items():
            if 's' in value:
                dummy = value.format('0')
            else:
                dummy = value.format(0)
            span = max(len(dummy), len(key)) + 2
            key_format = '{:^' + str(span) + '}'
            line += key_format.format(key)
        return line

    def row_string(self, row_data):
        line = ''
        for key, value in self.head.items():
            data = value.format(row_data[key])
            span = max(len(key), len(data)) + 2
            line += ' ' * (span - len(data) - 1) + data + ' '
        return line

import shutil
import os
def create_folder(folder_name, exist_ok=False):
    if not exist_ok and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=exist_ok)

def delete_folder(folder_name):
    shutil.rmtree(folder_name)

from py_diff_pd.core.py_diff_pd_core import StdRealVector
def to_std_real_vector(v):
    v = ndarray(v).ravel()
    n = v.size
    v_array = StdRealVector(n)
    for i in range(n):
        v_array[i] = v[i]
    return v_array

from py_diff_pd.core.py_diff_pd_core import StdIntVector
def copy_std_int_vector(v):
    v2 = StdIntVector(v.size())
    for i, a in enumerate(v):
        v2[i] = a
    return v2

def to_std_int_vector(v):
    v2 = StdIntVector(len(v))
    for i, a in enumerate(v):
        v2[i] = a
    return v2

from py_diff_pd.core.py_diff_pd_core import StdMap
def to_std_map(opt):
    opt_map = StdMap()
    for k, v in opt.items():
        opt_map[k] = float(v)
    return opt_map

# Rotation.
# Input (rpy): a 3D vector (roll, pitch, yaw).
# Output (R): a 3 x 3 rotation matrix.
def rpy_to_rotation(rpy):
    rpy = ndarray(rpy).ravel()
    assert rpy.size == 3
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = ndarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = ndarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = ndarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return R_yaw @ R_pitch @ R_roll

# Gradients of rotation.
# Input (rpy): a 3D vector (roll, pitch, yaw).
# Output (R): three 3 x 3 matrices.
def rpy_to_rotation_gradient(rpy):
    rpy = ndarray(rpy).ravel()
    assert rpy.size == 3
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = ndarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    dR_droll = ndarray([[0, 0, 0], [0, -sr, -cr], [0, cr, -sr]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = ndarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    dR_dpitch = ndarray([[-sp, 0, cp], [0, 0, 0], [-cp, 0, -sp]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = ndarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    dR_dyaw = ndarray([[-sy, -cy, 0], [cy, -sy, 0], [0, 0, 0]])

    return R_yaw @ R_pitch @ dR_droll, R_yaw @ dR_dpitch @ R_roll, dR_dyaw @ R_pitch @ R_roll

# Filter unreferenced vertices in a mesh.
# Input:
# - vertices: n x l.
# - elements: m x k.
# This function checks all rows in elements and generates a new (vertices, elements) pair such that all
# vertices are referenced.
def filter_unused_vertices(vertices, elements):
    vert_num = vertices.shape[0]
    elem_num = elements.shape[0]

    used = np.zeros(vert_num)
    for e in elements:
        for ei in e:
            used[ei] = 1

    remap = np.ones(vert_num) * -1
    used_so_far = 0
    for idx, val in enumerate(used):
        if val > 0:
            remap[idx] = used_so_far
            used_so_far += 1

    new_vertices = []
    for idx, val in enumerate(used):
        if val > 0:
            new_vertices.append(vertices[idx])
    new_vertices = ndarray(new_vertices)

    new_elements = []
    for e in elements:
        new_ei = [remap[ei] for ei in e]
        new_elements.append(new_ei)
    new_elements = ndarray(new_elements).astype(np.int)
    return new_vertices, new_elements

import struct
# Read binary matrix data from cpp.
def load_matrix(file_name):
    with open(file_name, 'rb') as f:
        content = f.read()
        rows = struct.unpack('i', content[:4])[0]
        cols = struct.unpack('i', content[4:8])[0]
        offset = 8
        A = ndarray(struct.unpack('d' * rows * cols, content[8:])).reshape((rows, cols))
        return A