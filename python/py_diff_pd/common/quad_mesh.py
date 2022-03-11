import os
import struct
import numpy as np
from py_diff_pd.core.py_diff_pd_core import QuadMesh2d
from py_diff_pd.common.common import ndarray

def generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name):
    nx, ny = cell_nums
    with open(bin_file_name, 'wb') as f:
        vertex_num = (nx + 1) * (ny + 1)
        element_num = nx * ny
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', 4))
        # Vertices.
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', vertex_num))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vx = origin[0] + i * dx
                f.write(struct.pack('d', vx))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vy = origin[1] + j * dx
                f.write(struct.pack('d', vy))

        voxel_indices = np.full((nx, ny), -1, dtype=np.int)
        index = 0
        for i in range(nx):
            for j in range(ny):
                voxel_indices[i, j] = index
                index += 1
        vertex_indices = np.full((nx + 1, ny + 1), -1, dtype=np.int)
        index = 0
        for i in range(nx + 1):
            for j in range(ny + 1):
                vertex_indices[i, j] = index
                index += 1

        # Faces.
        f.write(struct.pack('i', 4))
        f.write(struct.pack('i', element_num))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j + 1))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j + 1))

    return voxel_indices, vertex_indices

# Extract boundary edges from a 2D mesh.
def get_boundary_edge(mesh):
    edges = set()
    element_num = mesh.NumOfElements()
    for e in range(element_num):
        vid = list(mesh.py_element(e))
        # Quad mesh.
        vij = [(vid[0], vid[2]), (vid[2], vid[3]), (vid[3], vid[1]), (vid[1], vid[0])]
        for vi, vj in vij:
            assert (vi, vj) not in edges
            if (vj, vi) in edges:
                edges.remove((vj, vi))
            else:
                edges.add((vi, vj))
    return list(edges)