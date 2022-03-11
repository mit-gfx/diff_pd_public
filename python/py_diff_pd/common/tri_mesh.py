import os
import struct
import numpy as np
from py_diff_pd.core.py_diff_pd_core import TriMesh2d
from py_diff_pd.common.common import ndarray

# use this function to generate *2D* triangle meshes.
# vertices: n x 2 numpy array.
# faces: m x 3 numpy array.
def generate_tri_mesh(vertices, faces, file_name, compute_normal=False):
    if str(file_name).endswith('.bin'):
        with open(file_name, 'wb') as f:
            f.write(struct.pack('i', 2))
            f.write(struct.pack('i', 3))
            # Vertices.
            vert_num, _ = ndarray(vertices).shape
            f.write(struct.pack('i', 2))
            f.write(struct.pack('i', vert_num))
            for v in vertices:
                f.write(struct.pack('d', v[0]))
            for v in vertices:
                f.write(struct.pack('d', v[1]))

            # Faces.
            faces = ndarray(faces).astype(np.int)
            face_num, _ = faces.shape
            f.write(struct.pack('i', 3))
            f.write(struct.pack('i', face_num))
            for j in range(3):
                for i in range(face_num):
                    f.write(struct.pack('i', faces[i, j]))
    elif str(file_name).endswith('.obj'):
        with open(file_name, 'w') as f_obj:
            vn = []
            if compute_normal:
                vert_num = len(vertices)
                vn = np.zeros((vert_num, 3))
                for ff in faces:
                    v0, v1, v2 = vertices[ff[0]], vertices[ff[1]], vertices[ff[2]]
                    weighted_norm = np.cross(v1 - v0, v2 - v1)
                    for i in range(3):
                        vn[ff[i]] += weighted_norm
                # Normalization.
                vn_len = np.sqrt(np.sum(vn ** 2, axis=1)) + 1e-6
                vn /= vn_len[:, None]
            for i, vv in enumerate(vertices):
                if compute_normal:
                    f_obj.write('vn {} {} {}\n'.format(vn[i][0], vn[i][1], vn[i][2]))
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            for ff in faces:
                if compute_normal:
                    f_obj.write('f {}//{} {}//{} {}//{}\n'.format(ff[0] + 1, ff[0] + 1,
                        ff[1] + 1, ff[1] + 1,
                        ff[2] + 1, ff[2] + 1))
                else:
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
    else:
        raise NotImplementedError

# Given tri_mesh, return vert and faces:
# - vertices: an n x 2 double array.
# - faces: an m x 3 integer array.
def tri2obj(tri_mesh, obj_file_name=None):
    vertex_num = tri_mesh.NumOfVertices()
    element_num = tri_mesh.NumOfElements()

    v = []
    for i in range(vertex_num):
        v.append(tri_mesh.py_vertex(i))
    v = ndarray(v)

    f = []
    for i in range(element_num):
        f.append(ndarray(tri_mesh.py_element(i)))
    f = ndarray(f).astype(np.int)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            for ff in f:
                f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))

    return v, f

def generate_circle_mesh(origin, radius, radius_bin_num, angle_bin_num, bin_file_name):
    origin = ndarray(origin).ravel()
    assert origin.size == 2
    radius = float(radius)
    assert radius > 0
    radius_bin_num = int(radius_bin_num)
    assert radius_bin_num > 0
    angle_bin_num = int(angle_bin_num)
    assert angle_bin_num >= 3

    # Assemble verts.
    vert = [origin,]
    dr = radius / radius_bin_num
    da = np.pi * 2 / angle_bin_num
    for i in range(radius_bin_num):
        for j in range(angle_bin_num):
            c, s = np.cos(j * da), np.sin(j * da)
            v = origin + ndarray([c, s]) * (i + 1) * dr
            vert.append(v)
    vert = ndarray(vert)

    # Assemble faces.
    face = []
    for i in range(angle_bin_num):
        face.append([0, i + 1, i + 2])
    face[-1][-1] = 1
    for i in range(radius_bin_num - 1):
        for j in range(angle_bin_num):
            i00 = i * angle_bin_num + 1 + j
            i01 = i00 + 1 if j < angle_bin_num - 1 else i00 + 1 - angle_bin_num
            i10 = i00 + angle_bin_num
            i11 = i01 + angle_bin_num
            face.append([i00, i10, i11])
            face.append([i00, i11, i01])

    face = ndarray(face).astype(np.int)

    generate_tri_mesh(vert, face, bin_file_name)

    return vert, face

# Extract boundary edges from a 2D mesh.
def get_boundary_edge(mesh):
    edges = set()
    element_num = mesh.NumOfElements()
    for e in range(element_num):
        vid = list(mesh.py_element(e))
        vij = [(vid[0], vid[1]), (vid[1], vid[2]), (vid[2], vid[0])]
        for vi, vj in vij:
            assert (vi, vj) not in edges
            if (vj, vi) in edges:
                edges.remove((vj, vi))
            else:
                edges.add((vi, vj))
    return list(edges)