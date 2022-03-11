import os
import struct
import numpy as np
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.common.common import ndarray

# voxels: an 0-1 array of size cell_x_num * cell_y_num * cell_z_num.
def generate_hex_mesh(voxels, dx, origin, bin_file_name, write=True):
    origin = np.asarray(origin, dtype=np.float64)
    cell_x, cell_y, cell_z = voxels.shape
    node_x, node_y, node_z = cell_x + 1, cell_y + 1, cell_z + 1
    vertex_flag = np.full((node_x, node_y, node_z), -1, dtype=np.int)
    for i in range(cell_x):
        for j in range(cell_y):
            for k in range(cell_z):
                if voxels[i][j][k]:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                vertex_flag[i + ii][j + jj][k + kk] = 0

    vertex_cnt = 0
    vertices = []
    for i in range(node_x):
        for j in range(node_y):
            for k in range(node_z):
                if vertex_flag[i][j][k] == 0:
                    vertex_flag[i][j][k] = vertex_cnt
                    vertices.append((origin[0] + dx * i,
                        origin[1] + dx * j,
                        origin[2] + dx * k))
                    vertex_cnt += 1

    voxel_indices = np.full((cell_x, cell_y, cell_z), -1, dtype=np.int)
    index = 0

    faces = []
    for i in range(cell_x):
        for j in range(cell_y):
            for k in range(cell_z):
                if voxels[i][j][k]:
                    face = []
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                face.append(vertex_flag[i + ii][j + jj][k + kk])
                    faces.append(face)
                    voxel_indices[i, j, k] = index
                    index += 1

    vertices = np.asarray(vertices, dtype=np.float64).T
    faces = np.asarray(faces, dtype=np.int).T

    if write:
        with open(bin_file_name, 'wb') as f:
            f.write(struct.pack('i', 3))
            f.write(struct.pack('i', 8))
            # Vertices.
            f.write(struct.pack('i', 3))
            f.write(struct.pack('i', vertices.shape[1]))
            f.write(struct.pack('d' * vertices.size, *list(vertices.ravel())))

            # Faces.
            f.write(struct.pack('i', 8))
            f.write(struct.pack('i', faces.shape[1]))
            f.write(struct.pack('i' * faces.size, *list(faces.ravel())))

    return voxel_indices, vertex_flag

def hex2obj_with_texture_coords(hex_mesh, hex_mesh_texture_coords, pbrt_file_name,
    compute_normal):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()
    hex_mesh_texture_coords = ndarray(hex_mesh_texture_coords)

    face_dict = {}
    face_idx = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (1, 5, 7, 3),
        (0, 2, 6, 4)
    ]
    for i in range(element_num):
        fi = ndarray(hex_mesh.py_element(i))
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)
    # Now f is a list of quads.

    v_out = []
    f_out = []
    texture_out = []
    v_cnt = 0
    for fi in f:
        fi_out = [v_cnt, v_cnt + 1, v_cnt + 2, v_cnt + 3]
        f_out.append(fi_out)
        v_cnt += 4
        for vi in fi:
            v_out.append(ndarray(hex_mesh.py_vertex(int(vi))))
            texture_out.append(hex_mesh_texture_coords[int(vi)])

    # Normals.
    vn = []
    if compute_normal:
        vert_num = len(v_out)
        vn = np.zeros((vert_num, 3))
        for ff in f_out:
            v0, v1, v2 = v_out[ff[0]], v_out[ff[1]], v_out[ff[2]]
            weighted_norm = np.cross(v1 - v0, v2 - v1)
            for i in range(3):
                vn[ff[i]] += weighted_norm
        # Normalization.
        vn_len = np.sqrt(np.sum(vn ** 2, axis=1)) + 1e-6
        vn /= vn_len[:, None]

    with open(pbrt_file_name, 'w') as f_pbrt:
        f_pbrt.write('AttributeBegin\n')
        f_pbrt.write('Shape "trianglemesh"\n')

        # Log point data.
        f_pbrt.write('  "point3 P" [\n')
        for vv in v_out:
            f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
        f_pbrt.write(']\n')

        # Log texture data.
        f_pbrt.write('  "float uv" [\n')
        for u, v in texture_out:
            f_pbrt.write('  {:f} {:f}\n'.format(u, v))
        f_pbrt.write(']\n')

        # Log face data.
        f_pbrt.write('  "integer indices" [\n')
        for ff in f_out:
            f_pbrt.write('  {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(ff[0], ff[1], ff[2], ff[0], ff[2], ff[3]))
        f_pbrt.write(']\n')

        # Log normal data.
        if compute_normal:
            f_pbrt.write('  "normal N" [\n')
            for vv in vn:
                f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            f_pbrt.write(']\n')

        f_pbrt.write('AttributeEnd\n')

# Given a hex mesh, save it as an obj file with texture coordinates.
def hex2obj_with_textures(hex_mesh, obj_file_name=None, pbrt_file_name=None,
    texture_map=None, compute_normal=False):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()

    face_dict = {}
    face_idx = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (1, 5, 7, 3),
        (0, 2, 6, 4)
    ]
    for i in range(element_num):
        fi = ndarray(hex_mesh.py_element(i))
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)
    # Now f is a list of quads.

    v_out = []
    f_out = []
    v_cnt = 0
    for fi in f:
        fi_out = [v_cnt, v_cnt + 1, v_cnt + 2, v_cnt + 3]
        f_out.append(fi_out)
        v_cnt += 4
        for vi in fi:
            v_out.append(ndarray(hex_mesh.py_vertex(int(vi))))

    if texture_map is None:
        texture_map = [[0, 0], [1, 0], [1, 1], [0, 1]]
    texture_map = ndarray(texture_map)
    assert texture_map.shape == (4, 2)

    # Normals.
    vn = []
    if compute_normal:
        vert_num = len(v_out)
        vn = np.zeros((vert_num, 3))
        for ff in f_out:
            v0, v1, v2 = v_out[ff[0]], v_out[ff[1]], v_out[ff[2]]
            weighted_norm = np.cross(v1 - v0, v2 - v1)
            for i in range(3):
                vn[ff[i]] += weighted_norm
        # Normalization.
        vn_len = np.sqrt(np.sum(vn ** 2, axis=1)) + 1e-6
        vn /= vn_len[:, None]

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for i, vv in enumerate(v_out):
                if compute_normal:
                    f_obj.write('vn {:6f} {:6f} {:6f}\n'.format(vn[i][0], vn[i][1], vn[i][2]))
                f_obj.write('v {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            for u, v in texture_map:
                f_obj.write('vt {:6f} {:6f}\n'.format(u, v))
            for ff in f_out:
                if compute_normal:
                    f_obj.write('f {:d}/1/{:d} {:d}/2/{:d} {:d}/3/{:d}\n'.format(
                        ff[0] + 1, ff[0] + 1,
                        ff[1] + 1, ff[1] + 1,
                        ff[2] + 1, ff[2] + 1))
                    f_obj.write('f {:d}/1/{:d} {:d}/3/{:d} {:d}/4/{:d}\n'.format(
                        ff[0] + 1, ff[0] + 1,
                        ff[2] + 1, ff[2] + 1,
                        ff[3] + 1, ff[3] + 1))
                else:
                    f_obj.write('f {:d}/1 {:d}/2 {:d}/3\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
                    f_obj.write('f {:d}/1 {:d}/3 {:d}/4\n'.format(ff[0] + 1, ff[2] + 1, ff[3] + 1))

    if pbrt_file_name is not None:
        with open(pbrt_file_name, 'w') as f_pbrt:
            f_pbrt.write('AttributeBegin\n')
            f_pbrt.write('Shape "trianglemesh"\n')

            # Log point data.
            f_pbrt.write('  "point3 P" [\n')
            for vv in v_out:
                f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            f_pbrt.write(']\n')

            # Log texture data.
            f_pbrt.write('  "float uv" [\n')
            for _ in range(int(len(v_out) / 4)):
                f_pbrt.write('  0 0\n')
                f_pbrt.write('  1 0\n')
                f_pbrt.write('  1 1\n')
                f_pbrt.write('  0 1\n')
            f_pbrt.write(']\n')

            # Log face data.
            f_pbrt.write('  "integer indices" [\n')
            for ff in f_out:
                f_pbrt.write('  {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(ff[0], ff[1], ff[2], ff[0], ff[2], ff[3]))
            f_pbrt.write(']\n')

            # Log normal data.
            if compute_normal:
                f_pbrt.write('  "normal N" [\n')
                for vv in vn:
                    f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
                f_pbrt.write(']\n')

            f_pbrt.write('AttributeEnd\n')

# Given a hex mesh, return the following:
# - vertices: an n x 3 double array.
# - faces: an m x 4 integer array.
def hex2obj(hex_mesh, obj_file_name=None, obj_type='quad'):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()

    v = []
    for i in range(vertex_num):
        v.append(hex_mesh.py_vertex(i))
    v = ndarray(v)

    face_dict = {}
    face_idx = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (1, 5, 7, 3),
        (0, 2, 6, 4)
    ]
    for i in range(element_num):
        fi = ndarray(hex_mesh.py_element(i))
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            if obj_type == 'quad':
                for ff in f:
                    f_obj.write('f {} {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1, ff[3] + 1))
            elif obj_type == 'tri':
                for ff in f:
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[2] + 1, ff[3] + 1))
            else:
                raise NotImplementedError

    return v, f

# Extract boundary faces from a 3D mesh.
def get_boundary_face(hex_mesh):
    faces = set()
    element_num = hex_mesh.NumOfElements()

    def hex_element_to_faces(vid):
        faces = [[vid[0], vid[1], vid[3], vid[2]],
            [vid[4], vid[6], vid[7], vid[5]],
            [vid[0], vid[4], vid[5], vid[1]],
            [vid[2], vid[3], vid[7], vid[6]],
            [vid[0], vid[2], vid[6], vid[4]],
            [vid[1], vid[5], vid[7], vid[3]],
        ]
        return faces

    def normalize_idx(l):
        min_idx = np.argmin(l)
        if min_idx == 0:
            l_ret = [l[0], l[1], l[2], l[3]]
        elif min_idx == 1:
            l_ret = [l[1], l[2], l[3], l[0]]
        elif min_idx == 2:
            l_ret = [l[2], l[3], l[0], l[1]]
        else:
            l_ret = [l[3], l[0], l[1], l[2]]
        return tuple(l_ret)

    for e in range(element_num):
        vid = list(hex_mesh.py_element(e))
        for l in hex_element_to_faces(vid):
            l = normalize_idx(l)
            assert l not in faces
            l_reversed = normalize_idx(list(reversed(l)))
            if l_reversed in faces:
                faces.remove(l_reversed)
            else:
                faces.add(l)
    return list(faces)

# Return a heuristic set of vertices that could be used for contact handling.
def get_contact_vertex(hex_mesh):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()
    v_maps = np.zeros((vertex_num, 8))
    for e in range(element_num):
        vertex_indices = list(hex_mesh.py_element(e))
        for i, vid in enumerate(vertex_indices):
            assert v_maps[vid][i] == 0
            v_maps[vid][i] = 1

    contact_vertices = []
    for i, v in enumerate(v_maps):
        # We consider the following vertices as contact vertices.
        if np.sum(v) == 1:
            contact_vertices.append(i)
        # We could add other cases, e.g., np.sum(v) == 2, if needed.
    return contact_vertices

# Input:
# - HexMesh
# - active_elements: a list of elements that you would like to keep.
# Output:
# - HexMesh 
def filter_hex(hex_mesh, active_elements):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()
    vertex_indices = np.zeros(vertex_num)
    for e_idx in active_elements:
        vertex_indices[list(hex_mesh.py_element(e_idx))] = 1
    remap = -np.ones(vertex_num)
    cnt = 0
    vertices = []
    for i in range(vertex_num):
        if vertex_indices[i] == 1:
            remap[i] = cnt
            cnt += 1
            vertices.append(ndarray(hex_mesh.py_vertex(i)))
    vertices = ndarray(vertices)
    faces = []
    for e_idx in active_elements:
        faces.append([remap[ei] for ei in list(hex_mesh.py_element(e_idx))])
    faces = ndarray(faces).astype(np.int)
    tmp_file_name = '.tmp.bin'

    vertices = vertices.T
    faces = faces.T
    with open(tmp_file_name, 'wb') as f:
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', 8))
        # Vertices.
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', vertices.shape[1]))
        f.write(struct.pack('d' * vertices.size, *list(vertices.ravel())))

        # Faces.
        f.write(struct.pack('i', 8))
        f.write(struct.pack('i', faces.shape[1]))
        f.write(struct.pack('i' * faces.size, *list(faces.ravel())))
    mesh = HexMesh3d()
    mesh.Initialize(tmp_file_name)
    os.remove(tmp_file_name)
    return mesh

# Convert triangle meshes into voxels.
# Input:
# - triangle mesh file name (obj, stl, ply, etc.)
# - dx: the size of the cell, which will be explained later.
# Output:
# - a 3D 0-1 array of size nx x ny x nz where nx, ny, and nz are the number of cells along x, y, and z axes.
#
# Algorithm:
# - Load the triangle mesh.
# - Rescale it so that the longest axis of the bounding box is 1.
# - Divide the whole bounding box into cells of size dx.
import trimesh

def voxelize(triangle_mesh_file_name, dx, feather=0.0):
    tri_mesh = trimesh.load(triangle_mesh_file_name)
    assert tri_mesh.is_watertight
    bbx_offset = np.min(tri_mesh.vertices, axis=0)
    tri_mesh.vertices -= bbx_offset
    bbx_extent = ndarray(tri_mesh.bounding_box.extents)
    tri_mesh.vertices /= np.max(bbx_extent)
    # Now tri_mesh.vertices is bounded by [0, 1].
    assert 0 < dx <= 0.5

    # Voxelization.
    cell_num = (ndarray(tri_mesh.bounding_box.extents) / dx).astype(np.int)
    voxels = np.zeros(cell_num)
    for i in range(cell_num[0]):
        for j in range(cell_num[1]):
            for k in range(cell_num[2]):
                center = ndarray([i + 0.5, j + 0.5, k + 0.5]) * dx
                signed_distance = trimesh.proximity.signed_distance(tri_mesh, center.reshape((1, 3)))
                if signed_distance > feather:
                    voxels[i][j][k] = 1
    return voxels
