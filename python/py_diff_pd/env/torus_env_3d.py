import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.project_path import root_path

class TorusEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 5e5
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([np.log10(2) + 5,])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'torus_analytic.bin'
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        torus_size = 0.1
        # Rescale the mesh.
        mesh.Scale(torus_size)
        tmp_bin_file_name = '.tmp.bin'
        mesh.SaveToFile(tmp_bin_file_name)

        deformable = HexDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)
        # Collisions.
        friction_node_idx = get_contact_vertex(mesh)
        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        plt.show()
        '''

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Initial states.
        dofs = deformable.dofs()
        print('Torus element: {:d}, DoFs: {:d}.'.format(mesh.NumOfElements(), dofs))
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Actuation.
        element_num = mesh.NumOfElements()
        self._actuator_stiffness_dofs = element_num
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        com = np.mean(q0.reshape((-1, 3)), axis=0)
        act_group_num = options['act_group_num'] if 'act_group_num' in options else 4
        act_groups = [[] for _ in range(act_group_num)]
        delta = np.pi * 2 / act_group_num
        for i in range(element_num):
            fi = ndarray(mesh.py_element(i))
            e_com = 0
            for vi in fi:
                e_com += ndarray(mesh.py_vertex(int(vi)))
            e_com /= len(fi)
            e_offset = e_com - com
            # Normal of this torus is [0, 1, 0].
            e_dir = np.cross(e_offset, ndarray([0, 1, 0]))
            e_dir = e_dir / np.linalg.norm(e_dir)
            deformable.AddActuation(actuator_stiffness[i], e_dir, [i,])
            angle = np.arctan2(e_offset[2], e_offset[0])
            idx = int(np.floor(angle / delta)) % act_group_num
            act_groups[idx].append(i)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__act_groups = act_groups

        scale = 3
        self._spp = options['spp'] if 'spp' in options else 8
        self._camera_pos = (0.5, -2, .25)
        self._camera_lookat = (0.5, 0, 0.1)
        self._color = (0.3, 0.7, 0.5)
        self._scale = scale

    def _actuator_parameter_to_stiffness(self, actuator_parameters):
        element_num = self._actuator_stiffness_dofs
        return ndarray(np.full((element_num,), 10 ** actuator_parameters[0]))

    # Returns a Jacobian:
    # Cols: actuator_parameters.
    # Rows: actuator stiffnesses.
    def _actuator_jacobian(self, actuator_parameters):
        n = self._actuator_stiffness_dofs
        jac = np.zeros((n, 1))
        # stiffness[i] = 10 ** actuator_parameters[i].
        for i in range(n):
            jac[i, 0] = (10 ** actuator_parameters[0]) * np.log(10)
        return ndarray(jac).copy()

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def act_groups(self):
        return self.__act_groups

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        loss = -com[0]
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[::3] = -1.0 / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
