import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import export_gif, render_hex_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
#import IPython

class HopperEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement']
        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']
        leg_width = options['leg_width']
        half_leg_height = options['half_leg_height']
        waist_height = options['waist_height']
        waist_width = options['waist_width']
        thickness = options['thickness']
        assert leg_width % 2 == 0
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([
            np.log10(2) + 5,
            np.log10(2) + 5,
            np.log10(2) + 5,
            np.log10(2) + 5,
        ])
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])

        # Mesh parameters.
        cell_nums = (leg_width * 2 + waist_width, thickness, 2 * half_leg_height + waist_height)
        cell_nums = [n * refinement for n in cell_nums]
        node_nums = [n + 1 for n in cell_nums]
        dx = 0.025 / refinement
        origin = (0, 0, 0)
        voxels = np.ones(cell_nums)
        for i in range(leg_width * refinement, (leg_width + waist_width) * refinement):
            for k in list(range(half_leg_height * refinement)) \
                + list(range((half_leg_height + waist_height) * refinement, (half_leg_height * 2 + waist_height) * refinement)):
                voxels[i, :, k] = 0
        for i in range((leg_width + waist_width) * refinement, (leg_width * 2 + waist_width) * refinement):
            for k in range((half_leg_height + waist_height) * refinement, (half_leg_height * 2 + waist_height) * refinement):
                voxels[i, :, k] = 0
        bin_file_name = str(folder / 'mesh.bin')
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        deformable = HexDeformable()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        # External force.
        deformable.AddStateForce('gravity', state_force_parameters)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Actuation.
        element_num = mesh.NumOfElements()
        left_leg_left_fiber = []
        left_leg_right_fiber = []
        right_leg_left_fiber = []
        right_leg_right_fiber = []
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)
            x = com[0] / (refinement * dx)
            if 0 < x < leg_width // 2:
                left_leg_left_fiber.append(i)
            elif leg_width // 2 < x < leg_width:
                left_leg_right_fiber.append(i)
            elif leg_width + waist_width < x < leg_width + waist_width + leg_width // 2:
                right_leg_left_fiber.append(i)
            elif leg_width + waist_width + leg_width // 2 < x < leg_width + waist_width + leg_width:
                right_leg_right_fiber.append(i)
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], left_leg_left_fiber)
        deformable.AddActuation(actuator_stiffness[1], [0.0, 0.0, 1.0], left_leg_right_fiber)
        deformable.AddActuation(actuator_stiffness[2], [0.0, 0.0, 1.0], right_leg_left_fiber)
        deformable.AddActuation(actuator_stiffness[3], [0.0, 0.0, 1.0], right_leg_right_fiber)

        # Collision.
        friction_node_idx = get_contact_vertex(mesh)
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)
        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        '''

        # Initial conditions.
        dofs = deformable.dofs()
        print('Hopper element: {:d}, DoFs: {:d}.'.format(element_num, dofs))
        # Perturb q0 a bit to avoid singular gradients in SVD.
        q0 = ndarray(mesh.py_vertices()) + np.random.uniform(low=-0.01 * dx, high=0.01 * dx, size=dofs)
        # 5 body lengths per second horizontally.
        v0_mult = 5 * dx * cell_nums[0]
        v0 = np.zeros(dofs)
        v0[::3] = v0_mult
        v0[2::3] = -v0_mult

        # Data members.
        self._deformable = deformable
        self._dx = dx
        self._q0 = q0
        self._v0 = v0
        self._f_ext = np.zeros(dofs)
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self._left_leg_left_fiber = left_leg_left_fiber
        self._left_leg_right_fiber = left_leg_right_fiber
        self._right_leg_left_fiber = right_leg_left_fiber
        self._right_leg_right_fiber = right_leg_right_fiber
        self.__node_nums = node_nums
        self.__element_num = element_num

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def element_num(self):
        return self.__element_num

    def left_leg_left_fiber(self):
        return self._left_leg_left_fiber

    def left_leg_right_fiber(self):
        return self._left_leg_right_fiber

    def right_leg_left_fiber(self):
        return self._right_leg_left_fiber

    def right_leg_right_fiber(self):
        return self._right_leg_right_fiber

    def _display_mesh(self, mesh_file, file_name):
        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=8,
            camera_pos=[0.6, -2.2, 1.2],
            camera_lookat=[0.5, 0.5, 0.2],
             transforms=[
                ('s', 4)
            ])

    def _loss_and_grad(self, q, v):
        dx = self._dx
        # The L2 norm of the difference in pos and vel state compared to a target
        target_q = np.copy(self._q0)
        target_q[::3] += 10 * 0.045

        #q_diff = q - target_q
        #loss = 0.5 * q_diff.dot(q_diff)
        q_x = np.reshape(q, (-1, 3))[:, 0]
        loss = -np.mean(q_x)
        
        #grad_q = q_diff
        grad_q = np.zeros(q.shape)
        vertex_num = int(q.size // 3)
        grad_q[::3] = -1.0 / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v
