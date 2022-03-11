from pathlib import Path
import time

import numpy as np

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray, create_folder, copy_std_int_vector
from py_diff_pd.common.display import export_gif, export_mp4
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import HexMesh3d

class EnvBase:
    def __init__(self, folder):
        self._deformable = None
        self._q0 = np.zeros(0)
        self._v0 = np.zeros(0)
        self._f_ext = np.zeros(0)
        self._youngs_modulus = 0
        self._poissons_ratio = 0
        self._actuator_parameters = np.zeros(0)
        self._state_force_parameters = np.zeros(0)
        self._stepwise_loss = False

        self._folder = Path(folder)

        # Rendering data members.
        self._spp = 4
        self._camera_pos = (0.4, -1, .25)
        self._camera_lookat = (0, 0.15, 0.15)
        self._color = (0.3, 0.7, 0.5)
        self._scale = 0.4
        self._resolution = (800, 800)

    def material_stiffness_differential(self, youngs_modulus, possions_ratio):
        raise NotImplementedError

    # Returns a 2 x 2 Jacobian:
    # Cols: youngs modulus, poissons ratio.
    # Rows: la, mu.
    def _material_jacobian(self, youngs_modulus, poissons_ratio):
        # la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        # mu = youngs_modulus / (2 * (1 + poissons_ratio))
        jac = np.zeros((2, 2))
        E = youngs_modulus
        nu = poissons_ratio
        jac[0, 0] = nu / ((1 + nu) * (1 - 2 * nu))
        jac[1, 0] = 1 / (2 * (1 + nu))
        jac[0, 1] = E * (1 + 2 * nu * nu) / (((1 + nu) * (1 - 2 * nu)) ** 2)
        jac[1, 1] = -(E / 2) / ((1 + nu) ** 2)
        return jac

    def _actuator_parameter_to_stiffness(self, actuator_parameters):
        return ndarray([10 ** p for p in actuator_parameters.ravel()])

    # Returns a Jacobian:
    # Cols: actuator_parameters.
    # Rows: actuator stiffnesses.
    def _actuator_jacobian(self, actuator_parameters):
        n = actuator_parameters.size
        jac = np.zeros((n, n))
        # stiffness[i] = 10 ** actuator_parameters[i].
        for i in range(n):
            jac[i, i] = (10 ** actuator_parameters[i]) * np.log(10)
        return ndarray(jac).copy()

    def is_dirichlet_dof(self, dof):
        raise NotImplementedError

    def deformable(self):
        return self._deformable

    def default_init_position(self):
        return np.copy(self._q0)

    def default_init_velocity(self):
        return np.copy(self._v0)

    def default_external_force(self):
        return np.copy(self._f_ext)

    # Default rendering method.
    # Modified through class data members.
    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': self._camera_pos,
            'camera_lookat': self._camera_lookat,
            'resolution': self._resolution
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=self._color, transforms=[
            ('s', self._scale),
        ])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 3)])

        renderer.render()

    # Return: loss, grad_q, grad_v.
    def _loss_and_grad(self, q, v):
        raise NotImplementedError

    def _stepwise_loss_and_grad(self, q, v, i):
        raise NotImplementedError

    # Input arguments:
    # dt: time step.
    # frame_num: number of frames.
    # method: 'semi_implicit' or 'newton_pcg' or 'newton_cholesky' or 'pd'.
    # method can be either a string or a tuple of two string (forward and backward).
    # opt: see each method. opt can optionally be a tuple of two options (forward and backward).
    # q0 and v0: if None, use the default initial values (see the two functions above).
    # act: either None or a list of size frame_num.
    # f_ext: either None or a list of size frame_num whose element is of size dofs.
    # requires_grad: True if you want to compute gradients.
    # vis_folder: if not None, `vis_folder.gif` will be generated under self._folder.
    #
    # Return value:
    # If require_grad=True: loss, info;
    # if require_grad=False: loss, grad, info.
    def simulate(self, dt, frame_num, method, opt, q0=None, v0=None, act=None, f_ext=None,
        require_grad=False, vis_folder=None, velocity_bound=np.inf, render_frame_skip=1):
        # Check input parameters.
        assert dt > 0
        assert frame_num > 0
        if isinstance(method, str):
            forward_method = method
            backward_method = method
        elif isinstance(method, tuple):
            assert len(method) == 2
            forward_method, backward_method = method
        else:
            raise NotImplementedError
        if isinstance(opt, dict):
            forward_opt = opt
            backward_opt = opt
        elif isinstance(opt, tuple):
            assert len(opt) == 2
            forward_opt, backward_opt = opt

        if q0 is None:
            sim_q0 = np.copy(self._q0)
        else:
            sim_q0 = np.copy(ndarray(q0))
        assert sim_q0.size == self._q0.size

        if v0 is None:
            sim_v0 = np.copy(self._v0)
        else:
            sim_v0 = np.copy(ndarray(v0))
        assert sim_v0.size == self._v0.size

        if act is None:
            sim_act = [np.zeros(self._deformable.act_dofs()) for _ in range(frame_num)]
        else:
            sim_act = [ndarray(a) for a in act]
        assert len(sim_act) == frame_num
        for a in sim_act:
            assert a.size == self._deformable.act_dofs()

        if f_ext is None:
            sim_f_ext = [self._f_ext for _ in range(frame_num)]
        else:
            sim_f_ext = [ndarray(f) for f in f_ext]
        assert len(sim_f_ext) == frame_num
        for f in sim_f_ext:
            assert f.size == self._deformable.dofs()

        if vis_folder is not None:
            create_folder(self._folder / vis_folder, exist_ok=False)

        # Forward simulation.
        t_begin = time.time()

        def clamp_velocity(unclamped_vel):
            clamped_vel = np.clip(np.copy(ndarray(unclamped_vel)), -velocity_bound, velocity_bound)
            return clamped_vel

        q = [sim_q0,]
        v = [sim_v0,]
        v_clamped = []
        # Computational graph:
        # Clamp v[i] to get v_clamped[i].
        # Forward sim(q[i], v_clamped[i]) to obtain q[i + 1] and v[i + 1].
        dofs = self._deformable.dofs()
        loss = 0
        grad_q = np.zeros(dofs)
        grad_v = np.zeros(dofs)
        grad_custom = {}
        active_contact_indices = [StdIntVector(0),]
        time_per_frame = []
        for i in range(frame_num):
            # Record the time.
            t_begin = time.time()

            # Computation begins here.
            v_clamped.append(clamp_velocity(v[-1]))
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            active_contact_idx = copy_std_int_vector(active_contact_indices[-1])
            self._deformable.PyForward(forward_method, q[-1], v_clamped[-1], sim_act[i], sim_f_ext[i], dt, forward_opt,
                q_next_array, v_next_array, active_contact_idx)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            active_contact_indices.append(active_contact_idx)
            if self._stepwise_loss:
                # See if a custom grad is provided.
                ret = self._stepwise_loss_and_grad(q_next, v_next, i + 1)
                l, grad_q, grad_v = ret[:3]
                if len(ret) > 3:
                    grad_c = ret[3]
                    for grad_c_key, grad_c_val in grad_c.items():
                        if grad_c_key in grad_custom:
                            grad_custom[grad_c_key] += grad_c_val
                        else:
                            grad_custom[grad_c_key] = grad_c_val
                loss += l
            elif i == frame_num - 1:
                ret = self._loss_and_grad(q_next, v_next)
                l, grad_q, grad_v = ret[:3]
                if len(ret) > 3:
                    grad_c = ret[3]
                    for grad_c_key, grad_c_val in grad_c.items():
                        if grad_c_key in grad_custom:
                            grad_custom[grad_c_key] += grad_c_val
                        else:
                            grad_custom[grad_c_key] = grad_c_val
                loss += l
            q.append(q_next)
            v.append(v_next)

            # Record the time.
            t_end = time.time()
            time_per_frame.append(t_end - t_begin)

        # Save data.
        info = { 'grad_custom': grad_custom }
        info['q'] = q
        info['v'] = v
        info['active_contact_indices'] = [list(a) for a in active_contact_indices]

        # Compute loss.
        info['forward_time'] = np.sum(time_per_frame)
        info['forward_time_per_frame'] = np.copy(ndarray(time_per_frame))

        if vis_folder is not None:
            t_begin = time.time()
            for i, qi in enumerate(q):
                if i % render_frame_skip != 0: continue
                mesh_file = str(self._folder / vis_folder / '{:04d}.bin'.format(i))
                self._deformable.PySaveToMeshFile(qi, mesh_file)
                self._display_mesh(mesh_file, self._folder / vis_folder / '{:04d}.png'.format(i))
            export_mp4(self._folder / vis_folder, self._folder / '{}.mp4'.format(vis_folder), 20)

            t_vis = time.time() - t_begin
            info['visualize_time'] = t_vis

        if not require_grad:
            return loss, info
        else:
            t_begin = time.time()
            dl_dq_next = np.copy(grad_q)
            dl_dv_next = np.copy(grad_v)
            act_dofs = self._deformable.act_dofs()
            dl_act = np.zeros((frame_num, act_dofs))
            dl_df_ext = np.zeros((frame_num, dofs))
            mat_w_dofs = self._deformable.NumOfPdElementEnergies()
            act_w_dofs = self._deformable.NumOfPdMuscleEnergies()
            state_p_dofs = self._deformable.NumOfStateForceParameters()
            dl_dmat_w = np.zeros(mat_w_dofs)
            dl_dact_w = np.zeros(act_w_dofs)
            dl_dstate_p = np.zeros(state_p_dofs)
            for i in reversed(range(frame_num)):
                # i -> i + 1.
                dl_dq = StdRealVector(dofs)
                dl_dv_clamped = StdRealVector(dofs)
                dl_da = StdRealVector(act_dofs)
                dl_df = StdRealVector(dofs)
                dl_dmat_wi = StdRealVector(mat_w_dofs)
                dl_dact_wi = StdRealVector(act_w_dofs)
                dl_dstate_pi = StdRealVector(state_p_dofs)
                self._deformable.PyBackward(backward_method, q[i], v_clamped[i], sim_act[i], sim_f_ext[i], dt,
                    q[i + 1], v[i + 1], active_contact_indices[i + 1], dl_dq_next, dl_dv_next,
                    backward_opt, dl_dq, dl_dv_clamped, dl_da, dl_df, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)
                # Backpropagate v_clamped[i] = clip(v[i], -velocity_bound, velocity_bound).
                dl_dv_clamped = ndarray(dl_dv_clamped)
                dl_dv = np.copy(dl_dv_clamped)
                for k in range(dofs):
                    if v[i][k] == -velocity_bound or v[i][k] == velocity_bound:
                        dl_dv[k] = 0
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                if self._stepwise_loss and i != 0:
                    ret = self._stepwise_loss_and_grad(q[i], v[i], i)
                    dqi, dvi = ret[1], ret[2]
                    dl_dq_next += ndarray(dqi)
                    dl_dv_next += ndarray(dvi)
                dl_act[i] = ndarray(dl_da)
                dl_df_ext[i] = ndarray(dl_df)
                dl_dmat_w += ndarray(dl_dmat_wi)
                dl_dact_w += ndarray(dl_dact_wi)
                dl_dstate_p += ndarray(dl_dstate_pi)
            grad = [np.copy(dl_dq_next), np.copy(dl_dv_next), dl_act, dl_df_ext]
            t_grad = time.time() - t_begin
            info['backward_time'] = t_grad
            info['material_parameter_gradients'] = ndarray(dl_dmat_w.T @ self.material_stiffness_differential(
                self._youngs_modulus, self._poissons_ratio)).ravel()
            if act_w_dofs > 0:
                info['actuator_parameter_gradients'] = ndarray(dl_dact_w.T @ self._actuator_jacobian(
                    self._actuator_parameters)).ravel()
            else:
                info['actuator_parameter_gradients'] = ndarray(np.zeros(0))
            if state_p_dofs > 0:
                info['state_force_parameter_gradients'] = ndarray(dl_dstate_p)
            else:
                info['state_force_parameter_gradients'] = ndarray(np.zeros(0))
            return loss, grad, info
