#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "solver/deformable_preconditioner.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices) const {
    for (const auto& pair : dirichlet_) CheckError(q_cur(pair.first) == pair.second, "Boundary conditions violated.");

    const int sample_num = GetNumOfSamplesInElement();
    // Implements w * S' * A' * (d(BP)/dF) * A * (Sx).

    const int element_num = mesh_.NumOfElements();
    // Project PdElementEnergy.
    pd_backward_local_element_matrices.resize(element_num);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim> wAtdBpA; wAtdBpA.setZero();
        for (int j = 0; j < sample_num; ++j) {
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> wdBp; wdBp.setZero();
            int energy_cnt = 0;
            for (const auto& energy : pd_element_energies_) {
                const real w = energy->stiffness() * element_volume_ / sample_num;
                const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dBp
                    = energy->ProjectToManifoldDifferential(F_auxiliary_[i][j], projections_[energy_cnt][i][j]);
                wdBp += w * dBp;
                ++energy_cnt;
            }
            wAtdBpA += finite_element_samples_[i][j].pd_At() * wdBp * finite_element_samples_[i][j].pd_A();
        }
        pd_backward_local_element_matrices[i] = wAtdBpA;
    }
    // Project PdMuscleEnergy.
    pd_backward_local_muscle_matrices.resize(pd_muscle_energies_.size());
    int energy_idx = 0;
    int act_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const auto& Mt = energy->Mt();
        const int element_cnt = static_cast<int>(pair.second.size());
        pd_backward_local_muscle_matrices[energy_idx].resize(element_cnt);
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            pd_backward_local_muscle_matrices[energy_idx][ei].setZero();
            const real wi = energy->stiffness() * element_volume_ / sample_num;
            for (int j = 0; j < sample_num; ++j) {
                Eigen::Matrix<real, vertex_dim, vertex_dim * vertex_dim> JF;
                Eigen::Matrix<real, vertex_dim, 1> Ja;
                energy->ProjectToManifoldDifferential(F_auxiliary_[i][j].F(), a_cur(act_idx + ei), JF, Ja);
                pd_backward_local_muscle_matrices[energy_idx][ei] += wi * finite_element_samples_[i][j].pd_At()
                    * Mt * JF * finite_element_samples_[i][j].pd_A();
            }
        }
        act_idx += element_cnt;
        ++energy_idx;
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ApplyProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur,
    const VectorXr& a_cur,
    const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices,
    const VectorXr& dq_cur) const {
    CheckError(act_dofs_ == static_cast<int>(a_cur.size()), "Inconsistent actuation size.");

    // Implements w * S' * A' * (d(BP)/dF) * A * (Sx).
    const int element_num = mesh_.NumOfElements();
    std::vector<Eigen::Matrix<real, vertex_dim, element_dim>> pd_rhss(element_num,
        Eigen::Matrix<real, vertex_dim, element_dim>::Zero());
    // Project PdElementEnergy.
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto ddeformed = ScatterToElementFlattened(dq_cur, i);
        const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBpAx = pd_backward_local_element_matrices[i] * ddeformed;
        for (int k = 0; k < element_dim; ++k)
            pd_rhss[i].col(k) += wAtdBpAx.segment(k * vertex_dim, vertex_dim);
    }

    // Project PdMuscleEnergy:
    int act_idx = 0;
    int energy_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const int element_cnt = static_cast<int>(pair.second.size());
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const auto ddeformed = ScatterToElementFlattened(dq_cur, i);
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBpAx = pd_backward_local_muscle_matrices[energy_idx][ei] * ddeformed;
            for (int k = 0; k < element_dim; ++k)
                pd_rhss[i].col(k) += wAtdBpAx.segment(k * vertex_dim, vertex_dim);
        }
        act_idx += element_cnt;
        ++energy_idx;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j)
            pd_rhs.segment(vertex_dim * vi(j), vertex_dim) += pd_rhss[i].col(j);
    }

    // Project PdVertexEnergy.
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness();
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> dBptAd =
                energy->ProjectToManifoldDifferential(q_cur.segment(vertex_dim * idx, vertex_dim))
                * dq_cur.segment(vertex_dim * idx, vertex_dim);
            pd_rhs.segment(vertex_dim * idx, vertex_dim) += wi * dBptAd;
        }
    }
    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
    VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const {
    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    CheckError(options.find("use_bfgs") != options.end(), "Missing option use_bfgs.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));
    const bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = 0;
    int max_ls_iter = 0;
    if (use_bfgs) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        CheckError(bfgs_history_size >= 1, "Invalid bfgs_history_size.");
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));
    }
    // Optional flag for using acceleration technique in solving contacts. This should be used for
    // benchmarking only.
    bool use_acc = true;
    if (options.find("use_acc") != options.end()) {
        use_acc = static_cast<bool>(options.at("use_acc"));
        if (!use_acc && verbose_level > 0) {
            PrintWarning("use_acc is disabled. Unless you are benchmarking speed, you should not disable use_acc.");
        }
    }
    // Optional flag for using sparse matrices in PdLhsSolve.
    bool use_sparse = false;
    if (options.find("use_sparse") != options.end()) {
        use_sparse = static_cast<bool>(options.at("use_sparse"));
        if (use_sparse && verbose_level > 0) {
            PrintWarning("use_sparse is enabled. This is recommended when contact DoFs are large (e.g., > 300).");
        }
    }

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(method, dt, options);

    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_da = VectorXr::Zero(act_dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
    const int mat_w_dofs = NumOfPdElementEnergies();
    const int act_w_dofs = NumOfPdMuscleEnergies();
    dl_dmat_w = VectorXr::Zero(mat_w_dofs);
    dl_dact_w = VectorXr::Zero(act_w_dofs);

    const real h = dt;
    const real inv_h = ToReal(1) / h;
    // TODO: this mass is incorrect for tri or tet meshes.
    const real mass = element_volume_ * density_;
    const real h2m = h * h / mass;
    const real inv_h2m = mass / (h * h);
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> pd_backward_local_element_matrices;
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>> pd_backward_local_muscle_matrices;
    const bool use_precomputed_data = !pd_element_energies_.empty();
    if (use_precomputed_data) ComputeDeformationGradientAuxiliaryDataAndProjection(q_next);
    SetupProjectiveDynamicsLocalStepDifferential(q_next, a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices);

    // Forward:
    // Step 1:
    // rhs_basic = q + hv + h2m * f_ext + h2m * f_state(q, v).
    const VectorXr f_state_force = ForwardStateForce(q, v);
    // const VectorXr rhs_basic = q + h * v + h2m * f_ext + h2m * f_state_force;
    // Step 2:
    // rhs_dirichlet = rhs_basic(DoFs in dirichlet_) = dirichlet_.second.
    // VectorXr rhs_dirichlet = rhs_basic;
    // for (const auto& pair : dirichlet_) rhs_dirichlet(pair.first) = pair.second;
    // Step 3:
    // rhs = rhs_dirichlet(DoFs in active_contact_idx) = q.
    // VectorXr rhs = rhs_dirichlet;
    // for (const int idx : active_contact_idx) {
    //     for (int i = 0; i < vertex_dim; ++i) {
    //         rhs(idx * vertex_dim + i) = q(idx * vertex_dim + i);
    //     }
    // }
    // Step 4:
    // q_next - h2m * (f_pd(q_next) + f_act(q_next, a) + f_ela(q_next)) = rhs.
    // inv_h2m * q_next - (f_pd(q_next) + f_act(q_next, a) + f_ela(q_next)) = inv_h2m * rhs.
    std::map<int, real> augmented_dirichlet = dirichlet_;
    std::map<int, real> additional_dirichlet;
    for (const int idx : active_contact_idx) {
        for (int i = 0; i < vertex_dim; ++i) {
            const int dof = idx * vertex_dim + i;
            augmented_dirichlet[dof] = q(dof);
            additional_dirichlet[dof] = q(dof);
        }
    }
    // Step 5:
    // v_next = (q_next - q) / h.

    // Backward:
    // Step 5:
    // v_next = (q_next - q) / h.
    dl_dq += -dl_dv_next * inv_h;
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next * inv_h;

    // Step 4:
    // Newton equivalence:
    // Eigen::SimplicialLDLT<SparseMatrix> cholesky;
    // const SparseMatrix op = NewtonMatrix(q_next, a, inv_h2m, augmented_dirichlet);
    // cholesky.compute(op);
    // const VectorXr dl_drhs_intermediate = cholesky.solve(dl_dq_next_agg);
    // CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    VectorXr dl_drhs_intermediate;
    if (material_) {
        const SparseMatrix op = NewtonMatrix(q_next, a, inv_h2m, augmented_dirichlet, use_precomputed_data);
        if (EndsWith(method, "pcg")) {
            // The user is using a non-PD material model. Need special treatment.
            // Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            // const SparseMatrix op = NewtonMatrix(q_next, a, inv_h2m, augmented_dirichlet, use_precomputed_data);
            // cholesky.compute(op);
            // const VectorXr dl_drhs_intermediate = cholesky.solve(dl_dq_next_agg);
            // CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
            global_additional_dirichlet_boundary = additional_dirichlet;
            global_pd_backward_method = method;
            AssignToGlobalDeformable();
            Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::DeformablePreconditioner<real>> cg;
            const real tol = rel_tol + abs_tol / dl_dq_next_agg.norm();
            cg.setTolerance(tol);
            cg.compute(op);
            dl_drhs_intermediate = cg.solve(dl_dq_next_agg);
            CheckError(cg.info() == Eigen::Success, "CG solver failed.");
            ClearGlobalDeformable();
        } else {
            // Cholesky by default.
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            cholesky.compute(op);
            dl_drhs_intermediate = cholesky.solve(dl_dq_next_agg);
            CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
        }
    } else {
        // The PD equivalent (using the notation from the paper:)
        // A is the matrix in PdLhsSolve.
        // dA is the matrix in ApplyProjectiveDynamicsLocalStepDifferential.
        // In Newton's method: op * dl_drhs_intermediate = dl_dq_next_agg.
        // In PD:
        // (A - dA) * dl_drhs = dl_dq_next_agg.
        // Rows and cols in additional_dirichlet are erased.
        // This is equivalent to minimizing the following objective:
        // S := A - dA.
        // b := dl_dq_next_agg.
        // min_x 0.5 * x' * S * x - b' * x.
        // Note that in the minimization problem above, x(augment_dirichlet) must be 0, otherwise its gradient
        // will no longer be Sx - b. To see this point, assume x = [x_free, x_fixed] = [x_0, x_1] and S is defined as follows:
        // S = [S_00, S_01]
        //     [S_10, S_11]
        // min_x 0.5 * x_0 * S_00 * x_0 + x_1' * S_10 * x_0 - b * x_0.
        // Then its gradients become S_00 * x_0 + x_1' * S_10 - b, which is no longer what we want.
        // To resolve this issue, we set x(augmented_dirichlet) = 0. This will help us infer the gradients correctly.
        VectorXr x_sol = VectorXr::Zero(dofs_);   // Use the same initial guess as in Newton's method.
        VectorXr selected = VectorXr::Ones(dofs_);
        for (const auto& pair : augmented_dirichlet) {
            x_sol(pair.first) = 0;
            selected(pair.first) = 0;
        }
        VectorXr Sx_sol = PdLhsMatrixOp(x_sol, additional_dirichlet) - ApplyProjectiveDynamicsLocalStepDifferential(q_next,
            a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, x_sol);
        VectorXr grad_sol = (Sx_sol - dl_dq_next_agg).array() * selected.array();
        real obj_sol = 0.5 * x_sol.dot(Sx_sol) - dl_dq_next_agg.dot(x_sol);
        bool success = false;
        // Initialize queues for BFGS.
        std::deque<VectorXr> si_history, xi_history;
        std::deque<VectorXr> yi_history, gi_history;
        for (int i = 0; i < max_pd_iter; ++i) {
            if (verbose_level > 0) PrintInfo("PD iteration: " + std::to_string(i));
            if (use_bfgs) {
                // At each iteration, we maintain:
                // - x_sol
                // - Sx_sol
                // - grad_sol
                // - obj_sol
                // BFGS's direction: quasi_newton_direction = B * grad_sol.
                VectorXr quasi_newton_direction = VectorXr::Zero(dofs_);
                // Current solution: x_sol.
                // Current gradient: grad_sol.
                const int bfgs_size = static_cast<int>(xi_history.size());
                if (bfgs_size == 0) {
                    // Initially, the queue is empty. We use A as our initial guess of Hessian (not the inverse!).
                    xi_history.push_back(x_sol);
                    gi_history.push_back(grad_sol);
                    quasi_newton_direction = PdLhsSolve(method, grad_sol, additional_dirichlet, use_acc, use_sparse);
                } else {
                    const VectorXr x_sol_last = xi_history.back();
                    const VectorXr grad_sol_last = gi_history.back();
                    xi_history.push_back(x_sol);
                    gi_history.push_back(grad_sol);
                    si_history.push_back(x_sol - x_sol_last);
                    yi_history.push_back(grad_sol - grad_sol_last);
                    if (bfgs_size == bfgs_history_size + 1) {
                        xi_history.pop_front();
                        gi_history.pop_front();
                        si_history.pop_front();
                        yi_history.pop_front();
                    }
                    VectorXr bfgs_q = grad_sol;
                    std::deque<real> rhoi_history, alphai_history;
                    for (auto sit = si_history.crbegin(), yit = yi_history.crbegin(); sit != si_history.crend(); ++sit, ++yit) {
                        const VectorXr& yi = *yit;
                        const VectorXr& si = *sit;
                        const real rhoi = 1 / yi.dot(si);
                        const real alphai = rhoi * si.dot(bfgs_q);
                        rhoi_history.push_front(rhoi);
                        alphai_history.push_front(alphai);
                        bfgs_q -= alphai * yi;
                    }
                    // H0k = PdLhsSolve(I);
                    VectorXr z = PdLhsSolve(method, bfgs_q, additional_dirichlet, use_acc, use_sparse);
                    auto sit = si_history.cbegin(), yit = yi_history.cbegin();
                    auto rhoit = rhoi_history.cbegin(), alphait = alphai_history.cbegin();
                    for (; sit != si_history.cend(); ++sit, ++yit, ++rhoit, ++alphait) {
                        const real rhoi = *rhoit;
                        const real alphai = *alphait;
                        const VectorXr& si = *sit;
                        const VectorXr& yi = *yit;
                        const real betai = rhoi * yi.dot(z);
                        z += si * (alphai - betai);
                    }
                    quasi_newton_direction = z;
                }
                quasi_newton_direction = quasi_newton_direction.array() * selected.array();
                if (quasi_newton_direction.dot(grad_sol) < -ToReal(1e-4)) { // TODO: replace 1e-4 with a relative threshold.
                    // This implies the (inverse of) Hessian is indefinite, which means the objective to be minimized will
                    // become unbounded below. In this case, we choose to switch back to Newton's method.
                    success = false;
                    PrintWarning("Indefinite Hessian. BFGS is minimizing an unbounded objective.");
                    break;
                }

                // Line search --- keep in mind that grad/newton_direction points to the direction that *increases* the objective.
                if (verbose_level > 1) Tic();
                real step_size = 1;
                VectorXr x_sol_next = x_sol - step_size * quasi_newton_direction;
                VectorXr Sx_sol_next = PdLhsMatrixOp(x_sol_next, additional_dirichlet) -
                    ApplyProjectiveDynamicsLocalStepDifferential(q_next,
                        a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, x_sol_next);
                VectorXr grad_sol_next = (Sx_sol_next - dl_dq_next_agg).array() * selected.array();
                real obj_next = 0.5 * x_sol_next.dot(Sx_sol_next) - dl_dq_next_agg.dot(x_sol_next);
                const real gamma = ToReal(1e-4);
                bool ls_success = false;
                for (int j = 0; j < max_ls_iter; ++j) {
                    // Directional gradient: obj(q_sol - step_size * newton_direction)
                    //                     = obj_sol - step_size * newton_direction.dot(grad_sol)
                    const real obj_cond = obj_sol - gamma * step_size * grad_sol.dot(quasi_newton_direction);
                    const bool descend_condition = !std::isnan(obj_next) && obj_next < obj_cond + std::numeric_limits<real>::epsilon();
                    if (descend_condition) {
                        ls_success = true;
                        break;
                    }
                    step_size /= 2;
                    x_sol_next = x_sol - step_size * quasi_newton_direction;
                    Sx_sol_next = PdLhsMatrixOp(x_sol_next, additional_dirichlet) -
                        ApplyProjectiveDynamicsLocalStepDifferential(q_next,
                            a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, x_sol_next);
                    grad_sol_next = (Sx_sol_next - dl_dq_next_agg).array() * selected.array();
                    obj_next = 0.5 * x_sol_next.dot(Sx_sol_next) - dl_dq_next_agg.dot(x_sol_next);
                    if (verbose_level > 0) PrintInfo("Line search iteration: " + std::to_string(j));
                    if (verbose_level > 1) {
                        std::cout << "step size: " << step_size << std::endl;
                        std::cout << "obj_sol: " << obj_sol << ", "
                            << "obj_cond: " << obj_cond << ", "
                            << "obj_next: " << obj_next << ", "
                            << "obj_cond - obj_sol: " << obj_cond - obj_sol << ", "
                            << "obj_next - obj_sol: " << obj_next - obj_sol << std::endl;
                    }
                }
                if (verbose_level > 1) {
                    Toc("line search");
                    if (!ls_success) {
                        PrintWarning("Line search fails after " + std::to_string(max_ls_iter) + " trials.");
                    }
                }

                if (verbose_level > 1) std::cout << "obj_sol = " << obj_sol << ", obj_next = " << obj_next << std::endl;
                // Update.
                x_sol = x_sol_next;
                Sx_sol = Sx_sol_next;
                grad_sol = grad_sol_next;
                obj_sol = obj_next;
            } else {
                // Update w/o BFGS.
                // Local step:
                const VectorXr pd_rhs = (dl_dq_next_agg + ApplyProjectiveDynamicsLocalStepDifferential(q_next, a,
                    pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, x_sol)
                ).array() * selected.array();
                // Global step:
                x_sol = (PdLhsSolve(method, pd_rhs, additional_dirichlet, use_acc, use_sparse).array() * selected.array());
                Sx_sol = PdLhsMatrixOp(x_sol, additional_dirichlet) - ApplyProjectiveDynamicsLocalStepDifferential(q_next,
                    a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, x_sol);
                grad_sol = (Sx_sol - dl_dq_next_agg).array() * selected.array();
                obj_sol = 0.5 * x_sol.dot(Sx_sol) - dl_dq_next_agg.dot(x_sol);
            }

            // Check for convergence --- gradients must be zero.
            const real abs_error = grad_sol.norm();
            const real rhs_norm = dl_dq_next_agg.norm();
            if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
            if (abs_error <= rel_tol * rhs_norm + abs_tol) {
                success = true;
                for (const auto& pair : augmented_dirichlet) x_sol(pair.first) = dl_dq_next_agg(pair.first);
                break;
            }
        }
        dl_drhs_intermediate = x_sol;
        if (!success) {
            // Switch back to Newton's method.
            PrintWarning("PD backward: switching to Cholesky decomposition");
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            const SparseMatrix op = NewtonMatrix(q_next, a, inv_h2m, augmented_dirichlet, use_precomputed_data);
            cholesky.compute(op);
            dl_drhs_intermediate = cholesky.solve(dl_dq_next_agg);
            CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
        }
    }

    VectorXr dl_drhs = dl_drhs_intermediate * inv_h2m;
    // Now dl_drhs_free is correct. Working on dl_drhs_fixed next.
    for (const auto& pair: augmented_dirichlet) dl_drhs(pair.first) = dl_dq_next_agg(pair.first);
    // dl_drhs_fixed += -dl_dq_next_free * [dlhs/ dq_next_free]^(-1) * dlhs / drhs_fixed.
    // Let J = [A,  B] = NewtonMatrixOp(q_next, a, inv_h2m, {}).
    //         [B', C]
    // Let A corresponds to fixed dofs.
    // dl_drhs_fixed += -dl_dq_next_free * inv(C) * B'.
    // dl_drhs_intermediate_free = dl_dq_next_free * inv(C).
    VectorXr adjoint = dl_drhs_intermediate;
    for (const auto& pair : augmented_dirichlet) adjoint(pair.first) = 0;
    const VectorXr dfixed = NewtonMatrixOp(q_next, a, inv_h2m, {}, -adjoint);
    for (const auto& pair : augmented_dirichlet) dl_drhs(pair.first) += dfixed(pair.first);

    // Backpropagate a -> q_next and act_w -> q_next
    // inv_h2m * q_next_free - (f_ela(q_next_free; rhs_fixed) + f_pd(q_next_free; rhs_fixed)
    //     + f_act(q_next_free; rhs_fixed, a)) = inv_h2m * rhs_free.
    // C * dq_next_free / da - df_act / da = 0.
    // dl_da += dl_dq_next_agg * inv(C) * df_act / da.
    SparseMatrixElements nonzeros_q, nonzeros_a, nonzeros_act_w;
    ActuationForceDifferential(q_next, a, nonzeros_q, nonzeros_a, nonzeros_act_w);
    dl_da += VectorSparseMatrixProduct(adjoint, dofs_, act_dofs_, nonzeros_a);
    dl_dact_w += VectorSparseMatrixProduct(adjoint, dofs_, act_w_dofs, nonzeros_act_w);
    // Equivalent code:
    // dl_da += VectorXr(adjoint.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a));
    // dl_dact_w += VectorXr(adjoint.transpose() * ToSparseMatrix(dofs_, act_w_dofs, nonzeros_act_w));

    // Backpropagate w -> q_next.
    SparseMatrixElements nonzeros_mat_w;
    PdEnergyForceDifferential(q_next, false, true, use_precomputed_data, nonzeros_q, nonzeros_mat_w);
    dl_dmat_w += VectorSparseMatrixProduct(adjoint, dofs_, mat_w_dofs, nonzeros_mat_w);
    // Equivalent code:
    // dl_dw += VectorXr(adjoint.transpose() * ToSparseMatrix(dofs_, mat_w_dofs, nonzeros_mat_w));

    // Step 3:
    // rhs = rhs_dirichlet(DoFs in active_contact_idx) = q.
    VectorXr dl_drhs_dirichlet = dl_drhs;
    for (const int idx : active_contact_idx) {
        for (int i = 0; i < vertex_dim; ++i) {
            const int dof = idx * vertex_dim + i;
            dl_drhs_dirichlet(dof) = 0;
            dl_dq(dof) += dl_drhs(dof);
        }
    }

    // Step 2:
    // rhs_dirichlet = rhs_basic(DoFs in dirichlet_) = dirichlet_.second.
    VectorXr dl_drhs_basic = dl_drhs_dirichlet;
    for (const auto& pair : dirichlet_) {
        dl_drhs_basic(pair.first) = 0;
    }

    // Step 1:
    // rhs_basic = q + hv + h2m * f_ext + h2m * f_state(q, v).
    dl_dq += dl_drhs_basic;
    dl_dv += dl_drhs_basic * h;
    dl_df_ext += dl_drhs_basic * h2m;
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, f_state_force, dl_drhs_basic * h2m, dl_dq_single, dl_dv_single, dl_dstate_p);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
