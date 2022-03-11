#include "fem/deformable.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
    std::vector<int>& active_contact_idx) const {
    CheckError(!frictional_boundary_, "Semi-implicit methods do not support collisions.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    // Semi-implicit Euler.
    // Step 1: compute the predicted velocity:
    // v_pred = v + h / m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, a))
    const real mass = density_ * element_volume_;
    const VectorXr v_pred = v + dt / mass * (f_ext + ElasticForce(q) + ForwardStateForce(q, v)
        + PdEnergyForce(q, false) + ActuationForce(q, a));
    // Step 2: compute q_next via the semi-implicit rule:
    q_next = q + v_pred * dt;
    // Step 3: enforce dirichlet boundary conditions.
    for (const auto& pair : dirichlet_) q_next(pair.first) = pair.second;
    // Step 4: compute v_next.
    v_next = (q_next - q) / dt;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
    VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const {
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_da = VectorXr::Zero(act_dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
    const int mat_w_dofs = NumOfPdElementEnergies();
    const int act_w_dofs = NumOfPdMuscleEnergies();
    dl_dmat_w = VectorXr::Zero(mat_w_dofs);
    dl_dact_w = VectorXr::Zero(act_w_dofs);

    // Step 4: v_next = (q_next_after_collision - q) / dt;
    const real inv_dt = 1 / dt;
    const VectorXr dl_dq_next_after_collision = dl_dq_next + dl_dv_next * inv_dt;
    dl_dq += -dl_dv_next * inv_dt;

    VectorXr dl_dv_pred = VectorXr::Zero(dofs_);
    const real mass = density_ * element_volume_;
    const VectorXr v_pred = v + dt / mass * (f_ext + ElasticForce(q) + ForwardStateForce(q, v)
        + PdEnergyForce(q, false) + ActuationForce(q, a));

    // Step 3: dirichlet boundaries.
    VectorXr dl_dq_next_pred = dl_dq_next_after_collision;
    for (const auto& pair : dirichlet_) dl_dq_next_pred(pair.first) = 0;

    // Step 2: q_next_pred = q + v_pred * dt.
    dl_dq += dl_dq_next_pred;
    dl_dv_pred += dl_dq_next_pred * dt;

    // Step 1: v_pred = v + h / m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, a)).
    dl_dv += dl_dv_pred;
    const real hm = dt / mass;
    dl_df_ext += dl_dv_pred * hm;
    // f_ela(q).
    dl_dq += ElasticForceDifferential(q, dl_dv_pred) * hm;
    // f_state(q, v).
    VectorXr dl_dq_from_f_state, dl_dv_from_f_state;
    BackwardStateForce(q, v, ForwardStateForce(q, v), dl_dv_pred * hm, dl_dq_from_f_state, dl_dv_from_f_state, dl_dstate_p);
    dl_dq += dl_dq_from_f_state;
    dl_dv += dl_dv_from_f_state;
    // f_pd(q, w).
    SparseMatrixElements dpd_dq, dpd_dw;
    PdEnergyForceDifferential(q, false, true, false, dpd_dq, dpd_dw);
    dl_dq += PdEnergyForceDifferential(q, dl_dv_pred * hm, VectorXr::Zero(mat_w_dofs));
    dl_dmat_w += VectorXr(dl_dv_pred.transpose() * ToSparseMatrix(dofs_, mat_w_dofs, dpd_dw)) * hm;
    // f_act(q, a).
    SparseMatrixElements nonzeros_dq, nonzeros_da, nonzeros_dw;
    ActuationForceDifferential(q, a, nonzeros_dq, nonzeros_da, nonzeros_dw);
    const SparseMatrix dact_dq = ToSparseMatrix(dofs_, dofs_, nonzeros_dq);
    const SparseMatrix dact_da = ToSparseMatrix(dofs_, act_dofs_, nonzeros_da);
    const SparseMatrix dact_dw = ToSparseMatrix(dofs_, act_w_dofs, nonzeros_dw);
    dl_dq += VectorXr(dl_dv_pred.transpose() * dact_dq) * hm;
    dl_da += VectorXr(dl_dv_pred.transpose() * dact_da) * hm;
    dl_dact_w += VectorXr(dl_dv_pred.transpose() * dact_dw) * hm;
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
