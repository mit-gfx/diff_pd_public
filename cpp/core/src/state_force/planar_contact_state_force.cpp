#include "state_force/planar_contact_state_force.h"

template<int vertex_dim>
void PlanarContactStateForce<vertex_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, 1>& normal, const real offset,
    const int p, const real kn, const real kf, const real mu) {
    Vector3r parameters(kn, kf, mu);
    StateForce<vertex_dim>::set_parameters(parameters);
    const real norm = normal.norm();
    CheckError(norm > 1e-5, "Singular normal.");
    normal_ = normal / norm;
    offset_ = offset / norm;
    nnt_ = normal_ * normal_.transpose();
    p_ = p;
}

template<int vertex_dim>
const VectorXr PlanarContactStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    const int vertex_num = dofs / vertex_dim;
    const real eps = std::numeric_limits<real>::epsilon();
    VectorXr f = VectorXr::Zero(dofs);
    for (int i = 0; i < vertex_num; ++i) {
        const real d = normal_.dot(q.segment(vertex_dim * i, vertex_dim)) + offset_;
        // This is the distance to the plane. Negative distances mean penetration.
        const real fn_mag = kn() * std::pow(-std::min(ToReal(0.0), d), p_ - 1);
        const Eigen::Matrix<real, vertex_dim, 1> fn = fn_mag * normal_;
        // Compute the frictional force.
        const Eigen::Matrix<real, vertex_dim, 1> vi = v.segment(vertex_dim * i, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> us = vi - vi.dot(normal_) * normal_;
        const real us_mag = us.norm();
        const Eigen::Matrix<real, vertex_dim, 1> ff = -std::min(kf(), mu() * fn_mag / (us_mag + eps)) * us;
        f.segment(vertex_dim * i, vertex_dim) = fn + ff;
    }
    return f;
}

template<int vertex_dim>
void PlanarContactStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    CheckError(q.size() == v.size() && v.size() == f.size() && f.size() == dl_df.size(), "Inconsistent vector size.");
    const int vertex_num = dofs / vertex_dim;
    const real eps = std::numeric_limits<real>::epsilon();
    dl_dq = VectorXr::Zero(dofs);
    dl_dv = VectorXr::Zero(dofs);
    dl_dp = VectorXr::Zero(3);
    for (int i = 0; i < vertex_num; ++i) {
        const real d = normal_.dot(q.segment(vertex_dim * i, vertex_dim)) + offset_;
        const Eigen::Matrix<real, vertex_dim, 1> dd_dq = normal_;
        const real fn_mag = kn() * std::pow(-std::min(ToReal(0.0), d), p_ - 1);
        const Eigen::Matrix<real, vertex_dim, 1> dfn_mag_dq = kn() * (p_ - 1)
            * std::pow(-std::min(ToReal(0.0), d), p_ - 2) * -(0 <= d ? Eigen::Matrix<real, vertex_dim, 1>::Zero() : dd_dq);
        Vector3r dfn_mag_dp(fn_mag / kn(), 0, 0);
        // const Eigen::Matrix<real, vertex_dim, 1> fn = fn_mag * normal_;
        const Eigen::Matrix<real, vertex_dim, vertex_dim> dfn_dq = normal_ * dfn_mag_dq.transpose();
        const Eigen::Matrix<real, vertex_dim, 3> dfn_dp = normal_ * dfn_mag_dp.transpose();
        // Compute the frictional force.
        const Eigen::Matrix<real, vertex_dim, 1> vi = v.segment(vertex_dim * i, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> us = vi - vi.dot(normal_) * normal_;
        const Eigen::Matrix<real, vertex_dim, vertex_dim> dus_dv =
            Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity() - nnt_;
        const real us_mag = us.norm();
        const Eigen::Matrix<real, vertex_dim, 1> dus_mag_dv = dus_dv * us / (us_mag + eps);
        // const Eigen::Matrix<real, vertex_dim, 1> ff = -std::min(kf(), mu() * fn_mag / (us_mag + eps)) * us;
        Eigen::Matrix<real, vertex_dim, vertex_dim> dff_dq; dff_dq.setZero();
        Eigen::Matrix<real, vertex_dim, vertex_dim> dff_dv; dff_dv.setZero();
        Eigen::Matrix<real, vertex_dim, 3> dff_dp; dff_dp.setZero();
        if (kf() <= mu() * fn_mag / (us_mag + eps)) {
            // ff = -kf * us.
            dff_dv = -kf() * dus_dv;
            dff_dp.col(1) = -us;
        } else {
            // ff = -mu * fn_mag / (us_mag + eps) * us.
            dff_dq = -mu() * us / (us_mag + eps) * dfn_mag_dq.transpose();
            dff_dv = -mu() * fn_mag * (dus_dv / (us_mag + eps) - us * dus_mag_dv.transpose() / ((us_mag + eps) * (us_mag + eps)));
            dff_dp = -us / (us_mag + eps) * (mu() * dfn_mag_dp.transpose() + RowVector3r(0, 0, fn_mag));
        }
        dl_dq.segment(vertex_dim * i, vertex_dim) = (dfn_dq + dff_dq).transpose() * dl_df.segment(vertex_dim * i, vertex_dim);
        dl_dv.segment(vertex_dim * i, vertex_dim) = dff_dv.transpose() * dl_df.segment(vertex_dim * i, vertex_dim);
        dl_dp += VectorXr(dl_df.segment(vertex_dim * i, vertex_dim).transpose() * (dfn_dp + dff_dp));
    }
}

template class PlanarContactStateForce<2>;
template class PlanarContactStateForce<3>;