#include "state_force/arc_contact_state_force.h"

template<>
void ArcContactStateForce<2>::Initialize(const Vector2r& center, const Vector2r& dir, const Vector2r& start,
    const real radius, const real angle, const int p, const real kn, const real kf, const real mu) {
    Vector3r parameters(kn, kf, mu);
    StateForce<2>::set_parameters(parameters);
    center_ = center;
    dir_.setZero(); // Does not matter.
    start_ = start / start.norm();              // Local x.
    aux_ = Vector2r(-start_.y(), start_.x());   // Local y.
    radius_ = radius;
    angle_ = angle;
    p_ = p;
}

template<>
void ArcContactStateForce<3>::Initialize(const Vector3r& center, const Vector3r& dir, const Vector3r& start,
    const real radius, const real angle, const int p, const real kn, const real kf, const real mu) {
    Vector3r parameters(kn, kf, mu);
    StateForce<3>::set_parameters(parameters);
    center_ = center;
    dir_ = dir / dir.norm();
    start_ = start / start.norm();  // Local x.
    aux_ = dir_.cross(start_);      // Local y.
    radius_ = radius;
    angle_ = angle;
    p_ = p;
}

template<int vertex_dim>
const VectorXr ArcContactStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    const int vertex_num = dofs / vertex_dim;
    const real eps = std::numeric_limits<real>::epsilon();
    VectorXr f = VectorXr::Zero(dofs);
    for (int i = 0; i < vertex_num; ++i) {
        // Compute the distance to the plane. Negative distances mean penetration.
        const Eigen::Matrix<real, vertex_dim, 1> qi_local = q.segment(vertex_dim * i, vertex_dim) - center_;
        const auto q_proj = qi_local - qi_local.dot(dir_) * dir_;
        // Compute angle.
        const real angle = std::atan2(q_proj.dot(aux_), q_proj.dot(start_));
        if (0 < angle && angle < angle_) {
            // Collision happens.
            const real q_proj_norm = q_proj.norm();
            const real d = radius_ - q_proj_norm;
            // This is the distance to the plane. Negative distances mean penetration.
            const real fn_mag = kn() * std::pow(-std::min(ToReal(0.0), d), p_ - 1);
            // Compute the contact force.
            const Eigen::Matrix<real, vertex_dim, 1> normal = -q_proj / (q_proj.norm() + eps);
            const Eigen::Matrix<real, vertex_dim, 1> fn = fn_mag * normal;
            // Compute the frictional force.
            const Eigen::Matrix<real, vertex_dim, 1> vi = v.segment(vertex_dim * i, vertex_dim);
            const Eigen::Matrix<real, vertex_dim, 1> us = vi - vi.dot(normal) * normal;
            const real us_mag = us.norm();
            const Eigen::Matrix<real, vertex_dim, 1> ff = -std::min(kf(), mu() * fn_mag / (us_mag + eps)) * us;
            f.segment(vertex_dim * i, vertex_dim) = fn + ff;
        }
    }
    return f;
}

template<int vertex_dim>
void ArcContactStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    CheckError(q.size() == v.size() && v.size() == f.size() && f.size() == dl_df.size(), "Inconsistent vector size.");
    const int vertex_num = dofs / vertex_dim;
    const real eps = std::numeric_limits<real>::epsilon();
    dl_dq = VectorXr::Zero(dofs);
    dl_dv = VectorXr::Zero(dofs);
    dl_dp = VectorXr::Zero(3);
    const Eigen::Matrix<real, vertex_dim, vertex_dim> I = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
    for (int i = 0; i < vertex_num; ++i) {
        // Compute the distance to the plane. Negative distances mean penetration.
        const Eigen::Matrix<real, vertex_dim, 1> qi_local = q.segment(vertex_dim * i, vertex_dim) - center_;
        const auto q_proj = qi_local - qi_local.dot(dir_) * dir_;
        const Eigen::Matrix<real, vertex_dim, vertex_dim> dq_proj_dq = I - dir_ * dir_.transpose();
        // Compute angle.
        const real angle = std::atan2(q_proj.dot(aux_), q_proj.dot(start_));
        if (0 < angle && angle < angle_) {
            // Collision happens.
            const real q_proj_norm = q_proj.norm();
            const Eigen::Matrix<real, vertex_dim, 1> dq_proj_norm_dq = dq_proj_dq.transpose() * q_proj / (q_proj_norm + eps);
            const real d = radius_ - q_proj_norm;
            const Eigen::Matrix<real, vertex_dim, 1> dd_dq = -dq_proj_norm_dq;
            // This is the distance to the plane. Negative distances mean penetration.
            const real fn_mag = kn() * std::pow(-std::min(ToReal(0.0), d), p_ - 1);
            const Eigen::Matrix<real, vertex_dim, 1> dfn_mag_dq = kn() * (p_ - 1)
                * std::pow(-std::min(ToReal(0.0), d), p_ - 2) * -(0 <= d ? Eigen::Matrix<real, vertex_dim, 1>::Zero() : dd_dq);
            Vector3r dfn_mag_dp(fn_mag / kn(), 0, 0);
            // Compute the contact force.
            const auto normal = -q_proj / (q_proj_norm + eps);
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dnormal_dq = -(I - normal * normal.transpose()) / (q_proj_norm + eps) * dq_proj_dq;
            // const Eigen::Matrix<real, vertex_dim, 1> fn = fn_mag * normal;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dfn_dq = normal * dfn_mag_dq.transpose() + dnormal_dq * fn_mag;
            const Eigen::Matrix<real, vertex_dim, 3> dfn_dp = normal * dfn_mag_dp.transpose();
            // Compute the frictional force.
            const Eigen::Matrix<real, vertex_dim, 1> vi = v.segment(vertex_dim * i, vertex_dim);
            const Eigen::Matrix<real, vertex_dim, 1> us = vi - vi.dot(normal) * normal;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dus_dq = -(normal * vi.transpose() + I * (vi.dot(normal))) * dnormal_dq;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dus_dv =
                Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity() - normal * normal.transpose();
            const real us_mag = us.norm();
            const Eigen::Matrix<real, vertex_dim, 1> dus_mag_dq = dus_dq * us / (us_mag + eps);
            const Eigen::Matrix<real, vertex_dim, 1> dus_mag_dv = dus_dv * us / (us_mag + eps);
            // const Eigen::Matrix<real, vertex_dim, 1> ff = -std::min(kf(), mu() * fn_mag / (us_mag + eps)) * us;
            Eigen::Matrix<real, vertex_dim, vertex_dim> dff_dq; dff_dq.setZero();
            Eigen::Matrix<real, vertex_dim, vertex_dim> dff_dv; dff_dv.setZero();
            Eigen::Matrix<real, vertex_dim, 3> dff_dp; dff_dp.setZero();
            if (kf() <= mu() * fn_mag / (us_mag + eps)) {
                // ff = -kf * us.
                dff_dq = -kf() * dus_dq;
                dff_dv = -kf() * dus_dv;
                dff_dp.col(1) = -us;
            } else {
                // ff = -mu * fn_mag / (us_mag + eps) * us.
                dff_dq = -mu() * (
                    us * dfn_mag_dq.transpose() / (us_mag + eps) +
                    fn_mag * (dus_dq / (us_mag + eps) - us * dus_mag_dq.transpose() / ((us_mag + eps) * (us_mag + eps)))
                );
                dff_dv = -mu() * fn_mag * (dus_dv / (us_mag + eps) - us * dus_mag_dv.transpose() / ((us_mag + eps) * (us_mag + eps)));
                dff_dp = -us / (us_mag + eps) * (mu() * dfn_mag_dp.transpose() + RowVector3r(0, 0, fn_mag));
            }   
            // f.segment(vertex_dim * i, vertex_dim) = fn + ff;
            dl_dq.segment(vertex_dim * i, vertex_dim) = (dfn_dq + dff_dq).transpose() * dl_df.segment(vertex_dim * i, vertex_dim);
            dl_dv.segment(vertex_dim * i, vertex_dim) = dff_dv.transpose() * dl_df.segment(vertex_dim * i, vertex_dim);
            dl_dp += VectorXr(dl_df.segment(vertex_dim * i, vertex_dim).transpose() * (dfn_dp + dff_dp));
        }
    }
}

template class ArcContactStateForce<2>;
template class ArcContactStateForce<3>;