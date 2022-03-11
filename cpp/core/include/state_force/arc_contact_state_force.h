#ifndef STATE_FORCE_ARC_CONTACT_STATE_FORCE_H
#define STATE_FORCE_ARC_CONTACT_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

// Obstacle is defined by:
// - center: 2D/3D.
// - dir: 2D/3D. Does not have to be normalized.
// - start: 2D/3D. Must be orthogonal to dir. Does not have to be normalized.
// - radius: >0.
// - angle: in radians.

// The deformable body almost always stays on the positive side.
// This class defines the following smooth contact and friction force from [Macklin et al 2020], Sec. 5 and 6:
// "Primal/Dual Descent Methods for Dynamics"

template<int vertex_dim>
class ArcContactStateForce : public StateForce<vertex_dim> {
public:
    void Initialize(
        const Eigen::Matrix<real, vertex_dim, 1>& center,
        const Eigen::Matrix<real, vertex_dim, 1>& dir,
        const Eigen::Matrix<real, vertex_dim, 1>& start,
        const real radius,
        const real angle,
        // Non-trainable parameter(s). See [Macklin et al. 2020] for the definition.
        const int p,
        // Trainable parameters go below. See [Macklin et al. 2020] for the definition of each.
        const real kn,
        const real kf,
        const real mu);
    void PyInitialize(const std::array<real, vertex_dim>& center,
        const std::array<real, vertex_dim>& dir,
        const std::array<real, vertex_dim>& start,
        const real radius,
        const real angle,
        const int p, const real kn, const real kf, const real mu) {
        Eigen::Matrix<real, vertex_dim, 1> center_eig, dir_eig, start_eig;
        for (int i = 0; i < vertex_dim; ++i) {
            center_eig(i) = center[i];
            dir_eig(i) = dir[i];
            start_eig(i) = start[i];
        }
        Initialize(center_eig, dir_eig, start_eig, radius, angle, p, kn, kf, mu);
    }

    const real kn() const { return StateForce<vertex_dim>::parameters()(0); }
    const real kf() const { return StateForce<vertex_dim>::parameters()(1); }
    const real mu() const { return StateForce<vertex_dim>::parameters()(2); }
    const int p() const { return p_; }
    const Eigen::Matrix<real, vertex_dim, 1>& center() const { return center_; }
    const Eigen::Matrix<real, vertex_dim, 1>& dir() const { return dir_; }
    const Eigen::Matrix<real, vertex_dim, 1>& start() const { return start_; }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    Eigen::Matrix<real, vertex_dim, 1> center_;
    Eigen::Matrix<real, vertex_dim, 1> dir_;
    Eigen::Matrix<real, vertex_dim, 1> start_;
    Eigen::Matrix<real, vertex_dim, 1> aux_;
    real radius_;
    real angle_;
    int p_;
};

#endif