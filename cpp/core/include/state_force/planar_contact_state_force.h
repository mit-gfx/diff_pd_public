#ifndef STATE_FORCE_PLANAR_CONTACT_STATE_FORCE_H
#define STATE_FORCE_PLANAR_CONTACT_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

// Obstacle: normal.dot(q) + offset <= 0. In other words, the deformable body almost always stays on the positive side.
// This class defines the following smooth contact and friction force from [Macklin et al 2020], Sec. 5 and 6:
// "Primal/Dual Descent Methods for Dynamics"

template<int vertex_dim>
class PlanarContactStateForce : public StateForce<vertex_dim> {
public:
    void Initialize(const Eigen::Matrix<real, vertex_dim, 1>& normal, const real offset,
        // Non-trainable parameter(s). See [Macklin et al. 2020] for the definition.
        const int p,
        // Trainable parameters go below. See [Macklin et al. 2020] for the definition of each.
        const real kn,
        const real kf,
        const real mu);
    void PyInitialize(const std::array<real, vertex_dim>& normal, const real offset, const int p,
        const real kn, const real kf, const real mu) {
        Eigen::Matrix<real, vertex_dim, 1> normal_eig;
        for (int i = 0; i < vertex_dim; ++i) normal_eig[i] = normal[i];
        Initialize(normal_eig, offset, p, kn, kf, mu);
    }

    const real kn() const { return StateForce<vertex_dim>::parameters()(0); }
    const real kf() const { return StateForce<vertex_dim>::parameters()(1); }
    const real mu() const { return StateForce<vertex_dim>::parameters()(2); }
    const int p() const { return p_; }
    const Eigen::Matrix<real, vertex_dim, 1>& normal() const { return normal_; }
    const real offset() const { return offset_; }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    Eigen::Matrix<real, vertex_dim, 1> normal_;
    Eigen::Matrix<real, vertex_dim, vertex_dim> nnt_;   // normal * normal.T.
    real offset_;
    int p_;
};

#endif