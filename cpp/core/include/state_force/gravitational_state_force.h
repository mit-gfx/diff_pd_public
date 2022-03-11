#ifndef STATE_FORCE_GRAVITATIONAL_STATE_FORCE_H
#define STATE_FORCE_GRAVITATIONAL_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

template<int vertex_dim>
class GravitationalStateForce : public StateForce<vertex_dim> {
public:
    GravitationalStateForce();

    void Initialize(const real mass, const Eigen::Matrix<real, vertex_dim, 1>& g);
    void PyInitialize(const real mass, const std::array<real, vertex_dim>& g) {
        Eigen::Matrix<real, vertex_dim, 1> g_eig;
        for (int i = 0; i < vertex_dim; ++i) g_eig[i] = g[i];
        Initialize(mass, g_eig);
    }

    const real mass() const { return mass_; }
    const Eigen::Matrix<real, vertex_dim, 1> g() const { return StateForce<vertex_dim>::parameters().head(vertex_dim); }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    real mass_;
};

#endif