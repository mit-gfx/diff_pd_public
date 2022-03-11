#include "state_force/gravitational_state_force.h"

template<int vertex_dim>
GravitationalStateForce<vertex_dim>::GravitationalStateForce()
    : StateForce<vertex_dim>(), mass_(0) {}

template<int vertex_dim>
void GravitationalStateForce<vertex_dim>::Initialize(const real mass, const Eigen::Matrix<real, vertex_dim, 1>& g) {
    mass_ = mass;
    StateForce<vertex_dim>::set_parameters(g);
}

template<int vertex_dim>
const VectorXr GravitationalStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    VectorXr f = VectorXr::Zero(dofs);
    for (int i = 0; i < dofs; ++i)
        f(i) = mass_ * g()(i % vertex_dim);
    return f;
}

template<int vertex_dim>
void GravitationalStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(StateForce<vertex_dim>::NumOfParameters());
    const int dofs = static_cast<int>(q.size());
    for (int i = 0; i < dofs; ++i) {
        dl_dp(i % vertex_dim) += dl_df(i) * mass_;
    }
}

template class GravitationalStateForce<2>;
template class GravitationalStateForce<3>;