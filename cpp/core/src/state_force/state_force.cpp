#include "state_force/state_force.h"

template<int vertex_dim>
const VectorXr StateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    return VectorXr::Zero(q.size());
}

template<int vertex_dim>
void StateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(parameters_.size());
}

template<int vertex_dim>
const std::vector<real> StateForce<vertex_dim>::PyForwardForce(const std::vector<real>& q, const std::vector<real>& v) const {
    return ToStdVector(ForwardForce(ToEigenVector(q), ToEigenVector(v)));
}

template<int vertex_dim>
void StateForce<vertex_dim>::PyBackwardForce(const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq,
    std::vector<real>& dl_dv, std::vector<real>& dl_dp) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_dp_eig;
    BackwardForce(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f), ToEigenVector(dl_df),
        dl_dq_eig, dl_dv_eig, dl_dp_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_dp = ToStdVector(dl_dp_eig);
}

template class StateForce<2>;
template class StateForce<3>;