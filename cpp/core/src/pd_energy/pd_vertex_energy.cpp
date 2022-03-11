#include "pd_energy/pd_vertex_energy.h"

template<int dim>
void PdVertexEnergy<dim>::Initialize(const real stiffness) {
    stiffness_ = stiffness;   
}

template<int dim>
const real PdVertexEnergy<dim>::PotentialEnergy(const Eigen::Matrix<real, dim, 1>& q) const {
    return stiffness_ * 0.5 * (q - ProjectToManifold(q)).squaredNorm();
}

template<int dim>
const Eigen::Matrix<real, dim, 1> PdVertexEnergy<dim>::PotentialForce(
    const Eigen::Matrix<real, dim, 1>& q) const {
    return -stiffness_ * (q - ProjectToManifold(q));
}

template<int dim>
const Eigen::Matrix<real, dim, 1> PdVertexEnergy<dim>::PotentialForceDifferential(
    const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& dq) const {
    return -stiffness_ * (dq - ProjectToManifoldDifferential(q, dq));
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdVertexEnergy<dim>::PotentialForceDifferential(
    const Eigen::Matrix<real, dim, 1>& q) const {
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    return -stiffness_ * (I - ProjectToManifoldDifferential(q));
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdVertexEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, 1>& q) const {
    Eigen::Matrix<real, dim, dim> J;
    J.setZero();
    for (int i = 0; i < dim; ++i) {
        Eigen::Matrix<real, dim, 1> dq;
        dq.setZero();
        dq(i) = 1;
        J.col(i) = ProjectToManifoldDifferential(q, dq);
    }
    return J;
}

template class PdVertexEnergy<2>;
template class PdVertexEnergy<3>;