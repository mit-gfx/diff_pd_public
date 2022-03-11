#include "pd_energy/planar_collision_pd_vertex_energy.h"
#include "common/common.h"

template<int dim>
void PlanarCollisionPdVertexEnergy<dim>::Initialize(const real stiffness,
    const Eigen::Matrix<real, dim, 1>& normal, const real offset) {
    PdVertexEnergy<dim>::Initialize(stiffness);
    const real norm = normal.norm();
    CheckError(norm > 1e-5, "Singular normal.");
    normal_ = normal / norm;
    offset_ = offset / norm;
    nnt_ = normal_ * normal_.transpose();
}

template<int dim>
const Eigen::Matrix<real, dim, 1> PlanarCollisionPdVertexEnergy<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, 1>& q) const {
    const real d = normal_.dot(q) + offset_;
    return q - std::min(ToReal(0), d) * normal_;
}

template<int dim>
const Eigen::Matrix<real, dim, 1> PlanarCollisionPdVertexEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& dq) const {
    const real d = normal_.dot(q) + offset_;
    // d <= 0 => q - d * normal_.
    if (d <= 0) return dq - nnt_ * dq;
    else return dq;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PlanarCollisionPdVertexEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, 1>& q) const {
    const real d = normal_.dot(q) + offset_;
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    if (d <= 0) return I - nnt_;
    else return I;
}

template class PlanarCollisionPdVertexEnergy<2>;
template class PlanarCollisionPdVertexEnergy<3>;