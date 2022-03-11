#include "material/linear.h"
#include "common/geometry.h"

template<int dim>
const real LinearMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e = 0.5 * (F + F.transpose()) - I;
    const real trace_e = e.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * e.array().square().sum() + la / 2 * (trace_e * trace_e);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> LinearMaterial<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * (F + F.transpose() - 2 * I) + la * (F - I).trace() * I;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> LinearMaterial<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * (dF + dF.transpose()) + la * dF.trace() * I;
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> LinearMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    Eigen::Matrix<real, dim * dim, dim * dim> ret;
    ret.setZero();
    // mu * dF.
    for (int i = 0; i < dim * dim; ++i) ret(i, i) = mu;
    // mu * dF.transpose().
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            ret(i * dim + j, j * dim + i) += mu;
    // la * dF.trace() * I.
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            ret(i * dim + i, j * dim + j) += la;
    return ret;
}

template<int dim>
const real LinearMaterial<dim>::ComputeAverageStiffness(const real singular_value_range) const {
    CheckError(false, "Not implemented yet.");
    return 0;
}

template class LinearMaterial<2>;
template class LinearMaterial<3>;