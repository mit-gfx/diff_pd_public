#include "pd_energy/pd_element_energy.h"

template<int dim>
void PdElementEnergy<dim>::Initialize(const real stiffness) { stiffness_ = stiffness; }

template<int dim>
const real PdElementEnergy<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    return stiffness_ * 0.5 * (F - ProjectToManifold(F)).squaredNorm();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    return stiffness_ * (F - ProjectToManifold(F));
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    return stiffness_ * (dF - ProjectToManifoldDifferential(F, dF));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> I;
    I.setZero();
    for (int i = 0; i < dim * dim; ++i) I(i, i) = 1;
    return stiffness_ * (I - ProjectToManifoldDifferential(F));
}

template<int dim>
const real PdElementEnergy<dim>::EnergyDensity(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection) const {
    return stiffness_ * 0.5 * (F_auxiliary.F() - projection).squaredNorm();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensor(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection) const {
    return stiffness_ * (F_auxiliary.F() - projection);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdElementEnergy<dim>::StressTensorDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
    const Eigen::Matrix<real, dim, dim>& projection, const Eigen::Matrix<real, dim, dim>& dF) const {
    return stiffness_ * (dF - ProjectToManifoldDifferential(F_auxiliary, projection, dF));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::StressTensorDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection) const {
    Eigen::Matrix<real, dim * dim, dim * dim> I;
    I.setZero();
    for (int i = 0; i < dim * dim; ++i) I(i, i) = 1;
    return stiffness_ * (I - ProjectToManifoldDifferential(F_auxiliary, projection));
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> J;
    J.setZero();
    for (int i = 0; i < dim * dim; ++i) {
        Eigen::Matrix<real, dim * dim, 1> dF;
        dF.setZero();
        dF(i) = 1;
        const Eigen::Matrix<real, dim, dim> F_col = ProjectToManifoldDifferential(F,
            Eigen::Map<const Eigen::Matrix<real, dim, dim>>(dF.data(), dim, dim));
        J.col(i) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(F_col.data(), F_col.size());
    }
    return J;
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> PdElementEnergy<dim>::ProjectToManifoldDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection) const {
    Eigen::Matrix<real, dim * dim, dim * dim> J;
    J.setZero();
    for (int i = 0; i < dim * dim; ++i) {
        Eigen::Matrix<real, dim * dim, 1> dF;
        dF.setZero();
        dF(i) = 1;
        const Eigen::Matrix<real, dim, dim> F_col = ProjectToManifoldDifferential(F_auxiliary, projection,
            Eigen::Map<const Eigen::Matrix<real, dim, dim>>(dF.data(), dim, dim));
        J.col(i) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(F_col.data(), F_col.size());
    }
    return J;
}

template class PdElementEnergy<2>;
template class PdElementEnergy<3>;