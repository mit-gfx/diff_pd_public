#ifndef PD_ENERGY_PD_ELEMENT_ENERGY_H
#define PD_ENERGY_PD_ELEMENT_ENERGY_H

#include "common/config.h"
#include "pd_energy/deformation_gradient_auxiliary_data.h"

// \Psi = stiffness / 2 * \|F - proj(F)\|^2.
// So w_i in the 2014 PD paper equals stiffness * cell_volume / sample_num.
template<int dim>
class PdElementEnergy {
public:
    PdElementEnergy() {}
    virtual ~PdElementEnergy() {}

    void Initialize(const real stiffness);

    const real stiffness() const { return stiffness_; }

    const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const;
    const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F) const;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const;
    const Eigen::Matrix<real, dim * dim, dim * dim> StressTensorDifferential(
        const Eigen::Matrix<real, dim, dim>& F) const;

    const real EnergyDensity(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const;
    const Eigen::Matrix<real, dim, dim> StressTensor(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection,
        const Eigen::Matrix<real, dim, dim>& dF) const;
    const Eigen::Matrix<real, dim * dim, dim * dim> StressTensorDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const;

    virtual const Eigen::Matrix<real, dim, dim> ProjectToManifold(const Eigen::Matrix<real, dim, dim>& F) const = 0;
    virtual const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF
    ) const = 0;
    virtual const Eigen::Matrix<real, dim * dim, dim * dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, dim, dim>& F) const;

    virtual const Eigen::Matrix<real, dim, dim> ProjectToManifold(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const = 0;
    virtual const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection,
        const Eigen::Matrix<real, dim, dim>& dF) const = 0;
    virtual const Eigen::Matrix<real, dim * dim, dim * dim> ProjectToManifoldDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary,
        const Eigen::Matrix<real, dim, dim>& projection) const;

private:
    real stiffness_;
};

#endif