#ifndef PD_ENERGY_COROTATED_PD_ELEMENT_ENERGY_H
#define PD_ENERGY_COROTATED_PD_ELEMENT_ENERGY_H

#include "pd_energy/pd_element_energy.h"

template<int dim>
class CorotatedPdElementEnergy : public PdElementEnergy<dim> {
public:
    const Eigen::Matrix<real, dim, dim> ProjectToManifold(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF
    ) const override;

    const Eigen::Matrix<real, dim, dim> ProjectToManifold(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const override;
    const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection,
        const Eigen::Matrix<real, dim, dim>& dF) const override;
};

#endif