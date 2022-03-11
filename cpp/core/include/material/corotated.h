#ifndef MATERIAL_COROTATED_H
#define MATERIAL_COROTATED_H

#include "material/material.h"

template<int dim>
class CorotatedMaterial : public Material<dim> {
public:
    const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F) const override;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const override;
    const Eigen::Matrix<real, dim * dim, dim * dim> StressTensorDifferential(
        const Eigen::Matrix<real, dim, dim>& F) const override;

    const real ComputeAverageStiffness(const real singular_value_range) const override;
};

#endif