#ifndef MATERIAL_NEOHOOKEAN_H
#define MATERIAL_NEOHOOKEAN_H

#include "material/material.h"

template<int dim>
class NeohookeanMaterial : public Material<dim> {
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