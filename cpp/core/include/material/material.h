#ifndef MATERIAL_MATERIAL_H
#define MATERIAL_MATERIAL_H

#include "common/config.h"

template<int dim>
class Material {
public:
    Material();
    virtual ~Material() {}

    void Initialize(const real youngs_modulus, const real poissons_ratio);

    const real youngs_modulus() const {
        return youngs_modulus_;
    }
    const real poissons_ratio() const {
        return poissons_ratio_;
    }
    const real lambda() const {
        return lambda_;
    }
    const real mu() const {
        return mu_;
    }

    virtual const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const = 0;
    virtual const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F) const = 0;
    virtual const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const = 0;
    virtual const Eigen::Matrix<real, dim * dim, dim * dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F) const = 0;

    // Used by the quasi-Newton methods.
    // See https://tiantianliu.cn/papers/liu17quasi/liu17quasi.pdf
    // singular_value_range is in [0, 1] so that [x_start, x_end] = [1 - singular_value_range, 1 + singular_value_range].
    virtual const real ComputeAverageStiffness(const real singular_value_range) const = 0;

private:
    real youngs_modulus_;
    real poissons_ratio_;
    real lambda_;
    real mu_;
};

#endif