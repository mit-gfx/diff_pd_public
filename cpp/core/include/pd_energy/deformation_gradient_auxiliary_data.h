#ifndef PD_ENERGY_DEFORMATION_GRADIENT_AUXILIARY_DATA_H
#define PD_ENERGY_DEFORMATION_GRADIENT_AUXILIARY_DATA_H

#include "common/config.h"

template<int dim>
class DeformationGradientAuxiliaryData {
public:
    DeformationGradientAuxiliaryData() {}
    ~DeformationGradientAuxiliaryData() {}

    void Initialize(const Eigen::Matrix<real, dim, dim>& F);
    const Eigen::Matrix<real, dim, dim>& F() const { return F_; }
    const Eigen::Matrix<real, dim, dim>& U() const { return U_; }
    const Eigen::Matrix<real, dim, dim>& V() const { return V_; }
    const Eigen::Matrix<real, dim, 1>& sig() const { return sig_; }
    const Eigen::Matrix<real, dim, dim>& R() const { return R_; }
    const Eigen::Matrix<real, dim, dim>& S() const { return S_; }

private:
    // F = U * sig * V.transpose()
    // F = R * S.
    Eigen::Matrix<real, dim, dim> F_;
    Eigen::Matrix<real, dim, dim> U_, V_;
    Eigen::Matrix<real, dim, 1> sig_;
    Eigen::Matrix<real, dim, dim> R_;
    Eigen::Matrix<real, dim, dim> S_;
};

#endif