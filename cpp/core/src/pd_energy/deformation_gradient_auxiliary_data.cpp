#include "pd_energy/deformation_gradient_auxiliary_data.h"

template<int dim>
void DeformationGradientAuxiliaryData<dim>::Initialize(const Eigen::Matrix<real, dim, dim>& F) {
    F_ = F;
    const Eigen::JacobiSVD<Eigen::Matrix<real, dim, dim>> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    sig_ = svd.singularValues();
    R_ = U_ * V_.transpose();
    S_ = V_ * sig_.asDiagonal() * V_.transpose();
}

template class DeformationGradientAuxiliaryData<2>;
template class DeformationGradientAuxiliaryData<3>;