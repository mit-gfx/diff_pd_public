#include "pd_energy/volume_pd_element_energy.h"
#include "common/common.h"
#include "common/geometry.h"

template<int dim>
const Eigen::Matrix<real, dim, dim> VolumePdElementEnergy<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, dim>& F) const {
    DeformationGradientAuxiliaryData<dim> F_auxiliary;
    F_auxiliary.Initialize(F);
    return ProjectToManifold(F_auxiliary);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> VolumePdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const {
    DeformationGradientAuxiliaryData<dim> F_auxiliary;
    F_auxiliary.Initialize(F);
    return ProjectToManifoldDifferential(F_auxiliary, Eigen::Matrix<real, dim, dim>::Zero(), dF);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> VolumePdElementEnergy<dim>::ProjectToManifold(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const {
    const Eigen::Matrix<real, dim, 1>& sig = F_auxiliary.sig();
    const Eigen::Matrix<real, dim, dim>& U = F_auxiliary.U();
    const Eigen::Matrix<real, dim, dim>& V = F_auxiliary.V();
    // F = U * sig * V.transpose();
    // Now solve the problem:
    // min \|sig - sig*\|^2
    // s.t. \Pi sig* = 1.
    // See the appendix of the 2014 PD paper for the derivation.
    // Let D = sig* - sig.
    // min \|D\|^2 s.t. \Pi (sig(i) + D(i)) = 1.

    const real eps = std::numeric_limits<real>::epsilon();
    CheckError(sig.minCoeff() >= eps, "Singular F.");

    // Initial guess.
    Eigen::Matrix<real, dim, 1> D = sig / std::pow(sig.prod(), ToReal(1) / dim) - sig;
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i) {
        // Compute C(D) = \Pi (sig_i + D_i) - 1.
        const real C = (sig + D).prod() - 1;
        // Compute grad C(D).
        Eigen::Matrix<real, dim, 1> grad_C = Eigen::Matrix<real, dim, 1>::Ones();
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                if (j == k) continue;
                grad_C(j) *= (sig(k) + D(k));
            }
        }
        const Eigen::Matrix<real, dim, 1> D_next = (grad_C.dot(D) - C) / grad_C.squaredNorm() * grad_C;
        const real diff = (D_next - D).cwiseAbs().maxCoeff();
        if (diff <= eps) break;
        // Update.
        D = D_next;
    }
    return U * (sig + D).asDiagonal() * V.transpose();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> VolumePdElementEnergy<dim>::ProjectToManifoldDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const Eigen::Matrix<real, dim, 1>& sig = F_auxiliary.sig();
    const Eigen::Matrix<real, dim, dim>& U = F_auxiliary.U();
    const Eigen::Matrix<real, dim, dim>& V = F_auxiliary.V();
    const Eigen::Matrix<real, dim, dim>& F = F_auxiliary.F();
    // F = U * sig * V.transpose();
    // Now solve the problem:
    // min \|sig - sig*\|^2
    // s.t. \Pi sig* = 1.
    // See the appendix of the 2014 PD paper for the derivation.
    // Let D = sig* - sig.
    // min \|D\|^2 s.t. \Pi (sig(i) + D(i)) = 1.

    const real eps = std::numeric_limits<real>::epsilon();
    CheckError(sig.minCoeff() >= eps, "Singular F.");

    // Initial guess.
    Eigen::Matrix<real, dim, 1> D = sig / std::pow(sig.prod(), ToReal(1) / dim) - sig;
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i) {
        // Compute C(D) = \Pi (sig_i + D_i) - 1.
        const real C = (sig + D).prod() - 1;
        // Compute grad C(D).
        Eigen::Matrix<real, dim, 1> grad_C = Eigen::Matrix<real, dim, 1>::Ones();
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                if (j == k) continue;
                grad_C(j) *= (sig(k) + D(k));
            }
        }
        const Eigen::Matrix<real, dim, 1> D_next = (grad_C.dot(D) - C) / grad_C.squaredNorm() * grad_C;
        const real diff = (D_next - D).cwiseAbs().maxCoeff();
        if (diff <= eps) break;
        // Update.
        D = D_next;
    }
    // Step 1: compute SVD gradients.
    Eigen::Matrix<real, dim, 1> grad_C = Eigen::Matrix<real, dim, 1>::Ones();
    for (int j = 0; j < dim; ++j) {
        for (int k = 0; k < dim; ++k) {
            if (j == k) continue;
            grad_C(j) *= (sig(k) + D(k));
        }
    }
    Eigen::Matrix<real, dim, dim> H = Eigen::Matrix<real, dim, dim>::Ones();
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            if (i > j) continue;
            else if (i == j) H(i, j) = 0;
            else {
                // Compute H(i, j).
                for (int k = 0; k < dim; ++k) {
                    if (k == i || k == j) continue;
                    H(i, j) *= (sig(k) + D(k));
                }
                H(j, i) = H(i, j);
            }
        }
    // Step 1: compute dSVD.
    Eigen::Matrix<real, dim, 1> dsig;
    Eigen::Matrix<real, dim, dim> dU, dV;
    dSvd(F, U, sig, V, dF, dU, dsig, dV);

    // Step 2: compute the derivatives of D.
    // min \|D\|^2 s.t. \Pi (sig(i) + D(i)) = 1.
    // KKT conditions:
    // D + la * grad C = 0.
    // C = 0.

    // Avoid dividing zero elements.
    const real la_abs = D.norm() / grad_C.norm();
    real la = la_abs;
    if ((D - la_abs * grad_C).squaredNorm() < (D + la_abs * grad_C).squaredNorm()) la = -la_abs;
    // dD + la * Hess C * dD + la * Hess C * dsig + grad C * dla = 0.
    // grad C * dD + grad C * dsig = 0.
    // [I + la * Hess C, grad C] * [dD ] = [-la * Hess C * dsig]
    // [grad C,               0]   [dla]   [-grad C * dsig]
    Eigen::Matrix<real, dim + 1, dim + 1> A; A.setZero();
    A.topLeftCorner(dim, dim) = Eigen::Matrix<real, dim, dim>::Identity() + la * H;
    A.topRightCorner(dim, 1) = grad_C;
    A.bottomLeftCorner(1, dim) = grad_C.transpose();
    Eigen::Matrix<real, dim + 1, 1> b;
    b.head(dim) = -la * H * dsig;
    b(dim) = -grad_C.dot(dsig);
    const Eigen::Matrix<real, dim + 1, 1> dDla = A.inverse() * b;
    const Eigen::Matrix<real, dim, 1> dD = dDla.head(dim);

    // Step 3: putting it together.
    // dU * (sig + D) * Vt + U * (dsig + dD) * Vt + U * (sig + D) * dVt.
    return dF + dU * D.asDiagonal() * V.transpose() + U * dD.asDiagonal() * V.transpose()
        + U * D.asDiagonal() * dV.transpose();
}

template class VolumePdElementEnergy<2>;
template class VolumePdElementEnergy<3>;