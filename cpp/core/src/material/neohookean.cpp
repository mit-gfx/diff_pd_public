#include "material/neohookean.h"
#include "common/geometry.h"

template<int dim>
const real NeohookeanMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    const real I1 = F.squaredNorm();
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    return mu / 2 * (I1 - dim) - mu * log_j + la / 2 * log_j * log_j;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeohookeanMaterial<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    // Useful derivatives:
    // grad J = grad |F| = |F| * F^-T
    // grad log(J) = F^-T
    // grad mu / 2 * (I1 - dim) = mu / 2 * (F : F - dim) = mu * F
    // grad mu * log_J = mu * F^-T
    // grad la / 2 * log_j^2 = la * log_j * F^-T.
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv_T = F.inverse().transpose();
    return mu * (F - F_inv_T) + la * log_j * F_inv_T;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeohookeanMaterial<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv = F.inverse();
    const Eigen::Matrix<real, dim, dim> F_inv_T = F_inv.transpose();
    // F * F_inv = I.
    // dF * F_inv + F * dF_inv = 0.
    // dF_inv = -F_inv * dF * F_inv.
    const Eigen::Matrix<real, dim, dim> dF_inv = -F_inv * dF * F_inv;
    const Eigen::Matrix<real, dim, dim> dF_inv_T = dF_inv.transpose();
    const real dlog_j = (F_inv_T.array() * dF.array()).sum();
    return mu * (dF - dF_inv_T) + la * (log_j * dF_inv_T + dlog_j * F_inv_T);
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> NeohookeanMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> ret; ret.setZero();
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv = F.inverse();
    const Eigen::Matrix<real, dim, dim> F_inv_T = F_inv.transpose();
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            Eigen::Matrix<real, dim, dim> dF; dF.setZero(); dF(i, j) = 1;
            const Eigen::Matrix<real, dim, dim> dF_inv = -F_inv.col(i) * F_inv.row(j);
            const Eigen::Matrix<real, dim, dim> dF_inv_T = dF_inv.transpose();
            const real dlog_j = F_inv_T(i, j);
            const Eigen::Matrix<real, dim, dim> dP = mu * (dF - dF_inv_T) + la * (log_j * dF_inv_T + dlog_j * F_inv_T);
            const int idx = i + j * dim;
            ret.col(idx) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(dP.data(), dP.size());
        }
    return ret;
}

template<int dim>
const real NeohookeanMaterial<dim>::ComputeAverageStiffness(const real singular_value_range) const {
    // The energy density function is as follows:
    // mu / 2 * (I1 - dim) - mu * log(J) + la / 2 * log(J) * log(J).
    // F = U * S * Vt.
    // I1 = tr(F.T * F) = tr(V * S * S * Vt) = tr(S * S) = sum si^2.
    // J = |F| = \Pi * si
    // Psi = mu / 2 * I1 - mu / 2 * dim - mu * log(J) + la / 2 * log(J) * log(J)
    //     = mu * (si^2 / 2 - log(si)) + la / 2 * log(J) * log(J) - mu / 2 * dim.
    //
    // Now we compute the derivatives of Psi w.r.t. s1 at s2 = s3 = 1:
    // Psi' = mu * (s1 - 1 / s1) + la * log(s1) / s1
    const real x_start = 1 - singular_value_range;
    const real x_end = 1 + singular_value_range;
    //   \int (k(s1 - 1) - Psi'(s1))^2 ds1
    // = \int k^2(s1 - 1)^2 ds1 - 2\int k(s1 - 1) Psi'(s1) ds1
    // = k^2 [\int (s1 - 1)^2 ds1] - 2 k [\int (s1 - 1) Psi'(s1) ds1]
    // The best k = [\int (s1 - 1) Psi'(s1)] / [\int (s1 - 1)^2 ds1]
    const real k_below = ToReal(1.0) / 3 * (std::pow(x_end - 1, 3) - std::pow(x_start - 1, 3));
    // (s1 - 1) * Psi'(s1) = mu * s1^2 - mu + la * log(s1) - mu * s1 + mu / s1 - la * log(s1) / s1
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    // (x ln x)' = 1 + ln x.
    // (x ln x - x)' = ln x + 1 - 1 = ln x.
    const auto Mu = [](const real x) { return x * x * x / 3 - x - x * x / 2 + std::log(x); };
    const auto La = [](const real x) { return x * std::log(x) - x - 0.5 * std::pow(std::log(x), 2); };
    const real mu_coeff = (Mu(x_end) - Mu(x_start)) / k_below;
    const real la_coeff = (La(x_end) - La(x_start)) / k_below;
    return mu * mu_coeff + la * la_coeff;
}

template class NeohookeanMaterial<2>;
template class NeohookeanMaterial<3>;