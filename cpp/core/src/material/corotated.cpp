#include "material/corotated.h"
#include "common/geometry.h"

template<int dim>
const real CorotatedMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const real trace_e_c = e_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * e_c.array().square().sum() + la / 2 * (trace_e_c * trace_e_c);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedMaterial<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const real trace_e_c = e_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return R * (2 * mu * e_c + la * trace_e_c * I);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedMaterial<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> dR = dRFromdF(F, R, S, dF);
    const Eigen::Matrix<real, dim, dim> dS = R.transpose() * (dF - dR * S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const Eigen::Matrix<real, dim, dim> de_c = dS;
    const real trace_e_c = e_c.trace();
    const real dtrace_e_c = de_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return dR * (2 * mu * e_c + la * trace_e_c * I)
        + R * (2 * mu * de_c + la * dtrace_e_c * I);
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> CorotatedMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const auto dR_flattened = dRFromdF(F, R, S);
    Eigen::Matrix<real, dim * dim, dim * dim> ret; ret.setZero();
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const int idx = i + j * dim;
            Eigen::Matrix<real, dim, dim> dF; dF.setZero();
            dF(i, j) = 1;
            const Eigen::Matrix<real, dim, dim> dR = Eigen::Map<const Eigen::Matrix<real, dim, dim>>(
                dR_flattened.col(idx).data(), dim, dim
            );
            const Eigen::Matrix<real, dim, dim> dS = R.transpose() * (dF - dR * S);
            const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
            const Eigen::Matrix<real, dim, dim> e_c = S - I;
            const Eigen::Matrix<real, dim, dim> de_c = dS;
            const real trace_e_c = e_c.trace();
            const real dtrace_e_c = de_c.trace();
            const real mu = Material<dim>::mu();
            const real la = Material<dim>::lambda();
            const Eigen::Matrix<real, dim, dim> dP = dR * (2 * mu * e_c + la * trace_e_c * I)
                + R * (2 * mu * de_c + la * dtrace_e_c * I);
            ret.col(idx) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(dP.data(), dP.size());
        }
    return ret;
}

template<int dim>
const real CorotatedMaterial<dim>::ComputeAverageStiffness(const real singular_value_range) const {
    CheckError(false, "Not implemented yet.");
    return 0;
}

template class CorotatedMaterial<2>;
template class CorotatedMaterial<3>;