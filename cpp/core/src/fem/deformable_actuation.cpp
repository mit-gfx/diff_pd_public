#include "fem/deformable.h"
#include "common/geometry.h"
#include "pd_energy/pd_muscle_energy.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AddActuation(const real stiffness, const std::array<real, vertex_dim>& fiber_direction,
    const std::vector<int>& indices) {
    auto energy = std::make_shared<PdMuscleEnergy<vertex_dim>>();
    Eigen::Matrix<real, vertex_dim, 1> m;
    for (int i = 0; i < vertex_dim; ++i) m(i) = fiber_direction[i];
    energy->Initialize(stiffness, m);

    // Check muscles.
    std::set<int> unique_idx;
    std::vector<int> element_idx;
    const int element_num = mesh_.NumOfElements();
    for (const int idx : indices) {
        CheckError(0 <= idx && idx < element_num, "Element index out of bound.");
        CheckError(unique_idx.find(idx) == unique_idx.end(), "Duplicated elements.");
        element_idx.push_back(idx);
        unique_idx.insert(idx);
    }

    pd_muscle_energies_.push_back(std::make_pair(energy, element_idx));
    act_dofs_ += static_cast<int>(element_idx.size());
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ActuationEnergy(const VectorXr& q, const VectorXr& a) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = GetNumOfSamplesInElement();
    real total_energy = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int element_idx : pair.second) {
            const auto qi = ScatterToElement(q, element_idx);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(element_idx, qi, j);
                total_energy += energy->EnergyDensity(F, a(act_idx)) * element_volume_ / sample_num;
            }
            ++act_idx;
        }
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    return total_energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ActuationForce(const VectorXr& q, const VectorXr& a) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = GetNumOfSamplesInElement();
    VectorXr f = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int element_idx : pair.second) {
            const auto vi = mesh_.element(element_idx);
            const auto deformed = ScatterToElement(q, element_idx);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(element_idx, deformed, j);
                const auto P = energy->StressTensor(F, a(act_idx));
                const Eigen::Matrix<real, 1, element_dim * vertex_dim> f_kd =
                    -Flatten(P).transpose() * finite_element_samples_[element_idx][j].dF_dxkd_flattened()
                        * element_volume_ / sample_num;

                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        f(vertex_dim * vi(k) + d) += f_kd(k * vertex_dim + d);
            }
            ++act_idx;
        }
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    return f;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
    const VectorXr& dq, const VectorXr& da, const VectorXr& dw) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()) && a.size() == da.size(), "Inconsistent actuation size.");
    CheckError(static_cast<int>(dw.size()) == NumOfPdMuscleEnergies(), "Inconsistent dw size.");
    int act_idx = 0;
    int muscle_idx = 0;
    const int sample_num = GetNumOfSamplesInElement();
    VectorXr df = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const real inv_w = 1 / energy->stiffness();
        for (const int element_idx : pair.second) {
            const auto vi = mesh_.element(element_idx);
            const auto deformed = ScatterToElement(q, element_idx);
            const auto ddeformed = ScatterToElement(dq, element_idx);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(element_idx, deformed, j);
                const auto dF = DeformationGradient(element_idx, ddeformed, j);
                // We use the assumption that stress tensor is linear to w.
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dP =
                    energy->StressTensorDifferential(F, a(act_idx), dF, da(act_idx)) +
                    energy->StressTensor(F, a(act_idx)) * inv_w * dw(muscle_idx);
                const Eigen::Matrix<real, 1, element_dim * vertex_dim> df_kd =
                    -Flatten(dP).transpose() * finite_element_samples_[element_idx][j].dF_dxkd_flattened()
                        * element_volume_ / sample_num;

                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        df(vertex_dim * vi(k) + d) += df_kd(k * vertex_dim + d);
            }
            ++act_idx;
        }
        ++muscle_idx;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    CheckError(muscle_idx == NumOfPdMuscleEnergies(), "You loop over muscles has introduced a bug.");
    return df;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
    SparseMatrixElements& dq, SparseMatrixElements& da, SparseMatrixElements& dw) const {
    dq.clear();
    da.clear();
    dw.clear();
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    int muscle_idx = 0;
    const int sample_num = GetNumOfSamplesInElement();

    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const real inv_w = 1 / energy->stiffness();
        for (const int i : pair.second) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElement(q, i);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(i, deformed, j);
                MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
                for (int s = 0; s < element_dim; ++s)
                    for (int t = 0; t < vertex_dim; ++t) {
                        const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                            Eigen::Matrix<real, vertex_dim, 1>::Unit(t)
                                * finite_element_samples_[i][j].grad_undeformed_sample_weight().row(s);
                        dF.col(s * vertex_dim + t) += Flatten(dF_single);
                }
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dPdF;
                Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dPda;
                energy->StressTensorDifferential(F, a(act_idx), dPdF, dPda);
                const auto dPF = dPdF * dF;
                const auto dPa = dPda * 1;
                Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dPdw = Flatten(energy->StressTensor(F, a(act_idx))) * inv_w;
                const auto dPw = dPdw * 1;

                const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kdF =
                    -dPF.transpose() * finite_element_samples_[i][j].dF_dxkd_flattened()
                        * element_volume_ / sample_num;
                const Eigen::Matrix<real, 1, element_dim * vertex_dim> df_kda = -dPa.transpose()
                    * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
                const Eigen::Matrix<real, 1, element_dim * vertex_dim> df_kdw = -dPw.transpose()
                    * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d) {
                        // State.
                        for (int s = 0; s < element_dim; ++s)
                            for (int t = 0; t < vertex_dim; ++t)
                                dq.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                    vertex_dim * vi(s) + t, df_kdF(s * vertex_dim + t, k * vertex_dim + d)));
                        // Action.
                        da.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d, act_idx, df_kda(k * vertex_dim + d)));
                        // Stiffness.
                        dw.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d, muscle_idx, df_kdw(k * vertex_dim + d)));
                    }
            }
            ++act_idx;
        }
        ++muscle_idx;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    CheckError(muscle_idx == NumOfPdMuscleEnergies(), "Your loop over muscles has introduced a bug.");
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyActuationEnergy(const std::vector<real>& q, const std::vector<real>& a) const {
    return ActuationEnergy(ToEigenVector(q), ToEigenVector(a));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyActuationForce(const std::vector<real>& q,
    const std::vector<real>& a) const {
    return ToStdVector(ActuationForce(ToEigenVector(q), ToEigenVector(a)));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyActuationForceDifferential(const std::vector<real>& q,
    const std::vector<real>& a, const std::vector<real>& dq, const std::vector<real>& da, const std::vector<real>& dw) const {
    return ToStdVector(ActuationForceDifferential(ToEigenVector(q), ToEigenVector(a), ToEigenVector(dq),
        ToEigenVector(da), ToEigenVector(dw)));
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyActuationForceDifferential(const std::vector<real>& q, const std::vector<real>& a,
    std::vector<std::vector<real>>& dq, std::vector<std::vector<real>>& da, std::vector<std::vector<real>>& dw) const {
    PrintWarning("PyActuationForceDifferential should only be used for small-scale problems and for testing purposes.");
    SparseMatrixElements nonzeros_dq, nonzeros_da, nonzeros_dw;
    ActuationForceDifferential(ToEigenVector(q), ToEigenVector(a), nonzeros_dq, nonzeros_da, nonzeros_dw);
    dq.resize(dofs_);
    da.resize(dofs_);
    dw.resize(dofs_);
    const int act_w_dofs = NumOfPdMuscleEnergies();
    for (int i = 0; i < dofs_; ++i) {
        dq[i].resize(dofs_);
        std::fill(dq[i].begin(), dq[i].end(), 0);
        da[i].resize(act_dofs_);
        std::fill(da[i].begin(), da[i].end(), 0);
        dw[i].resize(act_w_dofs);
        std::fill(dw[i].begin(), dw[i].end(), 0);
    }
    for (const auto& triplet: nonzeros_dq) dq[triplet.row()][triplet.col()] += triplet.value();
    for (const auto& triplet: nonzeros_da) da[triplet.row()][triplet.col()] += triplet.value();
    for (const auto& triplet: nonzeros_dw) dw[triplet.row()][triplet.col()] += triplet.value();
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;