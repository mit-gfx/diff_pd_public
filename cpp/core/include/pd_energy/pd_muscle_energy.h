#ifndef PD_ENERGY_PD_MUSCLE_ENERGY_H
#define PD_ENERGY_PD_MUSCLE_ENERGY_H

#include "common/config.h"

// This class implements the Contractile fiber in the SoftCon paper.
// Note that I reverse the definition of e: I find it easier to write code if I use r = e / l.
template<int dim>
class PdMuscleEnergy {
public:
    void Initialize(const real stiffness, const Eigen::Matrix<real, dim, 1>& fiber_direction);

    const real stiffness() const { return stiffness_; }
    const Eigen::Matrix<real, dim, 1>& fiber_direction() const { return fiber_direction_; }
    const SparseMatrix MtM() const { return MtM_; }
    const SparseMatrix Mt() const { return Mt_; }

    const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F, const real activation_level) const;
    const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F,
        const real activation_level) const;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const real activation_level, const Eigen::Matrix<real, dim, dim>& dF, const real dactivation_level) const;
    void StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F, const real activation_level,
        Eigen::Matrix<real, dim * dim, dim * dim>& dF, Eigen::Matrix<real, dim * dim, 1>& dactivation_level) const;

    // APIs needed by projective dynamics.
    const Eigen::Matrix<real, dim, 1> ProjectToManifold(const Eigen::Matrix<real, dim, dim>& F,
        const real activation_level) const;
    const Eigen::Matrix<real, dim, 1> ProjectToManifoldDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const real activation_level, const Eigen::Matrix<real, dim, dim>& dF, const real dactivation_level) const;
    void ProjectToManifoldDifferential(const Eigen::Matrix<real, dim, dim>& F, const real activation_level,
        Eigen::Matrix<real, dim, dim * dim>& dF, Eigen::Matrix<real, dim, 1>& dactivation_level) const;

private:
    real stiffness_;
    Eigen::Matrix<real, dim, 1> fiber_direction_;
    Eigen::Matrix<real, dim, dim> mmt_;
    // M * Flatten(F) = Fm.
    // MtM = M' * M.
    SparseMatrix Mt_;
    SparseMatrix MtM_;
};

#endif