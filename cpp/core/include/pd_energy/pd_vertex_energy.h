#ifndef PD_ENERGY_PD_VERTEX_ENERGY_H
#define PD_ENERGY_PD_VERTEX_ENERGY_H

#include "common/config.h"

// Implements w_i / 2 \|q_i - proj(q)\|^2.
// where q_i is a single node.
// In my code, w_i = stiffness.
template<int dim>
class PdVertexEnergy {
public:
    PdVertexEnergy() {}
    virtual ~PdVertexEnergy() {}

    void Initialize(const real stiffness);

    const real stiffness() const { return stiffness_; }

    const real PotentialEnergy(const Eigen::Matrix<real, dim, 1>& q) const;
    const Eigen::Matrix<real, dim, 1> PotentialForce(const Eigen::Matrix<real, dim, 1>& q) const;
    const Eigen::Matrix<real, dim, 1> PotentialForceDifferential(const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& dq) const;
    const Eigen::Matrix<real, dim, dim> PotentialForceDifferential(
        const Eigen::Matrix<real, dim, 1>& q) const;

    virtual const Eigen::Matrix<real, dim, 1> ProjectToManifold(const Eigen::Matrix<real, dim, 1>& q) const = 0;
    virtual const Eigen::Matrix<real, dim, 1> ProjectToManifoldDifferential(const Eigen::Matrix<real, dim, 1>& q,
        const Eigen::Matrix<real, dim, 1>& dq) const = 0;
    virtual const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, dim, 1>& q) const;

private:
    real stiffness_;
};

#endif