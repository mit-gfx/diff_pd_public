#ifndef FRICTION_PLANAR_FRICTIONAL_BOUNDARY_H
#define FRICTION_PLANAR_FRICTIONAL_BOUNDARY_H

#include "friction/frictional_boundary.h"

// Obstacle: normal.dot(q) + offset <= 0. In other words, the deformable body almost always stays on the positive side.
template<int dim>
class PlanarFrictionalBoundary : public FrictionalBoundary<dim> {
public:
    PlanarFrictionalBoundary();

    void Initialize(const Eigen::Matrix<real, dim, 1>& normal, const real offset);

    const Eigen::Matrix<real, dim, dim> GetLocalFrame(const Eigen::Matrix<real, dim, 1>& q) const;
    const real GetDistance(const Eigen::Matrix<real, dim, 1>& q) const;
    const bool ForwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit) const override;
    // q_hit = q + t_hit * v.
    void BackwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit, const Eigen::Matrix<real, dim, 1>& dl_dq_hit, Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv) const override;

private:
    Eigen::Matrix<real, dim, 1> normal_;
    real offset_;
};

#endif