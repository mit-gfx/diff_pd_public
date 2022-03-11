#include "friction/planar_frictional_boundary.h"
#include "common/common.h"

template<int dim>
PlanarFrictionalBoundary<dim>::PlanarFrictionalBoundary()
    : normal_(Eigen::Matrix<real, dim, 1>::UnitX()), offset_(0) {}

template<int dim>
void PlanarFrictionalBoundary<dim>::Initialize(const Eigen::Matrix<real, dim, 1>& normal, const real offset) {
    const real norm = normal.norm();
    CheckError(norm > std::numeric_limits<real>::epsilon(), "Singular normal.");
    normal_ = normal / norm;
    offset_ = offset / norm;
}

template<>
const Matrix2r PlanarFrictionalBoundary<2>::GetLocalFrame(const Vector2r& q) const {
    Matrix2r local;
    local.col(1) = normal_;
    local(0, 0) = normal_.y();
    local(1, 0) = -normal_.x();
    return local;
}

template<>
const Matrix3r PlanarFrictionalBoundary<3>::GetLocalFrame(const Vector3r& q) const {
    Matrix3r local;
    local.col(2) = normal_;
    Vector3r unit_x = Vector3r::Zero();
    for (int i = 0; i < 3; ++i) {
        const Vector3r x = normal_.cross(Vector3r::Unit(i));
        if (x.squaredNorm() > unit_x.squaredNorm()) unit_x = x;
    }
    unit_x /= unit_x.norm();
    local.col(0) = unit_x;
    Vector3r unit_y = normal_.cross(unit_x);
    unit_y /= unit_y.norm();
    local.col(1) = unit_y;
    return local;
}

template<int dim>
const real PlanarFrictionalBoundary<dim>::GetDistance(const Eigen::Matrix<real, dim, 1>& q) const {
    return q.dot(normal_) + offset_;
}

template<int dim>
const bool PlanarFrictionalBoundary<dim>::ForwardIntersect(const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v, const real dt, real& t_hit) const {
    const auto q_next = q + dt * v;
    // Check if q_next is below the plane.
    // If q_next is above the plane, it means the object is separating from the collision surface
    // and we should allow it regardless whether the current position is above or below the surface.
    const bool q_next_above = normal_.dot(q_next) + offset_ > 0;
    if (q_next_above) return false;
    // In all other cases, we compute the intersection. For singular cases when v is parallel to
    // the plane, we set t_hit = 0 and freeze the point in the current position.
    // normal_.dot(q + t_hit * v) + offset_ = 0.
    // normal_.dot(q) + t_hit * normal_.dot(v) + offset_ = 0.
    const real denom = normal_.dot(v);
    if (std::fabs(denom) > std::numeric_limits<real>::epsilon()) {
        const real inv_denom = 1 / denom;
        t_hit = -(normal_.dot(q) + offset_) * inv_denom;
    } else {
        t_hit = 0;
    }
    return true;
}

// q_hit = q + t_hit * v.
template<int dim>
void PlanarFrictionalBoundary<dim>::BackwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
    const real t_hit, const Eigen::Matrix<real, dim, 1>& dl_dq_hit, Eigen::Matrix<real, dim, 1>& dl_dq,
    Eigen::Matrix<real, dim, 1>& dl_dv) const {
    // q_hit = q + t_hit * v.
    // normal_.dot(q) + t_hit * normal_.dot(v) + offset_ = 0.
    // t_hit = -(normal_.dot(q) + offset_) / normal_.dot(v).
    // q_hit = q - (normal_.dot(q) + offset_) / normal_.dot(v) * v.
    const real denom = normal_.dot(v);
    if (std::fabs(denom) > std::numeric_limits<real>::epsilon()) {
        const real inv_denom = 1 / denom;
        const Eigen::Matrix<real, dim, 1> dt_hit_dq = -inv_denom * normal_;
        const Eigen::Matrix<real, dim, 1> dt_hit_dv = (normal_.dot(q) + offset_) * inv_denom * inv_denom * normal_;
        // q_hit = q + t_hit * v.
        // Jq = I + v * dt_hit_dq.transpose().
        // Jv = v * dt_hit_dv.transpose() + t_hit * I.
        dl_dq = dl_dq_hit + v.dot(dl_dq_hit) * dt_hit_dq;
        dl_dv = v.dot(dl_dq_hit) * dt_hit_dv + t_hit * dl_dq_hit;
    } else {
        // t_hit = 0.
        // q_hit = q.
        // Jq = I.
        // Jv = 0.
        dl_dq = dl_dq_hit;
        dl_dv = Eigen::Matrix<real, dim, 1>::Zero();
    }
}

template class PlanarFrictionalBoundary<2>;
template class PlanarFrictionalBoundary<3>;