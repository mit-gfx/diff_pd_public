#include "friction/spherical_frictional_boundary.h"
#include "common/common.h"

template<int dim>
SphericalFrictionalBoundary<dim>::SphericalFrictionalBoundary()
    : center_(Eigen::Matrix<real, dim, 1>::Zero()), radius_(0) {}

template<int dim>
void SphericalFrictionalBoundary<dim>::Initialize(const Eigen::Matrix<real, dim, 1>& center, const real radius) {
    center_ = center;
    radius_ = radius;
}

template<>
const Matrix2r SphericalFrictionalBoundary<2>::GetLocalFrame(const Vector2r& q) const {
    Matrix2r local;
    local.col(1) = (q - center_) / (q - center_).norm();
    local(0, 0) = local(1, 1);
    local(1, 0) = -local(0, 1);
    return local;
}

template<>
const Matrix3r SphericalFrictionalBoundary<3>::GetLocalFrame(const Vector3r& q) const {
    Matrix3r local;
    const Vector3r normal = (q - center_) / (q - center_).norm();
    local.col(2) = normal;
    Vector3r unit_x = Vector3r::Zero();
    for (int i = 0; i < 3; ++i) {
        const Vector3r x = normal.cross(Vector3r::Unit(i));
        if (x.squaredNorm() > unit_x.squaredNorm()) unit_x = x;
    }
    unit_x /= unit_x.norm();
    local.col(0) = unit_x;
    Vector3r unit_y = normal.cross(unit_x);
    unit_y /= unit_y.norm();
    local.col(1) = unit_y;
    return local;
}

template<int dim>
const real SphericalFrictionalBoundary<dim>::GetDistance(const Eigen::Matrix<real, dim, 1>& q) const {
    return (q - center_).norm() - radius_;
}

template<int dim>
const bool SphericalFrictionalBoundary<dim>::ForwardIntersect(const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v, const real dt, real& t_hit) const {
    const auto q_next = q + dt * v;
    // Check if q_next is below the plane.
    // If q_next is above the plane, it means the object is separating from the collision surface
    // and we should allow it regardless whether the current position is above or below the surface.
    const bool q_next_above = GetDistance(q_next) > 0;
    if (q_next_above) return false;
    // In all other cases, we compute the intersection.
    // |q + t_hit * v - center| = radius.
    // (t_hit * v + (q - center))^2 - radius^2 = 0.
    // t_hit^2 * (v^2) + (q - center)^2 + 2 * v * (q - center) * t_hit - radius^2 = 0.
    const real a = v.dot(v);
    const real b = 2 * v.dot(q - center_);
    const real c = (q - center_).dot(q - center_) - radius_ * radius_;
    const real delta = b * b - 4 * a * c;
    if (delta < 0) {
        // Singular case.
        t_hit = 0;
    } else {
        const real delta_sqrt = std::sqrt(delta);
        t_hit = (-b - delta_sqrt) / (2 * a);
    }
    return true;
}

// q_hit = q + t_hit * v.
template<int dim>
void SphericalFrictionalBoundary<dim>::BackwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
    const real t_hit, const Eigen::Matrix<real, dim, 1>& dl_dq_hit, Eigen::Matrix<real, dim, 1>& dl_dq,
    Eigen::Matrix<real, dim, 1>& dl_dv) const {
    // TODO.
    const real a = v.dot(v);
    const real b = 2 * v.dot(q - center_);
    const real c = (q - center_).dot(q - center_) - radius_ * radius_;
    const real delta = b * b - 4 * a * c;
    if (delta < 0) {
        // t_hit = 0.
        // q_hit = q.
        // Jq = I.
        // Jv = 0.
        dl_dq = dl_dq_hit;
        dl_dv = Eigen::Matrix<real, dim, 1>::Zero();
    } else {
        const real delta_sqrt = std::sqrt(delta);
        // t_hit = (-b - delta_sqrt) / (2 * a);
        const Eigen::Matrix<real, dim, 1> da_dv = 2 * v;
        const Eigen::Matrix<real, dim, 1> db_dq = 2 * v;
        const Eigen::Matrix<real, dim, 1> db_dv = 2 * (q - center_);
        const Eigen::Matrix<real, dim, 1> dc_dq = 2 * (q - center_);
        const Eigen::Matrix<real, dim, 1> ddelta_dq = 2 * b * db_dq - 4 * a * dc_dq;
        const Eigen::Matrix<real, dim, 1> ddelta_dv = 2 * b * db_dv - 4 * da_dv * c;
        const Eigen::Matrix<real, dim, 1> dsqrt_dq = 0.5 / delta_sqrt * ddelta_dq;
        const Eigen::Matrix<real, dim, 1> dsqrt_dv = 0.5 / delta_sqrt * ddelta_dv;
        // (f/g)' = f'/g - fg'/g^2 = (f'g - fg')/g^2.
        const real f = -b - delta_sqrt;
        const real g = 2 * a;
        const Eigen::Matrix<real, dim, 1> df_dq = -db_dq - dsqrt_dq;
        const Eigen::Matrix<real, dim, 1> df_dv = -db_dv - dsqrt_dv;
        const Eigen::Matrix<real, dim, 1> dg_dv = 2 * da_dv;
        const Eigen::Matrix<real, dim, 1> dt_hit_dq = df_dq / g;
        const Eigen::Matrix<real, dim, 1> dt_hit_dv = (df_dv * g - f * dg_dv) / (g * g);
        // q_hit = q + t_hit * v.
        // Jq = I + v * dt_hit_dq.transpose().
        // Jv = v * dt_hit_dv.transpose() + t_hit * I.
        dl_dq = dl_dq_hit + v.dot(dl_dq_hit) * dt_hit_dq;
        dl_dv = v.dot(dl_dq_hit) * dt_hit_dv + t_hit * dl_dq_hit;
    }
}

template class SphericalFrictionalBoundary<2>;
template class SphericalFrictionalBoundary<3>;