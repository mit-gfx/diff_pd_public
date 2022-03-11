#include "state_force/hydrodynamics_state_force.h"
#include "common/common.h"
#include "common/geometry.h"
#include "mesh/mesh.h"

template<int vertex_dim, int element_dim>
HydrodynamicsStateForce<vertex_dim, element_dim>::HydrodynamicsStateForce()
    : rho_(0), v_water_(Eigen::Matrix<real, vertex_dim, 1>::Zero()), max_force_(0), surface_face_num_(0) {}

template<int vertex_dim, int element_dim>
void HydrodynamicsStateForce<vertex_dim, element_dim>::Initialize(const real rho, const Eigen::Matrix<real, vertex_dim, 1>& v_water,
    const Eigen::Matrix<real, 4, 2>& Cd_points, const Eigen::Matrix<real, 4, 2>& Ct_points, const real max_force,
    const MatrixXi& surface_faces) {
    rho_ = rho;
    for (int i = 0; i < vertex_dim; ++i) v_water_(i) = v_water(i);
    VectorXr parameters = VectorXr::Zero(16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) {
            parameters(i * 2 + j) = Cd_points(i, j);
            parameters(8 + i * 2 + j) = Ct_points(i, j);
        }
    StateForce<vertex_dim>::set_parameters(parameters);

    max_force_ = max_force;

    surface_faces_ = surface_faces;
    surface_face_num_ = static_cast<int>(surface_faces_.cols());
    CheckError(static_cast<int>(surface_faces_.rows()) == Mesh<vertex_dim, element_dim>::GetNumOfVerticesInFace(),
        "Inconsistent surface dimension.");

    // Validate the Cd_points.
    /*
    CheckError(Cd_points(0, 0) == 0 && Cd_points(1, 0) > Cd_points(0, 0)
        && Cd_points(2, 0) > Cd_points(1, 0) && Cd_points(3, 0) > Cd_points(2, 0)
        && Cd_points(3, 0) == 1 && Cd_points(0, 1) == Cd_points(1, 1), "Cd_points have unexpected parameters.");
    // Validate the Ct_points.
    CheckError(Ct_points(0, 0) == -1 && Ct_points(1, 0) > Ct_points(0, 0)
        && Ct_points(2, 0) > Ct_points(1, 0) && Ct_points(3, 0) > Ct_points(2, 0)
        && Ct_points(3, 0) == 1, "Ct_points have unexpected parameters.");
    */
}

template<int vertex_dim, int element_dim>
void HydrodynamicsStateForce<vertex_dim, element_dim>::PyInitialize(const real rho, const std::array<real, vertex_dim>& v_water,
    const std::vector<real>& Cd_points, const std::vector<real>& Ct_points, const real max_force,
    const std::vector<int>& surface_faces) {
    rho_ = rho;
    for (int i = 0; i < vertex_dim; ++i) v_water_(i) = v_water[i];
    CheckError(static_cast<int>(Cd_points.size()) == 8, "Expect 8 numbers in Cd_points.");
    CheckError(static_cast<int>(Ct_points.size()) == 8, "Expect 8 numbers in Ct_points.");
    const int surface_faces_size = static_cast<int>(surface_faces.size());
    const int face_dim = Mesh<vertex_dim, element_dim>::GetNumOfVerticesInFace();
    CheckError(surface_faces_size >= face_dim && surface_faces_size % face_dim == 0, "Surface face size is incompatible with face dim.");
    VectorXr parameters = VectorXr::Zero(16);
    for (int i = 0; i < 8; ++i) {
        parameters(i) = Cd_points[i];
        parameters(8 + i) = Ct_points[i];
    }
    StateForce<vertex_dim>::set_parameters(parameters);

    max_force_ = max_force;

    surface_faces_ = MatrixXi::Zero(face_dim, surface_faces_size / face_dim);
    for (int i = 0; i < surface_faces_size / face_dim; ++i) {
        for (int j = 0; j < face_dim; ++j) {
            surface_faces_(j, i) = surface_faces[i * face_dim + j];
        }
    }
    surface_face_num_ = static_cast<int>(surface_faces_.cols());

    // Validate the Cd_points.
    /*
    CheckError(Cd_points[0] == 0 && Cd_points[2] > Cd_points[0]
        && Cd_points[4] > Cd_points[2] && Cd_points[6] > Cd_points[4]
        && Cd_points[6] == 1 && Cd_points[1] == Cd_points[3], "Cd_points have unexpected results.");
    // Validate the Ct_points.
    CheckError(Ct_points[0] == -1 && Ct_points[2] > Ct_points[0]
        && Ct_points[4] > Ct_points[2] && Ct_points[6] > Ct_points[4]
        && Ct_points[6] == 1, "Ct_points have unexpected parameters.");
    */
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, 4, 2> HydrodynamicsStateForce<vertex_dim, element_dim>::Cd_points() const {
    Eigen::Matrix<real, 4, 2> cd_points;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            cd_points(i, j) = StateForce<vertex_dim>::parameters()(i * 2 + j);
    return cd_points;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, 4, 2> HydrodynamicsStateForce<vertex_dim, element_dim>::Ct_points() const {
    Eigen::Matrix<real, 4, 2> ct_points;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            ct_points(i, j) = StateForce<vertex_dim>::parameters()(8 + i * 2 + j);
    return ct_points;
}

template<int vertex_dim, int element_dim>
const real HydrodynamicsStateForce<vertex_dim, element_dim>::Cd(const real angle) const {
    const real angle_normalized = angle / (Pi() / 2);
    // angle_normalized now becomes [-1, 1].
    real t = angle_normalized;
    if (angle_normalized < 0) t = -t;
    const auto cd_points = Cd_points();
    return (1 - t) * (1 - t) * (1 - t) * cd_points(0, 1) + 3 * (1 - t) * (1 - t) * t * cd_points(1, 1)
        + 3 * (1 - t) * t * t * cd_points(2, 1) + t * t * t * cd_points(3, 1);
}

template<int vertex_dim, int element_dim>
void HydrodynamicsStateForce<vertex_dim, element_dim>::CdGradients(const real angle,
    real& grad_angle, VectorXr& grad_parameters) const {
    const real angle_normalized = angle / (Pi() / 2);
    // angle_normalized now becomes [-1, 1].
    real t = angle_normalized;
    real dt = 2 / Pi();
    if (angle_normalized < 0) {
        t = -t;
        dt = -dt;
    }
    const auto cd_points = Cd_points();
    grad_angle = 3 * dt * (-(1 - t) * (1 - t) * cd_points(0, 1)
        + (-2 * (1 - t) * t + (1 - t) * (1 - t)) * cd_points(1, 1)
        + (-t * t + (1 - t) * 2 * t) * cd_points(2, 1)
        + t * t * cd_points(3, 1));
    grad_parameters = VectorXr::Zero(16);
    grad_parameters(1) = (1 - t) * (1 - t) * (1 - t);
    grad_parameters(3) = 3 * (1 - t) * (1 - t) * t;
    grad_parameters(5) = 3 * (1 - t) * t * t;
    grad_parameters(7) = t * t * t;
}

template<int vertex_dim, int element_dim>
const real HydrodynamicsStateForce<vertex_dim, element_dim>::Ct(const real angle) const {
    const real t = angle / (Pi() / 2) * 0.5 + 0.5;
    // Now t is between 0 and 1.
    const auto ct_points = Ct_points();
    return (1 - t) * (1 - t) * (1 - t) * ct_points(0, 1) + 3 * (1 - t) * (1 - t) * t * ct_points(1, 1)
        + 3 * (1 - t) * t * t * ct_points(2, 1) + t * t * t * ct_points(3, 1);
}

template<int vertex_dim, int element_dim>
void HydrodynamicsStateForce<vertex_dim, element_dim>::CtGradients(const real angle,
    real& grad_angle, VectorXr& grad_parameters) const {
    grad_angle = 0;
    grad_parameters = VectorXr::Zero(16);
    const real t = angle / (Pi() / 2) * 0.5 + 0.5;
    const real dt = 1 / Pi();
    const auto ct_points = Ct_points();
    grad_angle = 3 * dt * (-(1 - t) * (1 - t) * ct_points(0, 1)
        + (-2 * (1 - t) * t + (1 - t) * (1 - t)) * ct_points(1, 1)
        + (-t * t + (1 - t) * 2 * t) * ct_points(2, 1)
        + t * t * ct_points(3, 1));
    grad_parameters(8 + 1) = (1 - t) * (1 - t) * (1 - t);
    grad_parameters(8 + 3) = 3 * (1 - t) * (1 - t) * t;
    grad_parameters(8 + 5) = 3 * (1 - t) * t * t;
    grad_parameters(8 + 7) = t * t * t;
}

// Although the implementation for <2, 3> and <2, 4> are identical, we intentionally duplicate them in case
// that we want to implement specific versions for triangle meshes and quad meshes.
template<>
const VectorXr HydrodynamicsStateForce<2, 3>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr f = VectorXr::Zero(q.size());
    for (int i = 0; i < surface_face_num_; ++i) {
        const int i0 = surface_faces_(0, i);
        const int i1 = surface_faces_(1, i);
        const Vector2r p0 = q.segment(2 * i0, 2);
        const Vector2r p1 = q.segment(2 * i1, 2);
        const Vector2r v0 = v.segment(2 * i0, 2);
        const Vector2r v1 = v.segment(2 * i1, 2);
        // Compute A.
        const real A = (p0 - p1).norm();
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector2r v_rel = v_water_ - 0.5 * (v0 + v1);
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector2r d = v_rel / v_rel_len;
        // Compute n.
        // In 2D, we assume that the contour of the mesh is ccw.
        const Vector2r An = Vector2r(p1.y() - p0.y(), -p1.x() + p0.x());
        const Vector2r n = An / A;
        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real phi = std::acos(nd) - Pi() / 2;
        const Vector2r f_drag = 0.5 * rho_ * A * Cd(phi) * v_rel_len * v_rel;
        const Vector2r f_thrust = -0.5 * rho_ * Ct(phi) * v_rel_len * v_rel_len * An;
        // Added f_drag and f_thrust back to i0 and i1.
        const Vector2r f_node = (f_drag + f_thrust) * 0.5;
        f.segment(2 * i0, 2) += f_node;
        f.segment(2 * i1, 2) += f_node;
    }

    // Clamp.
    f = f.cwiseMax(-max_force_).cwiseMin(max_force_);

    return f;
}

template<>
void HydrodynamicsStateForce<2, 3>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    VectorXr dl_df_before_clamp = dl_df;
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        if (f(i) == max_force_ || f(i) == -max_force_) {
            dl_df_before_clamp(i) = 0;
        }
    }

    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(16);
    for (int i = 0; i < surface_face_num_; ++i) {
        const int i0 = surface_faces_(0, i);
        const int i1 = surface_faces_(1, i);
        const Vector2r p0 = q.segment(2 * i0, 2);
        const Vector2r p1 = q.segment(2 * i1, 2);
        const Vector2r v0 = v.segment(2 * i0, 2);
        const Vector2r v1 = v.segment(2 * i1, 2);
        // Compute A.
        const real A = (p0 - p1).norm();
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector2r v_rel = v_water_ - 0.5 * (v0 + v1);
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector2r d = v_rel / v_rel_len;

        // Compute derivatives of A and d.
        const Vector2r dA_dp0 = (p0 - p1) / A;
        const Vector2r dA_dp1 = -dA_dp0;
        const Matrix2r I = Matrix2r::Identity();
        const Matrix2r dd_dv0 = -0.5 * (I - d * d.transpose()) / v_rel_len;
        const Matrix2r dd_dv1 = dd_dv0;

        // Compute n.
        // In 2D, we assume that the contour of the mesh is ccw.
        const Vector2r An = Vector2r(p1.y() - p0.y(), -p1.x() + p0.x());
        const Vector2r n = An / A;

        // Compute derivatives of An and n.
        Matrix2r dAn_dp0;
        dAn_dp0 << 0, -1,
                    1, 0;
        const Matrix2r dAn_dp1 = -dAn_dp0;
        const Matrix2r dn_dp0 = (I - n * n.transpose()) / A * dAn_dp0;
        const Matrix2r dn_dp1 = -dn_dp0;

        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real nd2 = nd * nd;
        const real phi = std::acos(nd) - Pi() / 2;
        const real dacos = -1 / std::sqrt(1 - nd2);
        const RowVector2r dphi_dp0 = dacos * d.transpose() * dn_dp0;
        const RowVector2r dphi_dp1 = dacos * d.transpose() * dn_dp1;
        const RowVector2r dphi_dv0 = dacos * n.transpose() * dd_dv0;
        const RowVector2r dphi_dv1 = dacos * n.transpose() * dd_dv1;

        const real Cd_phi = Cd(phi);
        real Cd_grad_phi = 0;
        VectorXr Cd_grad_parameters = VectorXr::Zero(16);
        CdGradients(phi, Cd_grad_phi, Cd_grad_parameters);
        const Matrix2r df_drag_dp0 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp0.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp0);
        const Matrix2r df_drag_dp1 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp1.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp1);
        const Matrix2r df_drag_dv0 = 0.5 * rho_ * A * (-0.5 * I * Cd_phi * v_rel_len + v_rel * Cd_grad_phi * dphi_dv0 * v_rel_len
            -0.5 * v_rel * Cd_phi * d.transpose());
        const Matrix2r df_drag_dv1 = 0.5 * rho_ * A * (-0.5 * I * Cd_phi * v_rel_len + v_rel * Cd_grad_phi * dphi_dv1 * v_rel_len
            -0.5 * v_rel * Cd_phi * d.transpose());
        const Eigen::Matrix<real, 2, 16> df_drag_dp = 0.5 * rho_ * A * v_rel_len * v_rel * Cd_grad_parameters.transpose();

        const real Ct_phi = Ct(phi);
        real Ct_grad_phi = 0;
        VectorXr Ct_grad_parameters = VectorXr::Zero(16);
        CtGradients(phi, Ct_grad_phi, Ct_grad_parameters);
        const Matrix2r df_thrust_dp0 = -0.5 * rho_ * v_rel_len * v_rel_len * (
            An * Ct_grad_phi * dphi_dp0 + Ct_phi * dAn_dp0);
        const Matrix2r df_thrust_dp1 = -0.5 * rho_ * v_rel_len * v_rel_len * (
            An * Ct_grad_phi * dphi_dp1 + Ct_phi * dAn_dp1);
        const Matrix2r df_thrust_dv0 = -0.5 * rho_ * An * (
            Ct_grad_phi * dphi_dv0 * v_rel_len * v_rel_len - Ct_phi * v_rel.transpose());
        const Matrix2r df_thrust_dv1 = -0.5 * rho_ * An * (
            Ct_grad_phi * dphi_dv1 * v_rel_len * v_rel_len - Ct_phi * v_rel.transpose());
        const Eigen::Matrix<real, 2, 16> df_thrust_dp = -0.5 * rho_ * v_rel_len * v_rel_len * An * Ct_grad_parameters.transpose();

        // Added f_drag and f_thrust back to i0 and i1.
        const Matrix2r df_node_dp0 = (df_drag_dp0 + df_thrust_dp0) * 0.5;
        const Matrix2r df_node_dp1 = (df_drag_dp1 + df_thrust_dp1) * 0.5;
        const Matrix2r df_node_dv0 = (df_drag_dv0 + df_thrust_dv0) * 0.5;
        const Matrix2r df_node_dv1 = (df_drag_dv1 + df_thrust_dv1) * 0.5;
        const Eigen::Matrix<real, 2, 16> df_node_dp = (df_drag_dp + df_thrust_dp) * 0.5;

        for (int ii : { i0, i1 }) {
            const RowVector2r grad = dl_df_before_clamp.segment(2 * ii, 2);
            dl_dq.segment(2 * i0, 2) += Vector2r(grad * df_node_dp0);
            dl_dq.segment(2 * i1, 2) += Vector2r(grad * df_node_dp1);
            dl_dv.segment(2 * i0, 2) += Vector2r(grad * df_node_dv0);
            dl_dv.segment(2 * i1, 2) += Vector2r(grad * df_node_dv1);
            dl_dp += VectorXr(grad * df_node_dp);
        }
    }
}

template<>
const VectorXr HydrodynamicsStateForce<2, 4>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr f = VectorXr::Zero(q.size());
    for (int i = 0; i < surface_face_num_; ++i) {
        const int i0 = surface_faces_(0, i);
        const int i1 = surface_faces_(1, i);
        const Vector2r p0 = q.segment(2 * i0, 2);
        const Vector2r p1 = q.segment(2 * i1, 2);
        const Vector2r v0 = v.segment(2 * i0, 2);
        const Vector2r v1 = v.segment(2 * i1, 2);
        // Compute A.
        const real A = (p0 - p1).norm();
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector2r v_rel = v_water_ - 0.5 * (v0 + v1);
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector2r d = v_rel / v_rel_len;
        // Compute n.
        // In 2D, we assume that the contour of the mesh is ccw.
        const Vector2r An = Vector2r(p1.y() - p0.y(), -p1.x() + p0.x());
        const Vector2r n = An / A;
        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real phi = std::acos(nd) - Pi() / 2;
        const Vector2r f_drag = 0.5 * rho_ * A * Cd(phi) * v_rel_len * v_rel;
        const Vector2r f_thrust = -0.5 * rho_ * Ct(phi) * v_rel_len * v_rel_len * An;
        // Added f_drag and f_thrust back to i0 and i1.
        const Vector2r f_node = (f_drag + f_thrust) * 0.5;
        f.segment(2 * i0, 2) += f_node;
        f.segment(2 * i1, 2) += f_node;
    }

    // Clamp.
    f = f.cwiseMax(-max_force_).cwiseMin(max_force_);

    return f;
}

template<>
void HydrodynamicsStateForce<2, 4>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    VectorXr dl_df_before_clamp = dl_df;
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        if (f(i) == max_force_ || f(i) == -max_force_) {
            dl_df_before_clamp(i) = 0;
        }
    }

    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(16);
    for (int i = 0; i < surface_face_num_; ++i) {
        const int i0 = surface_faces_(0, i);
        const int i1 = surface_faces_(1, i);
        const Vector2r p0 = q.segment(2 * i0, 2);
        const Vector2r p1 = q.segment(2 * i1, 2);
        const Vector2r v0 = v.segment(2 * i0, 2);
        const Vector2r v1 = v.segment(2 * i1, 2);
        // Compute A.
        const real A = (p0 - p1).norm();
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector2r v_rel = v_water_ - 0.5 * (v0 + v1);
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector2r d = v_rel / v_rel_len;

        // Compute derivatives of A and d.
        const Vector2r dA_dp0 = (p0 - p1) / A;
        const Vector2r dA_dp1 = -dA_dp0;
        const Matrix2r I = Matrix2r::Identity();
        const Matrix2r dd_dv0 = -0.5 * (I - d * d.transpose()) / v_rel_len;
        const Matrix2r dd_dv1 = dd_dv0;

        // Compute n.
        // In 2D, we assume that the contour of the mesh is ccw.
        const Vector2r An = Vector2r(p1.y() - p0.y(), -p1.x() + p0.x());
        const Vector2r n = An / A;

        // Compute derivatives of An and n.
        Matrix2r dAn_dp0;
        dAn_dp0 << 0, -1,
                    1, 0;
        const Matrix2r dAn_dp1 = -dAn_dp0;
        const Matrix2r dn_dp0 = (I - n * n.transpose()) / A * dAn_dp0;
        const Matrix2r dn_dp1 = -dn_dp0;

        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real nd2 = nd * nd;
        const real phi = std::acos(nd) - Pi() / 2;
        const real dacos = -1 / std::sqrt(1 - nd2);
        const RowVector2r dphi_dp0 = dacos * d.transpose() * dn_dp0;
        const RowVector2r dphi_dp1 = dacos * d.transpose() * dn_dp1;
        const RowVector2r dphi_dv0 = dacos * n.transpose() * dd_dv0;
        const RowVector2r dphi_dv1 = dacos * n.transpose() * dd_dv1;

        const real Cd_phi = Cd(phi);
        real Cd_grad_phi = 0;
        VectorXr Cd_grad_parameters = VectorXr::Zero(16);
        CdGradients(phi, Cd_grad_phi, Cd_grad_parameters);
        const Matrix2r df_drag_dp0 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp0.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp0);
        const Matrix2r df_drag_dp1 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp1.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp1);
        const Matrix2r df_drag_dv0 = 0.5 * rho_ * A * (-0.5 * I * Cd_phi * v_rel_len + v_rel * Cd_grad_phi * dphi_dv0 * v_rel_len
            -0.5 * v_rel * Cd_phi * d.transpose());
        const Matrix2r df_drag_dv1 = 0.5 * rho_ * A * (-0.5 * I * Cd_phi * v_rel_len + v_rel * Cd_grad_phi * dphi_dv1 * v_rel_len
            -0.5 * v_rel * Cd_phi * d.transpose());
        const Eigen::Matrix<real, 2, 16> df_drag_dp = 0.5 * rho_ * A * v_rel_len * v_rel * Cd_grad_parameters.transpose();

        const real Ct_phi = Ct(phi);
        real Ct_grad_phi = 0;
        VectorXr Ct_grad_parameters = VectorXr::Zero(16);
        CtGradients(phi, Ct_grad_phi, Ct_grad_parameters);
        const Matrix2r df_thrust_dp0 = -0.5 * rho_ * v_rel_len * v_rel_len * (
            An * Ct_grad_phi * dphi_dp0 + Ct_phi * dAn_dp0);
        const Matrix2r df_thrust_dp1 = -0.5 * rho_ * v_rel_len * v_rel_len * (
            An * Ct_grad_phi * dphi_dp1 + Ct_phi * dAn_dp1);
        const Matrix2r df_thrust_dv0 = -0.5 * rho_ * An * (
            Ct_grad_phi * dphi_dv0 * v_rel_len * v_rel_len - Ct_phi * v_rel.transpose());
        const Matrix2r df_thrust_dv1 = -0.5 * rho_ * An * (
            Ct_grad_phi * dphi_dv1 * v_rel_len * v_rel_len - Ct_phi * v_rel.transpose());
        const Eigen::Matrix<real, 2, 16> df_thrust_dp = -0.5 * rho_ * v_rel_len * v_rel_len * An * Ct_grad_parameters.transpose();

        // Added f_drag and f_thrust back to i0 and i1.
        const Matrix2r df_node_dp0 = (df_drag_dp0 + df_thrust_dp0) * 0.5;
        const Matrix2r df_node_dp1 = (df_drag_dp1 + df_thrust_dp1) * 0.5;
        const Matrix2r df_node_dv0 = (df_drag_dv0 + df_thrust_dv0) * 0.5;
        const Matrix2r df_node_dv1 = (df_drag_dv1 + df_thrust_dv1) * 0.5;
        const Eigen::Matrix<real, 2, 16> df_node_dp = (df_drag_dp + df_thrust_dp) * 0.5;

        for (int ii : { i0, i1 }) {
            const RowVector2r grad = dl_df_before_clamp.segment(2 * ii, 2);
            dl_dq.segment(2 * i0, 2) += Vector2r(grad * df_node_dp0);
            dl_dq.segment(2 * i1, 2) += Vector2r(grad * df_node_dp1);
            dl_dv.segment(2 * i0, 2) += Vector2r(grad * df_node_dv0);
            dl_dv.segment(2 * i1, 2) += Vector2r(grad * df_node_dv1);
            dl_dp += VectorXr(grad * df_node_dp);
        }
    }
}

template<>
const VectorXr HydrodynamicsStateForce<3, 4>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr f = VectorXr::Zero(q.size());
    for (int i = 0; i < surface_face_num_; ++i) {
        Eigen::Matrix<real, 3, 3> pos_i, vel_i;
        for (int j = 0; j < 3; ++j) {
            const int i0 = surface_faces_(j, i);
            pos_i.col(j) = q.segment(3 * i0, 3);
            vel_i.col(j) = v.segment(3 * i0, 3);
        }
        const Vector3r p0 = pos_i.col(0);
        const Vector3r p1 = pos_i.col(1);
        const Vector3r p2 = pos_i.col(2);
        const Vector3r unnormalized_n = (p1 - p0).cross(p2 - p1);
        const real A = unnormalized_n.norm();
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector3r v_rel = v_water_ - (vel_i.col(0) + vel_i.col(1) + vel_i.col(2)) / 3;
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector3r d = v_rel / v_rel_len;
        // Compute normal.
        const Vector3r n = unnormalized_n / A;
        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real phi = std::acos(nd) - Pi() / 2;
        const Vector3r f_drag = 0.5 * rho_ * A * Cd(phi) * v_rel_len * v_rel;
        const Vector3r f_thrust = -0.5 * rho_ * A * Ct(phi) * v_rel_len * v_rel_len * n;
        // Added f_drag and f_thrust back to i0 and i1.
        const Vector3r f_node = (f_drag + f_thrust) / 3;
        for (int j = 0; j < 3; ++j) {
            const int i0 = surface_faces_(j, i);
            f.segment(3 * i0, 3) += f_node;
        }
    }

    // Clamp.
    f = f.cwiseMax(-max_force_).cwiseMin(max_force_);

    return f;
}

template<>
void HydrodynamicsStateForce<3, 4>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    VectorXr dl_df_before_clamp = dl_df;
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        if (f(i) == max_force_ || f(i) == -max_force_) {
            dl_df_before_clamp(i) = 0;
        }
    }

    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(16);
    for (int i = 0; i < surface_face_num_; ++i) {
        Eigen::Matrix<real, 3, 3> pos_i, vel_i;
        for (int j = 0; j < 3; ++j) {
            const int i0 = surface_faces_(j, i);
            pos_i.col(j) = q.segment(3 * i0, 3);
            vel_i.col(j) = v.segment(3 * i0, 3);
        }
        const real eps = std::numeric_limits<real>::epsilon();
        const Vector3r p0 = pos_i.col(0);
        const Vector3r p1 = pos_i.col(1);
        const Vector3r p2 = pos_i.col(2);
        const Vector3r unnormalized_n = (p1 - p0).cross(p2 - p1);
        const real A = unnormalized_n.norm();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector3r v_rel = v_water_ - (vel_i.col(0) + vel_i.col(1) + vel_i.col(2)) / 3;
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;

        // Compute derivatives of A.
        // un = (p1 - p0) x (p2 - p1).
        // A = un.norm();
        const Matrix3r dun_dp0 = SkewSymmetricMatrix(p2 - p1);
        const Matrix3r dun_dp1 = SkewSymmetricMatrix(p0 - p2); 
        const Matrix3r dun_dp2 = SkewSymmetricMatrix(p1 - p0);
        const Vector3r dA_dp0 = dun_dp0.transpose() * (unnormalized_n / A);
        const Vector3r dA_dp1 = dun_dp1.transpose() * (unnormalized_n / A);
        const Vector3r dA_dp2 = dun_dp2.transpose() * (unnormalized_n / A);

        const Vector3r d = v_rel / v_rel_len;
        // Compute derivatives of d.
        const Matrix3r I = Matrix3r::Identity();
        const Matrix3r dd_dv = -(I - d * d.transpose()) / (3 * v_rel_len);

        // Compute normal.
        const Vector3r n = unnormalized_n / A;
        Matrix3r dn_dp0 = Matrix3r::Zero();
        Matrix3r dn_dp1 = Matrix3r::Zero();
        Matrix3r dn_dp2 = Matrix3r::Zero();
        const Matrix3r Inn = (I - n * n.transpose()) / A;
        dn_dp0 = Inn * dun_dp0;
        dn_dp1 = Inn * dun_dp1;
        dn_dp2 = Inn * dun_dp2;

        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real nd2 = nd * nd;
        const real phi = std::acos(nd) - Pi() / 2;
        const real dacos = -1 / std::sqrt(1 - nd2);
        const RowVector3r dphi_dp0 = dacos * d.transpose() * dn_dp0;
        const RowVector3r dphi_dp1 = dacos * d.transpose() * dn_dp1;
        const RowVector3r dphi_dp2 = dacos * d.transpose() * dn_dp2;
        const RowVector3r dphi_dv = dacos * n.transpose() * dd_dv;

        const real Cd_phi = Cd(phi);
        real Cd_grad_phi = 0;
        VectorXr Cd_grad_parameters = VectorXr::Zero(16);
        CdGradients(phi, Cd_grad_phi, Cd_grad_parameters);
        const Matrix3r df_drag_dp0 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp0.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp0);
        const Matrix3r df_drag_dp1 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp1.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp1);
        const Matrix3r df_drag_dp2 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp2.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp2);
        const Matrix3r df_drag_dv = 0.5 * rho_ * A * (-I * Cd_phi * v_rel_len / 3 + v_rel * Cd_grad_phi * dphi_dv * v_rel_len
            - v_rel * Cd_phi * d.transpose() / 3);
        Eigen::Matrix<real, 3, 16> df_drag_dp = 0.5 * rho_ * A * v_rel_len * v_rel * Cd_grad_parameters.transpose();

        const real Ct_phi = Ct(phi);
        real Ct_grad_phi = 0;
        VectorXr Ct_grad_parameters = VectorXr::Zero(16);
        CtGradients(phi, Ct_grad_phi, Ct_grad_parameters);
        const Matrix3r df_thrust_dp0 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp0 * A * Ct_phi
            + n * dA_dp0.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp0);
        const Matrix3r df_thrust_dp1 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp1 * A * Ct_phi
            + n * dA_dp1.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp1);
        const Matrix3r df_thrust_dp2 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp2 * A * Ct_phi
            + n * dA_dp2.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp2);
        const Matrix3r df_thrust_dv = -0.5 * rho_ * A * n * (Ct_grad_phi * dphi_dv * v_rel_len * v_rel_len
            - 2 * Ct_phi * v_rel.transpose() / 3);
        Eigen::Matrix<real, 3, 16> df_thrust_dp = -0.5 * rho_ * A * v_rel_len * v_rel_len * n * Ct_grad_parameters.transpose();

        // Added f_drag and f_thrust back to i0 and i1.
        const Matrix3r df_node_dp0 = (df_drag_dp0 + df_thrust_dp0) / 3;
        const Matrix3r df_node_dp1 = (df_drag_dp1 + df_thrust_dp1) / 3;
        const Matrix3r df_node_dp2 = (df_drag_dp2 + df_thrust_dp2) / 3;
        const Matrix3r df_node_dv = (df_drag_dv + df_thrust_dv) / 3;
        const Eigen::Matrix<real, 3, 16> df_node_dp = (df_drag_dp + df_thrust_dp) / 3;

        for (int j = 0; j < 3; ++j) {
            const int i0 = surface_faces_(j, i);
            const RowVector3r grad = dl_df_before_clamp.segment(3 * i0, 3);
            dl_dq.segment(3 * surface_faces_(0, i), 3) += Vector3r(grad * df_node_dp0);
            dl_dq.segment(3 * surface_faces_(1, i), 3) += Vector3r(grad * df_node_dp1);
            dl_dq.segment(3 * surface_faces_(2, i), 3) += Vector3r(grad * df_node_dp2);
            for (int k = 0; k < 3; ++k)
                dl_dv.segment(3 * surface_faces_(k, i), 3) += Vector3r(grad * df_node_dv);
            dl_dp += VectorXr(grad * df_node_dp);
        }
    }
}

template<>
const VectorXr HydrodynamicsStateForce<3, 8>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr f = VectorXr::Zero(q.size());
    // Triangle indices.
    std::array<std::array<int, 3>, 4> triangles{
        std::array<int, 3>{ 0, 1, 2 },
        std::array<int, 3>{ 0, 2, 3 },
        std::array<int, 3>{ 0, 1, 3 },
        std::array<int, 3>{ 3, 1, 2 }
    };
    for (int i = 0; i < surface_face_num_; ++i) {
        Eigen::Matrix<real, 3, 4> pos_i, vel_i;
        for (int j = 0; j < 4; ++j) {
            const int i0 = surface_faces_(j, i);
            pos_i.col(j) = q.segment(3 * i0, 3);
            vel_i.col(j) = v.segment(3 * i0, 3);
        }
        // Triangles:
        // 0, 1, 2; 0, 2, 3.
        // 0, 1, 3; 3, 1, 2.
        real A = 0;
        Vector3r unnormalized_n = Vector3r::Zero();
        for (int j = 0; j < 4; ++j) {
            const Vector3r p0 = pos_i.col(triangles[j][0]);
            const Vector3r p1 = pos_i.col(triangles[j][1]);
            const Vector3r p2 = pos_i.col(triangles[j][2]);
            unnormalized_n += (p1 - p0).cross(p2 - p1);
            A += (p1 - p0).cross(p2 - p1).norm();
        }
        A /= 4;
        const real eps = std::numeric_limits<real>::epsilon();
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector3r v_rel = v_water_ - (vel_i.col(0) + vel_i.col(1) + vel_i.col(2) + vel_i.col(3)) * 0.25;
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;
        const Vector3r d = v_rel / v_rel_len;
        // Compute normal.
        const real n_norm = unnormalized_n.norm();
        Vector3r n = Vector3r::Zero();
        if (n_norm > eps) n = unnormalized_n / n_norm;
        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real phi = std::acos(nd) - Pi() / 2;
        const Vector3r f_drag = 0.5 * rho_ * A * Cd(phi) * v_rel_len * v_rel;
        const Vector3r f_thrust = -0.5 * rho_ * A * Ct(phi) * v_rel_len * v_rel_len * n;
        // Added f_drag and f_thrust back to i0 and i1.
        const Vector3r f_node = (f_drag + f_thrust) * 0.25;
        for (int j = 0; j < 4; ++j) {
            const int i0 = surface_faces_(j, i);
            f.segment(3 * i0, 3) += f_node;
        }
    }

    // Clamp.
    f = f.cwiseMax(-max_force_).cwiseMin(max_force_);

    return f;
}

template<>
void HydrodynamicsStateForce<3, 8>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    VectorXr dl_df_before_clamp = dl_df;
    for (int i = 0; i < static_cast<int>(q.size()); ++i) {
        if (f(i) == max_force_ || f(i) == -max_force_) {
            dl_df_before_clamp(i) = 0;
        }
    }

    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
    dl_dp = VectorXr::Zero(16);
    // Triangle indices.
    std::array<std::array<int, 3>, 4> triangles{
        std::array<int, 3>{ 0, 1, 2 },
        std::array<int, 3>{ 0, 2, 3 },
        std::array<int, 3>{ 0, 1, 3 },
        std::array<int, 3>{ 3, 1, 2 }
    };
    for (int i = 0; i < surface_face_num_; ++i) {
        Eigen::Matrix<real, 3, 4> pos_i, vel_i;
        for (int j = 0; j < 4; ++j) {
            const int i0 = surface_faces_(j, i);
            pos_i.col(j) = q.segment(3 * i0, 3);
            vel_i.col(j) = v.segment(3 * i0, 3);
        }
        // Triangles:
        // 0, 1, 2; 0, 2, 3.
        // 0, 1, 3; 3, 1, 2.
        real A = 0;
        Vector3r unnormalized_n = Vector3r::Zero();
        std::array<Vector3r, 4> normalized_Aj;
        const real eps = std::numeric_limits<real>::epsilon();
        for (int j = 0; j < 4; ++j) {
            const Vector3r p0 = pos_i.col(triangles[j][0]);
            const Vector3r p1 = pos_i.col(triangles[j][1]);
            const Vector3r p2 = pos_i.col(triangles[j][2]);
            const Vector3r Aj = (p1 - p0).cross(p2 - p1);
            unnormalized_n += Aj;
            const real Aj_norm = Aj.norm();
            A += Aj_norm;
            if (Aj_norm <= eps) normalized_Aj[j] = Vector3r::Zero();
            else normalized_Aj[j] = Aj / Aj_norm;
        }
        A /= 4;
        if (A <= eps) continue;
        // Compute v_rel.
        const Vector3r v_rel = v_water_ - (vel_i.col(0) + vel_i.col(1) + vel_i.col(2) + vel_i.col(3)) * 0.25;
        const real v_rel_len = v_rel.norm();
        if (v_rel_len <= eps) continue;

        // Compute derivatives of A.
        // un = (p1 - p0) x (p2 - p1) + (p2 - p0) x (p3 - p2) + (p1 - p0) x (p3 - p1) + (p1 - p3) x (p2 - p1).
        const Vector3r p0 = pos_i.col(0), p1 = pos_i.col(1), p2 = pos_i.col(2), p3 = pos_i.col(3);
        const Vector3r dA_dp0 = (SkewSymmetricMatrix(p1 - p2) * normalized_Aj[0]
            + SkewSymmetricMatrix(p2 - p3) * normalized_Aj[1]
            + SkewSymmetricMatrix(p1 - p3) * normalized_Aj[2]) / 4;
        const Vector3r dA_dp1 = (SkewSymmetricMatrix(p2 - p0) * normalized_Aj[0]
            + SkewSymmetricMatrix(p3 - p0) * normalized_Aj[2]
            + SkewSymmetricMatrix(p2 - p3) * normalized_Aj[3]) / 4;
        const Vector3r dA_dp2 = (SkewSymmetricMatrix(p0 - p1) * normalized_Aj[0]
            + SkewSymmetricMatrix(p3 - p0) * normalized_Aj[1]
            + SkewSymmetricMatrix(p3 - p1) * normalized_Aj[3]) / 4;
        const Vector3r dA_dp3 = (SkewSymmetricMatrix(p0 - p2) * normalized_Aj[1]
            + SkewSymmetricMatrix(p0 - p1) * normalized_Aj[2]
            + SkewSymmetricMatrix(p1 - p2) * normalized_Aj[3]) / 4;

        const Matrix3r dun_dp0 = SkewSymmetricMatrix(2 * (p3 - p1));
        const Matrix3r dun_dp1 = SkewSymmetricMatrix(2 * (p0 - p2));
        const Matrix3r dun_dp2 = SkewSymmetricMatrix(2 * (p1 - p3));
        const Matrix3r dun_dp3 = SkewSymmetricMatrix(2 * (p2 - p0));

        const Vector3r d = v_rel / v_rel_len;
        // Compute derivatives of d.
        const Matrix3r I = Matrix3r::Identity();
        const Matrix3r dd_dv = -0.25 * (I - d * d.transpose()) / v_rel_len;

        // Compute normal.
        const real n_norm = unnormalized_n.norm();
        Vector3r n = Vector3r::Zero();
        Matrix3r dn_dp0 = Matrix3r::Zero();
        Matrix3r dn_dp1 = Matrix3r::Zero();
        Matrix3r dn_dp2 = Matrix3r::Zero();
        Matrix3r dn_dp3 = Matrix3r::Zero();
        if (n_norm > eps) {
            n = unnormalized_n / n_norm;
            const Matrix3r Inn = (I - n * n.transpose()) / n_norm;
            dn_dp0 = Inn * dun_dp0;
            dn_dp1 = Inn * dun_dp1;
            dn_dp2 = Inn * dun_dp2;
            dn_dp3 = Inn * dun_dp3;
        }

        // Compute the angle of attack.
        const real nd = n.dot(d);
        if (nd >= 0.0) continue;
        const real nd2 = nd * nd;
        const real phi = std::acos(nd) - Pi() / 2;
        const real dacos = -1 / std::sqrt(1 - nd2);
        const RowVector3r dphi_dp0 = dacos * d.transpose() * dn_dp0;
        const RowVector3r dphi_dp1 = dacos * d.transpose() * dn_dp1;
        const RowVector3r dphi_dp2 = dacos * d.transpose() * dn_dp2;
        const RowVector3r dphi_dp3 = dacos * d.transpose() * dn_dp3;
        const RowVector3r dphi_dv = dacos * n.transpose() * dd_dv;

        const real Cd_phi = Cd(phi);
        real Cd_grad_phi = 0;
        VectorXr Cd_grad_parameters = VectorXr::Zero(16);
        CdGradients(phi, Cd_grad_phi, Cd_grad_parameters);
        const Matrix3r df_drag_dp0 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp0.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp0);
        const Matrix3r df_drag_dp1 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp1.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp1);
        const Matrix3r df_drag_dp2 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp2.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp2);
        const Matrix3r df_drag_dp3 = 0.5 * rho_ * v_rel_len * v_rel * (dA_dp3.transpose() * Cd_phi + A * Cd_grad_phi * dphi_dp3);
        const Matrix3r df_drag_dv = 0.5 * rho_ * A * (-0.25 * I * Cd_phi * v_rel_len + v_rel * Cd_grad_phi * dphi_dv * v_rel_len
            -0.25 * v_rel * Cd_phi * d.transpose());
        const Eigen::Matrix<real, 3, 16> df_drag_dp = 0.5 * rho_ * A * v_rel_len * v_rel * Cd_grad_parameters.transpose();

        const real Ct_phi = Ct(phi);
        real Ct_grad_phi = 0;
        VectorXr Ct_grad_parameters = VectorXr::Zero(16);
        CtGradients(phi, Ct_grad_phi, Ct_grad_parameters);
        const Matrix3r df_thrust_dp0 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp0 * A * Ct_phi
            + n * dA_dp0.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp0);
        const Matrix3r df_thrust_dp1 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp1 * A * Ct_phi
            + n * dA_dp1.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp1);
        const Matrix3r df_thrust_dp2 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp2 * A * Ct_phi
            + n * dA_dp2.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp2);
        const Matrix3r df_thrust_dp3 = -0.5 * rho_ * v_rel_len * v_rel_len * (dn_dp3 * A * Ct_phi
            + n * dA_dp3.transpose() * Ct_phi + n * A * Ct_grad_phi * dphi_dp3);
        const Matrix3r df_thrust_dv = -0.5 * rho_ * A * n * (Ct_grad_phi * dphi_dv * v_rel_len * v_rel_len
            -0.5 * Ct_phi * v_rel.transpose());
        const Eigen::Matrix<real, 3, 16> df_thrust_dp = -0.5 * rho_ * A * v_rel_len * v_rel_len * n * Ct_grad_parameters.transpose();

        // Added f_drag and f_thrust back to i0 and i1.
        const Matrix3r df_node_dp0 = (df_drag_dp0 + df_thrust_dp0) * 0.25;
        const Matrix3r df_node_dp1 = (df_drag_dp1 + df_thrust_dp1) * 0.25;
        const Matrix3r df_node_dp2 = (df_drag_dp2 + df_thrust_dp2) * 0.25;
        const Matrix3r df_node_dp3 = (df_drag_dp3 + df_thrust_dp3) * 0.25;
        const Matrix3r df_node_dv = (df_drag_dv + df_thrust_dv) * 0.25;
        const Eigen::Matrix<real, 3, 16> df_node_dp = (df_drag_dp + df_thrust_dp) * 0.25;

        for (int j = 0; j < 4; ++j) {
            const int i0 = surface_faces_(j, i);
            const RowVector3r grad = dl_df_before_clamp.segment(3 * i0, 3);
            dl_dq.segment(3 * surface_faces_(0, i), 3) += Vector3r(grad * df_node_dp0);
            dl_dq.segment(3 * surface_faces_(1, i), 3) += Vector3r(grad * df_node_dp1);
            dl_dq.segment(3 * surface_faces_(2, i), 3) += Vector3r(grad * df_node_dp2);
            dl_dq.segment(3 * surface_faces_(3, i), 3) += Vector3r(grad * df_node_dp3);
            for (int k = 0; k < 4; ++k)
                dl_dv.segment(3 * surface_faces_(k, i), 3) += Vector3r(grad * df_node_dv);
            dl_dp += VectorXr(grad * df_node_dp);
        }
    }
}

template class HydrodynamicsStateForce<2, 3>;
template class HydrodynamicsStateForce<2, 4>;
template class HydrodynamicsStateForce<3, 4>;
template class HydrodynamicsStateForce<3, 8>;