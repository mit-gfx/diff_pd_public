#include "state_force/billiard_ball_state_force.h"
#include "common/common.h"
#include "common/geometry.h"
#include "mesh/mesh.h"

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::Initialize(const real radius, const int single_ball_vertex_num,
    const std::vector<real>& stiffness, const std::vector<real>& frictional_coeff) {
    radius_ = radius;
    single_ball_vertex_num_ = single_ball_vertex_num;
    ball_num_ = static_cast<int>(stiffness.size());
    VectorXr parameters = VectorXr::Zero(2 * ball_num_);
    for (int i = 0; i < ball_num_; ++i) {
        parameters(i) = stiffness[i];
        parameters(ball_num_ + i) = frictional_coeff[i];
    }
    StateForce<vertex_dim>::set_parameters(parameters);
}

template<int vertex_dim>
const VectorXr BilliardBallStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    // Reshape q to n x dim.
    const int vertex_num = static_cast<int>(q.size()) / vertex_dim;
    CheckError(vertex_num * vertex_dim == static_cast<int>(q.size()) && vertex_num % single_ball_vertex_num_ == 0,
        "Incompatible vertex number.");
    const int ball_num = vertex_num / single_ball_vertex_num_;
    CheckError(ball_num == ball_num_, "Inconsistent ball_num.");
    std::vector<MatrixXr> vertices(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    std::vector<MatrixXr> vertex_velocities(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    for (int i = 0; i < ball_num; ++i)
        for (int j = 0; j < single_ball_vertex_num_; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                vertices[i](k, j) = q(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
                vertex_velocities[i](k, j) = v(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
            }

    // Compute the center of each ball.
    MatrixXr centers = MatrixXr::Zero(vertex_dim, ball_num);
    MatrixXr central_velocities = MatrixXr::Zero(vertex_dim, ball_num);
    for (int i = 0; i < ball_num; ++i) {
        centers.col(i) = vertices[i].rowwise().mean();
        central_velocities.col(i) = vertex_velocities[i].rowwise().mean();
    }

    // Compute the distance between centers.
    VectorXr force = VectorXr::Zero(q.size());
    for (int i = 0; i < ball_num; ++i)
        for (int j = i + 1; j < ball_num; ++j) {
            const VectorXr ci = centers.col(i);
            const VectorXr cj = centers.col(j);
            const VectorXr dir_i2j = cj - ci;
            const real cij_dist = dir_i2j.norm();
            // Now compute the spring force.
            const real fj_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness(j);
            const real fi_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness(i);
            const VectorXr i2j = dir_i2j / cij_dist;
            const VectorXr fj = i2j * fj_mag;
            const VectorXr fi = -i2j * fi_mag;
            // Next, compute the friction force.
            const real ffj_mag = fj_mag * frictional_coeff(j);
            const real ffi_mag = fi_mag * frictional_coeff(i);
            const VectorXr vi = central_velocities.col(i);
            const VectorXr vj = central_velocities.col(j);
            const VectorXr vi_in_j = vi - vj;
            // Test the friction direction.
            Eigen::Matrix<real, vertex_dim, 1> ffi_dir = Eigen::Matrix<real, vertex_dim, 1>::Zero();
            if (vi_in_j.x() * i2j.y() - vi_in_j.y() * i2j.x() > 0) {
                // Rotate i2j by 90 degrees to obtain ff_i.
                ffi_dir.x() = -i2j.y();
                ffi_dir.y() = i2j.x();
            } else {
                // Rotate i2j by -90 degrees to obtain ff_i.
                ffi_dir.x() = i2j.y();
                ffi_dir.y() = -i2j.x();
            }
            const VectorXr ffi = ffi_dir * ffi_mag;
            const VectorXr ffj = -ffi_dir * ffj_mag;
            // Now distribute fi and fj to ball i and j, respectively.
            for (int k = 0; k < single_ball_vertex_num_; ++k) {
                force.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) +=
                    (fi + ffi) / single_ball_vertex_num_;
                force.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) +=
                    (fj + ffj) / single_ball_vertex_num_;
            }
        }
    return force;
}

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dp = VectorXr::Zero(StateForce<vertex_dim>::NumOfParameters());

    // Reshape q to n x dim.
    const int vertex_num = static_cast<int>(q.size()) / vertex_dim;
    CheckError(vertex_num * vertex_dim == static_cast<int>(q.size()) && vertex_num % single_ball_vertex_num_ == 0,
        "Incompatible vertex number.");
    const int ball_num = vertex_num / single_ball_vertex_num_;
    std::vector<MatrixXr> vertices(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    std::vector<MatrixXr> vertex_velocities(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    for (int i = 0; i < ball_num; ++i)
        for (int j = 0; j < single_ball_vertex_num_; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                vertices[i](k, j) = q(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
                vertex_velocities[i](k, j) = v(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
            }

    // Compute the center of each ball.
    MatrixXr centers = MatrixXr::Zero(vertex_dim, ball_num);
    MatrixXr central_velocities = MatrixXr::Zero(vertex_dim, ball_num);
    for (int i = 0; i < ball_num; ++i) {
        centers.col(i) = vertices[i].rowwise().mean();
        central_velocities.col(i) = vertex_velocities[i].rowwise().mean();
    }

    // Work on dl_dq.
    dl_dq = VectorXr::Zero(q.size());
    for (int i = 0; i < ball_num; ++i)
        for (int j = i + 1; j < ball_num; ++j) {
            const VectorXr ci = centers.col(i);
            const VectorXr cj = centers.col(j);
            const VectorXr dir_i2j = cj - ci;
            const MatrixXr jac_dir_i2j_ci = -Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
            const MatrixXr jac_dir_i2j_cj = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
            const real cij_dist = dir_i2j.norm();
            // We allow extremely soft balls.
            CheckError(cij_dist > 0.1 * radius_, "Balls are too close to each other");
            const VectorXr d_cij_dist_ci = jac_dir_i2j_ci.transpose() * dir_i2j / cij_dist;
            const VectorXr d_cij_dist_cj = jac_dir_i2j_cj.transpose() * dir_i2j / cij_dist;
            // Now compute the force.
            const real fj_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness(j);
            const real fi_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness(i);
            VectorXr d_fj_mag_dp = VectorXr::Zero(2 * ball_num);
            VectorXr d_fi_mag_dp = VectorXr::Zero(2 * ball_num);
            d_fj_mag_dp(j) = std::max(2 * radius_ - cij_dist, 0.0);
            d_fi_mag_dp(i) = std::max(2 * radius_ - cij_dist, 0.0);
            VectorXr d_fj_mag_ci = VectorXr::Zero(vertex_dim);
            VectorXr d_fj_mag_cj = VectorXr::Zero(vertex_dim);
            VectorXr d_fi_mag_ci = VectorXr::Zero(vertex_dim);
            VectorXr d_fi_mag_cj = VectorXr::Zero(vertex_dim);
            if (fj_mag > 0) {
                d_fj_mag_ci = -stiffness(j) * d_cij_dist_ci;
                d_fj_mag_cj = -stiffness(j) * d_cij_dist_cj;
            }
            if (fi_mag > 0) {
                d_fi_mag_ci = -stiffness(i) * d_cij_dist_ci;
                d_fi_mag_cj = -stiffness(i) * d_cij_dist_cj;
            }
            const VectorXr i2j = dir_i2j / cij_dist;
            const MatrixXr jac_i2j_ci = jac_dir_i2j_ci / cij_dist - dir_i2j * d_cij_dist_ci.transpose() / (cij_dist * cij_dist);
            const MatrixXr jac_i2j_cj = jac_dir_i2j_cj / cij_dist - dir_i2j * d_cij_dist_cj.transpose() / (cij_dist * cij_dist);
            // const VectorXr fj = i2j * fj_mag;
            // const VectorXr fi = -i2j * fi_mag;
            const MatrixXr jac_fj_p = i2j * d_fj_mag_dp.transpose();
            const MatrixXr jac_fi_p = -i2j * d_fi_mag_dp.transpose();
            const MatrixXr jac_fj_ci = jac_i2j_ci * fj_mag + i2j * d_fj_mag_ci.transpose();
            const MatrixXr jac_fj_cj = jac_i2j_cj * fj_mag + i2j * d_fj_mag_cj.transpose();
            const MatrixXr jac_fi_ci = -jac_i2j_ci * fi_mag - i2j * d_fi_mag_ci.transpose();
            const MatrixXr jac_fi_cj = -jac_i2j_cj * fi_mag - i2j * d_fi_mag_cj.transpose();
            // Next, compute the friction force.
            // const real ffj_mag = fj_mag * frictional_coeff(j);
            // const real ffi_mag = fi_mag * frictional_coeff(i);
            const real ffj_mag = fj_mag * frictional_coeff(j);
            const real ffi_mag = fi_mag * frictional_coeff(i);
            VectorXr d_ffj_mag_dp = VectorXr::Zero(2 * single_ball_vertex_num_);
            VectorXr d_ffi_mag_dp = VectorXr::Zero(2 * single_ball_vertex_num_);
            d_ffj_mag_dp(ball_num + j) = fj_mag;
            d_ffj_mag_dp(j) = std::max(2 * radius_ - cij_dist, 0.0) * frictional_coeff(j);
            d_ffi_mag_dp(ball_num + i) = fi_mag;
            d_ffi_mag_dp(i) = std::max(2 * radius_ - cij_dist, 0.0) * frictional_coeff(i);
            const VectorXr d_ffj_mag_ci = d_fj_mag_ci * frictional_coeff(j);
            const VectorXr d_ffj_mag_cj = d_fj_mag_cj * frictional_coeff(j);
            const VectorXr d_ffi_mag_ci = d_fi_mag_ci * frictional_coeff(i);
            const VectorXr d_ffi_mag_cj = d_fi_mag_cj * frictional_coeff(i);
            const VectorXr vi = central_velocities.col(i);
            const VectorXr vj = central_velocities.col(j);
            const VectorXr vi_in_j = vi - vj;
            // Test the friction direction.
            Eigen::Matrix<real, vertex_dim, 1> ffi_dir = Eigen::Matrix<real, vertex_dim, 1>::Zero();
            MatrixXr jac_ffi_dir_ci = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            MatrixXr jac_ffi_dir_cj = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            if (vi_in_j.x() * i2j.y() - vi_in_j.y() * i2j.x() > 0) {
                // Rotate i2j by 90 degrees to obtain ff_i.
                ffi_dir.x() = -i2j.y();
                ffi_dir.y() = i2j.x();
                jac_ffi_dir_ci.row(0) = -jac_i2j_ci.row(1);
                jac_ffi_dir_ci.row(1) = jac_i2j_ci.row(0);
                jac_ffi_dir_cj.row(0) = -jac_i2j_cj.row(1);
                jac_ffi_dir_cj.row(1) = jac_i2j_cj.row(0);
            } else {
                // Rotate i2j by -90 degrees to obtain ff_i.
                ffi_dir.x() = i2j.y();
                ffi_dir.y() = -i2j.x();
                jac_ffi_dir_ci.row(0) = jac_i2j_ci.row(1);
                jac_ffi_dir_ci.row(1) = -jac_i2j_ci.row(0);
                jac_ffi_dir_cj.row(0) = jac_i2j_cj.row(1);
                jac_ffi_dir_cj.row(1) = -jac_i2j_cj.row(0);
            }
            // const VectorXr ffi = ffi_dir * ffi_mag;
            // const VectorXr ffj = -ffi_dir * ffj_mag;
            const MatrixXr jac_ffi_p = ffi_dir * d_ffi_mag_dp.transpose();
            const MatrixXr jac_ffj_p = -ffi_dir * d_ffj_mag_dp.transpose();
            const MatrixXr jac_ffi_ci = jac_ffi_dir_ci * ffi_mag + ffi_dir * d_ffi_mag_ci.transpose();
            const MatrixXr jac_ffi_cj = jac_ffi_dir_cj * ffi_mag + ffi_dir * d_ffi_mag_cj.transpose();
            const MatrixXr jac_ffj_ci = -jac_ffi_dir_ci * ffj_mag - ffi_dir * d_ffj_mag_ci.transpose();
            const MatrixXr jac_ffj_cj = -jac_ffi_dir_cj * ffj_mag - ffi_dir * d_ffj_mag_cj.transpose();
            // Now distribute fi and fj to ball i and j, respectively.
            VectorXr dl_ci = VectorXr::Zero(vertex_dim);
            VectorXr dl_cj = VectorXr::Zero(vertex_dim);
            for (int k = 0; k < single_ball_vertex_num_; ++k) {
                // force.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fi / single_ball_vertex_num_;
                // force.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fj / single_ball_vertex_num_;
                const VectorXr dl_dfi = dl_df.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim)
                    / single_ball_vertex_num_;
                const VectorXr dl_dfj = dl_df.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim)
                    / single_ball_vertex_num_;
                dl_ci += (jac_fi_ci + jac_ffi_ci).transpose() * dl_dfi + (jac_fj_ci + jac_ffj_ci).transpose() * dl_dfj;
                dl_cj += (jac_fi_cj + jac_ffi_cj).transpose() * dl_dfi + (jac_fj_cj + jac_ffj_cj).transpose() * dl_dfj;
                dl_dp += (jac_fi_p + jac_ffi_p).transpose() * dl_dfi + (jac_fj_p + jac_ffj_p).transpose() * dl_dfj;
            }
            // Backpropagate from centers[i] and centers[j] to q.
            for (int p = 0; p < single_ball_vertex_num_; ++p)
                for (int k = 0; k < vertex_dim; ++k) {
                    dl_dq(i * single_ball_vertex_num_ * vertex_dim + p * vertex_dim + k) += dl_ci(k) / single_ball_vertex_num_;
                    dl_dq(j * single_ball_vertex_num_ * vertex_dim + p * vertex_dim + k) += dl_cj(k) / single_ball_vertex_num_;
                }
        }

    // Work on dl_dv.
    dl_dv = VectorXr::Zero(v.size());
}

template class BilliardBallStateForce<2>;
template class BilliardBallStateForce<3>;