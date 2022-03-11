#include "fem/deformable.h"
#include "state_force/gravitational_state_force.h"
#include "state_force/planar_contact_state_force.h"
#include "state_force/arc_contact_state_force.h"
#include "state_force/hydrodynamics_state_force.h"
#include "state_force/billiard_ball_state_force.h"

// Add state-based forces.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AddStateForce(const std::string& force_type, const std::vector<real>& params) {
    const int param_size = static_cast<int>(params.size());
    if (force_type == "gravity") {
        CheckError(param_size == vertex_dim, "Inconsistent params for GravitionalStateForce.");
        Eigen::Matrix<real, vertex_dim, 1> g;
        for (int i = 0; i < vertex_dim; ++i) g[i] = params[i];
        const real mass = density_ * element_volume_;
        auto force = std::make_shared<GravitationalStateForce<vertex_dim>>();
        force->Initialize(mass, g);
        state_forces_.push_back(force);
    } else if (force_type == "planar_contact") {
        CheckError(param_size == 5 + vertex_dim, "Inconsistent params for PlanarContactStateForce.");
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal(i) = params[i];
        const real offset = params[vertex_dim];
        const int p = static_cast<int>(params[vertex_dim + 1]);
        const real kn = params[vertex_dim + 2];
        const real kf = params[vertex_dim + 3];
        const real mu = params[vertex_dim + 4];
        auto force = std::make_shared<PlanarContactStateForce<vertex_dim>>();
        force->Initialize(normal, offset, p, kn, kf, mu);
        state_forces_.push_back(force);
    } else if (force_type == "arc_contact") {
        CheckError(param_size == 6 + 3 * vertex_dim, "Inconsistent params for ArcContactStateForce.");
        Eigen::Matrix<real, vertex_dim, 1> center, dir, start;
        for (int i = 0; i < vertex_dim; ++i) {
            center(i) = params[i];
            dir(i) = params[vertex_dim + i];
            start(i) = params[2 * vertex_dim + i];
        }
        const real radius = params[3 * vertex_dim];
        const real angle = params[3 * vertex_dim + 1];
        const int p = static_cast<int>(params[3 * vertex_dim + 2]);
        const real kn = params[3 * vertex_dim + 3];
        const real kf = params[3 * vertex_dim + 4];
        const real mu = params[3 * vertex_dim + 5];
        auto force = std::make_shared<ArcContactStateForce<vertex_dim>>();
        force->Initialize(center, dir, start, radius, angle, p, kn, kf, mu);
        state_forces_.push_back(force);
    } else if (force_type == "hydrodynamics") {
        const int face_dim = mesh_.GetNumOfVerticesInFace();
        const int face_num = (param_size - 2 - vertex_dim - 8 * 2) / face_dim;
        CheckError(param_size == 2 + vertex_dim + 8 * 2 + face_num * face_dim && face_num >= 1,
            "Inconsistent params for HydrodynamicsStateForce.");
        const real rho = params[0];
        Eigen::Matrix<real, vertex_dim, 1> v_water;
        for (int i = 0; i < vertex_dim; ++i) v_water(i) = params[1 + i];
        Eigen::Matrix<real, 4, 2> Cd_points, Ct_points;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j) {
                Cd_points(i, j) = params[1 + vertex_dim + i * 2 + j];
                Ct_points(i, j) = params[1 + vertex_dim + 8 + i * 2 + j];
            }
        const real max_force = params[1 + vertex_dim + 16];
        MatrixXi surface_faces = MatrixXi::Zero(face_dim, face_num);
        for (int i = 0; i < face_num; ++i)
            for (int j = 0; j < face_dim; ++j)
                surface_faces(j, i) = static_cast<int>(params[2 + vertex_dim + 8 * 2 + i * face_dim + j]);
        auto force = std::make_shared<HydrodynamicsStateForce<vertex_dim, element_dim>>();
        force->Initialize(rho, v_water, Cd_points, Ct_points, max_force, surface_faces);
        state_forces_.push_back(force);
    } else if (force_type == "billiard_ball") {
        const real radius = params[0];
        const int single_ball_vertex_num = static_cast<int>(params[1]);
        const int ball_num = (param_size - 2) / 2;
        CheckError(param_size == 2 + ball_num * 2, "Inconsistent params for BilliardBallStateForce.");
        std::vector<real> stiffness(ball_num, 0);
        std::vector<real> frictional_coeff(ball_num, 0);
        for (int i = 0; i < ball_num; ++i) {
            stiffness[i] = params[2 + i];
            frictional_coeff[i] = params[2 + ball_num + i];
        }
        auto force = std::make_shared<BilliardBallStateForce<vertex_dim>>();
        force->Initialize(radius, single_ball_vertex_num, stiffness, frictional_coeff);
        state_forces_.push_back(force);
    } else {
        PrintError("Unsupported state force type: " + force_type);
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ForwardStateForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr force = VectorXr::Zero(dofs_);
    for (const auto& f : state_forces_) force += f->ForwardForce(q, v);
    return force;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardStateForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    std::vector<real> dl_dp_vec;
    for (const auto& f : state_forces_) {
        const VectorXr fi = f->ForwardForce(q, v);
        VectorXr dl_dqi, dl_dvi, dl_dpi;
        f->BackwardForce(q, v, fi, dl_df, dl_dqi, dl_dvi, dl_dpi);
        dl_dq += dl_dqi;
        dl_dv += dl_dvi;
        const int param_dof = f->NumOfParameters();
        for (int i = 0; i < param_dof; ++i)
            dl_dp_vec.push_back(dl_dpi(i));
    }
    dl_dp = ToEigenVector(dl_dp_vec);
}

template<int vertex_dim, int element_dim>
const int Deformable<vertex_dim, element_dim>::NumOfStateForceParameters() const {
    int dofs = 0;
    for (const auto& f : state_forces_) dofs += f->NumOfParameters();
    return dofs;
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyForwardStateForce(const std::vector<real>& q,
    const std::vector<real>& v) const {
    return ToStdVector(ForwardStateForce(ToEigenVector(q), ToEigenVector(v)));
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackwardStateForce(const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq,
    std::vector<real>& dl_dv, std::vector<real>& dl_dp) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_dp_eig;
    BackwardStateForce(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f), ToEigenVector(dl_df),
        dl_dq_eig, dl_dv_eig, dl_dp_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_dp = ToStdVector(dl_dp_eig);
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;