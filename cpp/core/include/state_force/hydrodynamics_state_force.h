#ifndef STATE_FORCE_HYDRODYNAMICS_STATE_FORCE_H
#define STATE_FORCE_HYDRODYNAMICS_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

// Please see Eq. (6) from the SoftCon paper for more details.
// The parameters are: Cd_points (4 x 2) and Ct_points (4 x 2):
// Cd_points(0, 0), Cd_points(0, 1), Cd_points(1, 0), Cd_points(1, 1), ...
// Ct_points(0, 0), Ct_points(0, 1), Ct_points(1, 0), Ct_points(1, 1), ...
template<int vertex_dim, int element_dim>
class HydrodynamicsStateForce : public StateForce<vertex_dim> {
public:
    HydrodynamicsStateForce();

    void Initialize(const real rho, const Eigen::Matrix<real, vertex_dim, 1>& v_water,
        const Eigen::Matrix<real, 4, 2>& Cd_points, const Eigen::Matrix<real, 4, 2>& Ct_points,
        const real max_force, const MatrixXi& surface_faces);
    void PyInitialize(const real rho, const std::array<real, vertex_dim>& v_water,
        const std::vector<real>& Cd_points, const std::vector<real>& Ct_points,
        const real max_force, const std::vector<int>& surface_faces);

    const Eigen::Matrix<real, 4, 2> Cd_points() const;
    const Eigen::Matrix<real, 4, 2> Ct_points() const;

    const real rho() const { return rho_; }
    const Eigen::Matrix<real, vertex_dim, 1>& v_water() const { return v_water_; }
    const real Cd(const real angle) const;
    const real Ct(const real angle) const;
    void CdGradients(const real angle, real& grad_angle, VectorXr& grad_parameters) const;
    void CtGradients(const real angle, real& grad_angle, VectorXr& grad_parameters) const;

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    real rho_;
    Eigen::Matrix<real, vertex_dim, 1> v_water_;
    Eigen::Matrix<real, 4, 2> Cd_points_;
    Eigen::Matrix<real, 4, 2> Ct_points_;
    real max_force_;
    MatrixXi surface_faces_;
    int surface_face_num_;
};

#endif