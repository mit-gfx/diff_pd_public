#ifndef STATE_FORCE_BILLIARD_BALL_STATE_FORCE_H
#define STATE_FORCE_BILLIARD_BALL_STATE_FORCE_H

#include "state_force/state_force.h"
#include "common/common.h"

// Trainable parameters: 
// stiffness and frictional_coeff.
template<int vertex_dim>
class BilliardBallStateForce : public StateForce<vertex_dim> {
public:
    void Initialize(const real radius, const int single_ball_vertex_num,
        const std::vector<real> &stiffness, const std::vector<real>& frictional_coeff);

    const real radius() const { return radius_; }
    const int single_ball_vertex_num() const { return single_ball_vertex_num_; }
    const real stiffness(const int ball_idx) const { return StateForce<vertex_dim>::parameters()(ball_idx); }
    const real frictional_coeff(const int ball_idx) const {
        return StateForce<vertex_dim>::parameters()(ball_num_ + ball_idx);
    }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const override;

private:
    real radius_;
    int single_ball_vertex_num_;
    int ball_num_;

    // Parameters for the impulse-based contact model (essentially a spring).
    // Step 1: Compute c.o.m. positions of each billiard ball from q.
    // Step 2: Use the stiffness to compute the spring force.
    // Step 3: Use the frictional_coeff to compute the friction force.
    // Step 4: Distribute the spring and frictional force equally to each vertex in q.
};

#endif