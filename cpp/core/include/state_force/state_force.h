#ifndef FEM_STATE_FORCE_H
#define FEM_STATE_FORCE_H

// This base class implements forces that depend on q and v.
#include "common/config.h"
#include "common/common.h"

// How to use this class:
// - Implement an Initialize() function. The initialize function takes as input some parameters.
// - In the Initialize() function, call set_parameters().
// See gravitational_state_force for an example.

template<int vertex_dim>
class StateForce {
public:
    StateForce() : parameters_(VectorXr::Zero(0)) {}
    virtual ~StateForce() {}

    virtual const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const;
    virtual void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const;

    // Python binding --- used for testing purposes only.
    const std::vector<real> PyForwardForce(const std::vector<real>& q, const std::vector<real>& v) const;
    void PyBackwardForce(const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq,
        std::vector<real>& dl_dv, std::vector<real>& dl_dp) const;

    const VectorXr& parameters() const { return parameters_; }
    const std::vector<real> py_parameters() const { return ToStdVector(parameters_); }
    const int NumOfParameters() const { return static_cast<int>(parameters_.size()); }

protected:
    void set_parameters(const VectorXr& parameters) { parameters_ = parameters; }

private:
    VectorXr parameters_;
};

#endif