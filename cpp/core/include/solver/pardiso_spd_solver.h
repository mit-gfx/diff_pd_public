#ifndef SOLVER_PARDISO_SPD_SOLVER_H
#define SOLVER_PARDISO_SPD_SOLVER_H

#include "common/config.h"

class PardisoSpdSolver {
public:
    PardisoSpdSolver(): ia_(nullptr), ja_(nullptr), a_(nullptr) {}
    ~PardisoSpdSolver();

    void Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options);
    const VectorXr Solve(const VectorXr& rhs);

private:
    int n_;
    int* ia_;
    int* ja_;
    double* a_;

    // Solver parameters.
    int mtype_; // Use -2 for real symmetric indefinte matrix, 2 for real SPD, and 1 for structurally symmetric.
    int solver_; // Use 1 for multi-recursive iterative solver.
    int msglvl_; // Output lever. 0 = no output. 1 = print statistical information.
    int maxfct_; // Maximum number of numerical factorizations.
    int mnum_; // Which factorization to use.
    // End of parameters.

    void* pt_[64];
    int iparm_[64];
    double dparm_[64];
};

#endif