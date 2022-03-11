#include <iostream>
#include "Eigen/Dense"
#include "common/config.h"
#include "common/common.h"
#include "solver/pardiso_spd_solver.h"

int main(int argc, char* argv[]) {
#ifdef PARDISO_AVAILABLE
    const int n = 12;
    SparseMatrix AtA[2];
    VectorXr b[2];
    for (int i = 0; i < 2; ++i) {
        const SparseMatrix A = MatrixXr::Random(n, n).sparseView(1, 0.25);
        AtA[i] = A.transpose() * A;
        b[i] = VectorXr::Random(n);
    }
    PardisoSpdSolver solver[2];
    std::map<std::string, real> options;
    options["thread_ct"] = 4;
    for (int i = 0; i < 2; ++i)
        solver[i].Compute(AtA[i], options);

    for (int i = 0; i < 2; ++i) {
        const VectorXr x = solver[i].Solve(b[i]);
        const real abs_error = (AtA[i] * x - b[i]).norm();
        const real rel_error = abs_error / b[i].norm();
        std::cout << "abs_error: " << abs_error << ", rel_error: " << rel_error << std::endl;
    }
#else
    PrintInfo("The program compiles fine. Pardiso is not detected.");
#endif
    return 0;
}
