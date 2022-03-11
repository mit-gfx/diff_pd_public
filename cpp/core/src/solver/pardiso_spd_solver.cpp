#include "solver/pardiso_spd_solver.h"
#include "common/common.h"

#ifdef PARDISO_AVAILABLE
// The following C-style code is required by Pardiso so we do not change it.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern "C" void pardisoinit (void*, int*, int*, int*, double*, int*);
extern "C" void pardiso(void*, int*, int*, int*, int*, int*, 
                      double*, int*, int*, int*, int*, int*,
                      int*, double*, double*, int*, double*);
extern "C" void pardiso_chkmatrix(int*, int*, double*, int*, int*, int*);
extern "C" void pardiso_chkvec(int*, int*, double*, int*);
extern "C" void pardiso_printstats(int*, int*, double*, int*, int*, int*, double*, int*);
#endif

void PardisoSpdSolver::Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options) {
#ifdef PARDISO_AVAILABLE
    const int thread_ct = static_cast<int>(options.at("thread_ct"));

    ia_ = nullptr;
    ja_ = nullptr;
    a_ = nullptr;

    // Solver parameters.
    mtype_ = 2;  // Use -2 for real symmetric indefinte matrix, 2 for real SPD, and 1 for structurally symmetric.
    solver_ = 0; // Use 1 for multi-recursive iterative solver.
    msglvl_ = 0; // Output lever. 0 = no output. 1 = print statistical information.
    maxfct_ = 1; // Maximum number of numerical factorizations.
    mnum_ = 1; // Which factorization to use.
    // End of parameters.

    CheckError(lhs.rows() == lhs.cols(), "A is not square.");
    n_ = static_cast<int>(lhs.cols());

    std::vector<int> ia_vec(n_ + 1, 0), ja_vec(0);
    std::vector<double> a_vec(0);
    // Explain the notation in Python:
    // For row i: ja[ia[i] : ia[i + 1]] is the indices of nonzero columns at row i.
    // a[ia[i] : ia[i + 1]] are the corresponding nonzero elements.
    for (int k = 0; k < lhs.outerSize(); ++k) {
        ia_vec[k + 1] = ia_vec[k];
        // Note that ia_vec[k] == ja_vec.size() == a_vec.size() is always true.
        // For symmetric matrices, Pardiso requires the diagonal elements to be always present, even if it is zero.
        std::deque<int> ja_row_k;
        std::deque<double> a_row_k;
        for (SparseMatrix::InnerIterator it(lhs, k); it; ++it) {
            // it.value() is the nonzero element.
            // it.row() is the row index.
            // it.col() is the column index and it equals k in this inner loop.
            // We make use of the fact that the matrix is symmetric.
            // Eigen guarantees row is sorted in each k.
            const int row = it.row();
            const double val = ToDouble(it.value());
            if (row < k) continue;
            // Now adding element value at (k, row) to the data.
            ja_row_k.push_back(row);
            a_row_k.push_back(val);
        }
        if (ja_row_k.empty() || ja_row_k.front() > k) {
            // Need to insert a fake diagonal element.
            ja_row_k.push_front(k);
            a_row_k.push_front(0);
        }
        ja_vec.insert(ja_vec.end(), ja_row_k.begin(), ja_row_k.end());
        a_vec.insert(a_vec.end(), a_row_k.begin(), a_row_k.end());
        ia_vec[k + 1] += static_cast<int>(ja_row_k.size());
    }
    // A Pardiso wrapper.
    ia_ = new int[ia_vec.size()];
    std::memcpy(ia_, ia_vec.data(), sizeof(int) * ia_vec.size());
    ja_ = new int[ja_vec.size()];
    std::memcpy(ja_, ja_vec.data(), sizeof(int) * ja_vec.size());
    a_ = new double[a_vec.size()];
    std::memcpy(a_, a_vec.data(), sizeof(double) * a_vec.size());
    // Convert matrix from 0-based C-notation to Fortran 1-based notation.
    int nnz = ia_[n_];
    for (int i = 0; i < n_ + 1; i++) {
        ia_[i] += 1;
    }
    for (int i = 0; i < nnz; i++) {
        ja_[i] += 1;
    }

    // Setup Pardiso control parameters.
    int error = 0;
    pardisoinit(pt_, &mtype_, &solver_, iparm_, dparm_, &error);

    if (error != 0) {
        if (error == -10)
            CheckError(false, "No license file found.");
        if (error == -11)
            CheckError(false, "License is expired.");
        if (error == -12)
            CheckError(false, "Wrong username or hostname.");
    }
    // Numbers of processors, value of OMP_NUM_THREADS.
    omp_set_num_threads(thread_ct);
    iparm_[2]  = thread_ct;

    // Check the consistency of the given matrix.
    // Use this functionality only for debugging purposes.
    pardiso_chkmatrix(&mtype_, &n_, a_, ia_, ja_, &error);
    CheckError(error == 0, "Pardiso solver: in consistency of matrix: " + std::to_string(error));

    // Reordering and Symbolic Factorization.
    // This step also allocates all memory that is necessary for the factorization.
    int phase = 11;
    double ddum;    // Double dummy.
    int idum;       // Integer dummy.
    int nrhs = 1;   // Number of right hand sides --- we only support 1 for now.
    pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, a_, ia_, ja_, &idum, &nrhs, iparm_, &msglvl_, &ddum, &ddum, &error, dparm_);
    CheckError(error == 0, "Pardiso solver: during symbolic factorization: " + std::to_string(error));
    if (msglvl_ > 0) {
        std::cout << "Number of nonzeros in factors = " << iparm_[17] << std::endl
            << "Number of factorization MFLOPS = " << iparm_[18] << std::endl;
    }

    // Numerical factorization.
    phase = 22;
    iparm_[32] = 1; // compute determinant.
    pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, a_, ia_, ja_, &idum, &nrhs,
            iparm_, &msglvl_, &ddum, &ddum, &error,  dparm_);
    CheckError(error == 0, "Pardiso: during numerical factorization: " + std::to_string(error));
#else
    CheckError(false, "You are calling Pardiso but you haven't compiled it yet.");
#endif
}

PardisoSpdSolver::~PardisoSpdSolver() {
#ifdef PARDISO_AVAILABLE
    if (ia_) {
        // Termination and release of memory.
        int phase = -1; // Release internal memory.
        double ddum;    // Double dummy.
        int idum;       // Integer dummy.
        int error = 0;
        int nrhs = 1;   // Number of right hand sides --- we only support 1 for now.
        pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, &ddum, ia_, ja_, &idum, &nrhs,
                iparm_, &msglvl_, &ddum, &ddum, &error, dparm_);
        CheckError(error == 0, "Pardiso: in releasing memory: " + std::to_string(error));
    }
    if (ia_) {
        delete[] ia_;
        ia_ = nullptr;
    }
    if (ja_) {
        delete[] ja_;
        ja_ = nullptr;
    }
    if (a_) {
        delete[] a_;
        a_ = nullptr;
    }
#endif
}

const VectorXr PardisoSpdSolver::Solve(const VectorXr& rhs) {
#ifdef PARDISO_AVAILABLE
    // We do not perform symmetry check as it is quite expensive in SparseMatrix.
    CheckError(n_ == static_cast<int>(rhs.size()), "Inconsistent rhs size: " + std::to_string(n_) + ", "
        + std::to_string(rhs.size()));
    CheckError(rhs.size() > 0, "You are passing empty rhs.");

    // Translating rhs.
    std::vector<double> b_vec(n_, 0);
    for (int i = 0; i < n_; ++i) b_vec[i] = ToDouble(rhs(i));
    double* b = b_vec.data();
    // Solution.
    std::vector<double> x_vec(n_, 0);
    double* x = x_vec.data();   // Solution will be written back to x.
    int nrhs = 1;               // Number of right hand sides. Use 1 because the input is VectorXr.

    // Pardiso control parameters.
    int error = 0;
    error  = 0; // Initialize error flag.
    // Checks the given vectors for infinite and NaN values.
    // Use this functionality only for debugging purposes.
    pardiso_chkvec(&n_, &nrhs, b, &error);
    CheckError(error == 0, "Pardiso solver: in right hand side: " + std::to_string(error));
 
    if (msglvl_ > 0) {
        // Prints information on the matrix to STDOUT.
        // Use this functionality only for debugging purposes. 
        pardiso_printstats(&mtype_, &n_, a_, ia_, ja_, &nrhs, b, &error);
        CheckError(error == 0, "Pardiso solver: error in printstats: " + std::to_string(error));
    }

    // Back substitution and iterative refinement.
    int phase = 33;
    // Max numbers of iterative refinement steps.
    iparm_[7] = 1;
    int idum;   // Integer dummy.
    pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, a_, ia_, ja_, &idum, &nrhs, iparm_, &msglvl_, b, x, &error, dparm_);
    CheckError(error == 0, "Pardiso: during solution: " + std::to_string(error));

    // Write back solution.
    VectorXr x_sol = VectorXr::Zero(n_);
    for (int i = 0; i < n_; ++i) x_sol(i) = ToReal(x[i]);

    return x_sol;
#else
    CheckError(false, "You are calling Pardiso but you haven't compiled it yet.");
    return VectorXr::Zero(rhs.size());
#endif
}