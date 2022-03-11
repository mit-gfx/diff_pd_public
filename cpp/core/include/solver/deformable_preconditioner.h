#ifndef SOLVER_Deformable_PRECONDITIONER_H
#define SOLVER_Deformable_PRECONDITIONER_H

#include "common/config.h"
#include "fem/deformable.h"

namespace Eigen {

// Based on the DiagonalPreconditioner in Eigen.
template <typename _Scalar>
class DeformablePreconditioner {
    typedef _Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, 1> Vector;
public:
    typedef typename Vector::StorageIndex StorageIndex;
    enum {
        ColsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic
    };

    DeformablePreconditioner() {}

    template<typename MatType>
    explicit DeformablePreconditioner(const MatType& mat)
        : rows_(mat.rows()), cols_(mat.cols()) {
        compute(mat);
    }

    Index rows() const { return rows_; }
    Index cols() const { return cols_; }

    template<typename MatType>
    DeformablePreconditioner& analyzePattern(const MatType& ) { return *this; }

    template<typename MatType>
    DeformablePreconditioner& factorize(const MatType& mat) { return *this; }

    template<typename MatType>
    DeformablePreconditioner& compute(const MatType& mat) { return *this; }

    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
        // What this function (and this preconditioner) intends to do:
        // - Given mat, find A that is similar to mat (This is done in factorize and compute).
        // - Solve A x = b and write the results back to x.
        if (global_vertex_dim == 2 && global_element_dim == 3) {
            x = reinterpret_cast<const Deformable<2, 3>*>(global_deformable)->PdLhsSolve(global_pd_backward_method,
                b, global_additional_dirichlet_boundary, true, false);
        } else if (global_vertex_dim == 2 && global_element_dim == 4) {
            x = reinterpret_cast<const Deformable<2, 4>*>(global_deformable)->PdLhsSolve(global_pd_backward_method,
                b, global_additional_dirichlet_boundary, true, false);
        } else if (global_vertex_dim == 3 && global_element_dim == 4) {
            x = reinterpret_cast<const Deformable<3, 4>*>(global_deformable)->PdLhsSolve(global_pd_backward_method,
                b, global_additional_dirichlet_boundary, true, false);
        } else if (global_vertex_dim == 3 && global_element_dim == 8) {
            x = reinterpret_cast<const Deformable<3, 8>*>(global_deformable)->PdLhsSolve(global_pd_backward_method,
                b, global_additional_dirichlet_boundary, true, false);
        } else {
            CheckError(false, "Unusual deformable.");
        }
    }

    template<typename Rhs> inline const Solve<DeformablePreconditioner, Rhs>
    solve(const MatrixBase<Rhs>& b) const {
        eigen_assert((rows_ == b.rows() && rows_ == cols_) &&
            && "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
        return Solve<DeformablePreconditioner, Rhs>(*this, b.derived());
    }

    ComputationInfo info() { return Success; }

protected:
    Index rows_, cols_;
};

}

#endif