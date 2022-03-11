#ifndef SOLVER_MATRIX_OP_H
#define SOLVER_MATRIX_OP_H

#include "common/config.h"

// The crazy Eigen matrix-free solver --- TBH I cannot understand why they have to make it this complicated.
class MatrixOp;

namespace Eigen {
    namespace internal {

    template<>
    struct traits<MatrixOp> : public Eigen::internal::traits<::SparseMatrix> {};

    }
}

class MatrixOp : public Eigen::EigenBase<MatrixOp> {
public:
    typedef real Scalar;
    typedef real RealScalar;
    typedef int StorageIndex;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };
    template<typename Rhs>
    Eigen::Product<MatrixOp, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<MatrixOp, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }
    // Custom APIs.
    MatrixOp(const int row_num, const int col_num,
        const std::function<const VectorXr(const VectorXr& x)>& op)
        : row_num_(row_num), col_num_(col_num), op_(op) {}
    Index rows() const { return row_num_; }
    Index cols() const { return col_num_; }
    const VectorXr Apply(const VectorXr& x) const { return op_(x); }

private:
    const int row_num_;
    const int col_num_;
    const std::function<const VectorXr(const VectorXr& x)> op_;
};

namespace Eigen {
    namespace internal {

    template<typename Rhs>
    struct generic_product_impl<MatrixOp, Rhs, SparseShape, DenseShape, GemvProduct>
        : generic_product_impl_base<MatrixOp, Rhs, generic_product_impl<MatrixOp, Rhs>> {
            typedef typename Product<MatrixOp, Rhs>::Scalar Scalar;
            template<typename Dest>
            static void scaleAndAddTo(Dest& dst, const MatrixOp& lhs, const Rhs& rhs, const Scalar& alpha) {
            // This method should implement "dst += alpha * lhs * rhs" inplace,
            // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
            assert(alpha == Scalar(1) && "Scaling is not implemented.");
            EIGEN_ONLY_USED_FOR_DEBUG(alpha);
            dst += lhs.Apply(rhs);
        }
    };

    }
}

#endif