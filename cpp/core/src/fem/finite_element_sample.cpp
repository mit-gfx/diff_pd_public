#include "fem/finite_element_sample.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
void FiniteElementSample<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, 1>& undeformed_sample,
    const Eigen::Matrix<real, element_dim, vertex_dim>& grad_undeformed_sample_weight) {
    undeformed_sample_ = undeformed_sample;
    grad_undeformed_sample_weight_ = grad_undeformed_sample_weight;

    // Initialize dF_dxkd_flattened.
    dF_dxkd_flattened_.setZero();
    // F_ij = xik * grad_kj.

    SparseMatrixElements pd_A_nonzeros, pd_At_nonzeros;
    for (int i = 0; i < vertex_dim; ++i)
        for (int j = 0; j < vertex_dim; ++j)
            for (int k = 0; k < element_dim; ++k) {
                const int row = i + j * vertex_dim;
                const int col = i + k * vertex_dim;
                const real val = grad_undeformed_sample_weight_(k, j);
                dF_dxkd_flattened_(row, col) = val;
                pd_A_nonzeros.push_back(Eigen::Triplet<real>(row, col, val));
                pd_At_nonzeros.push_back(Eigen::Triplet<real>(col, row, val));
            }

    pd_A_ = ToSparseMatrix(vertex_dim * vertex_dim, vertex_dim * element_dim, pd_A_nonzeros);
    pd_At_ = ToSparseMatrix(vertex_dim * element_dim, vertex_dim * vertex_dim, pd_At_nonzeros);
    pd_AtA_ = pd_At_ * pd_A_;
}

template class FiniteElementSample<2, 3>;
template class FiniteElementSample<2, 4>;
template class FiniteElementSample<3, 4>;
template class FiniteElementSample<3, 8>;