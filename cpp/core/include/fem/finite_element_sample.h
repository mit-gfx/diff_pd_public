#ifndef FEM_FINITE_ELEMENT_SAMPLE_H
#define FEM_FINITE_ELEMENT_SAMPLE_H

#include "common/config.h"

template<int vertex_dim, int element_dim>
class FiniteElementSample {
public:
    FiniteElementSample() {}

    void Initialize(const Eigen::Matrix<real, vertex_dim, 1>& undeformed_sample,
        const Eigen::Matrix<real, element_dim, vertex_dim>& grad_undeformed_sample_weight);

    const Eigen::Matrix<real, vertex_dim, 1>& undeformed_sample() const { return undeformed_sample_; }
    const Eigen::Matrix<real, element_dim, vertex_dim>& grad_undeformed_sample_weight() const {
        return grad_undeformed_sample_weight_;
    }
    const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim>& dF_dxkd_flattened() const {
        return dF_dxkd_flattened_;
    }
    const SparseMatrix& pd_A() const { return pd_A_; }
    const SparseMatrix& pd_At() const {
        return pd_At_;
    }
    const SparseMatrix& pd_AtA() const {
        return pd_AtA_;
    }

private:
    // Shape-function-related data members.
    // The shape of undeformed_samples_ are vertex_dim x sample_dim (returned from GetNumOfSamplesInElement).
    // For tri and tet meshes, sample_dim = 1 and the value of undeformed_samples_ does not matter.
    // For quad meshes, sample_dim = 4.
    // For hex meshes, sample_dim = 8.
    // The Deformable class assumes that the energy density is evaluated at undeformed_samples_ then scaled by
    // their corresponding volumes.
    //
    // To explain the meaning of each data member below, let's consider a single finite element and let Xi and xi be
    // the locations of its corners in the undeformed and deformed space, respectively.
    // Consider \phi(X) where X can be continuously varying in the finite element.
    // For tet and tri meshes:
    // \phi(X0) = A * X0 + b = x0
    // \phi(X1) = A * X1 + b = x1
    // \phi(X2) = A * X2 + b = x2

    // A [X1 - X0, X2 - X0] = [x1 - x0, x2 - x0]
    // A = [x1 - x0, x2 - x0] [X1 - X0, X2 - X0]^{-1}.
    // b is a function of (Xi, xi) but does not really matter any more.
    // The continuous version is x = \phi(X) = A * X. Therefore, F(X) = A = [x1 - x0, x2 - x0] [X1 - X0, X2 - X0]^{-1}.
    // Note that F is constant regardless of the location of the sample points.
    // If we let B = [X1 - X0, X2 - X0]^{-1}, we can rewrite F(X) as:
    // F(X) = (x1 - x0) * B.row(0) + (x2 - x0) * B.row(1)
    //      = x0 * (-B.row(0) - B.row(1)) + x1 * B.row(0) + x2 * B.row(1).
    //      = \sum xi B(Xi)
    //
    // For quad and hex meshes:
    // \phi(X) = \sum_i xi Ni (X; Xi) where Ni is the basis function dependent on Xi only.
    // F(X) = \sum xi grad_X Ni(X; Xi).
    //
    // The matrix undeformed_samples_ records the location of the samples in the material space to evaluate derived
    // quantities like energy density and stress tensor. The dimension is vertex_dim x sample_dim where sample_dim is
    // returned from GetNumOfSamplesInElement() above.
    // For tet and tri meshes, undeformed_samples_ do not matter.
    Eigen::Matrix<real, vertex_dim, 1> undeformed_sample_;

    // As you can see, in both cases, for a sample Xs, F(Xs) can be written as a matrix matrix product between
    // [x0, x1, x2, ...] and a element_dim x vertex_dim matrix dependent on Xi only. We can it grad_undeformed_sample_weights_.
    // Note that even for tet and tri, it is still understandable to call it "grad" because B can be understood as grad Ni
    // where Ni is a triangular basis function.
    Eigen::Matrix<real, element_dim, vertex_dim> grad_undeformed_sample_weight_;

    // Let Xs be a sample. The energy is written as:
    // E = \sum_s \Psi(Xs) dXs = \sum_s \Psi(F(Xs)) dXs.
    // Therefore, it only needs to know how to compute F.
    // Going one step further, the elastic force is re-written as:
    // fi = -dE/dxi = -\sum_s P(Xs) : dF(Xs) / dxi dXs.
    // Now you can see that we need to compute dF(Xs) / dxi.
    // This is the role of dF_dxkd_flattened_.
    // This is essentially the flattened version of grad_undeformed_sample_weights_.
    // F(Xs) = xi * grad_undeformed_sample_weights_[s].
    // dF_dxkd_flattened_[s] is of size (vertex_dim * vertex_dim) x (vertex_dim * element_dim).
    // dF_dxkd_flattened_[s](pq, ij) = dFpq / dxij.
    // The orders are: Fpq: F00, F10, F20, F01, F11, F21, ...
    // xij: x0(x), x0(y), x0(z), x1(x), x1(y), x1(z), ...
    Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> dF_dxkd_flattened_;

    // PD-related elements.
    // A is simply a flattened, sparse version of grad_undeformed_sample_weight_.
    // grad_undeformed_sample_weight_, dF_dxkd_flattened_, and pd_A_ are intentionally made duplicated.
    //
    // F = [x0, x1, x2, ...] * grad_undeformed_sample_weight_.
    // Flatten(F) = A * concatenate([x0, x1, x2, ...]).
    SparseMatrix pd_A_;     // (vertex_dim * vertex_dim) * (vertex_dim * element_dim).
    SparseMatrix pd_At_;    // (vertex_dim * element_dim) * (vertex_dim * vertex_dim).
    SparseMatrix pd_AtA_;   // (vertex_dim * element_dim) * (vertex_dim * element_dim).
};

#endif