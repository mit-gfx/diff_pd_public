#include "fem/tri_deformable.h"

void TriDeformable::InitializeFiniteElementSamples() {
    const int element_num = mesh().NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    finite_element_samples_.clear();
    finite_element_samples_.resize(element_num, std::vector<FiniteElementSample<2, 3>>(sample_num));

    for (int e_idx = 0; e_idx < element_num; ++e_idx) {
        // The actual sample does not matter.
        const Vector2r undeformed_sample(0, 0);
        Eigen::Matrix<real, 3, 2> grad_undeformed_sample_weight;
        grad_undeformed_sample_weight.setZero();
        // A [X1 - X0, X2 - X0] = [x1 - x0, x2 - x0]
        // A = [x1 - x0, x2 - x0] [X1 - X0, X2 - X0]^{-1}.
        // b is a function of (Xi, xi) but does not really matter any more.
        // The continuous version is x = \phi(X) = A * X. Therefore, F(X) = A = [x1 - x0, x2 - x0] [X1 - X0, X2 - X0]^{-1}.
        // Note that F is constant regardless of the location of the sample points.
        // If we let B = [X1 - X0, X2 - X0]^{-1}, we can rewrite F(X) as:
        // F(X) = (x1 - x0) * B.row(0) + (x2 - x0) * B.row(1)
        //      = x0 * (-B.row(0) - B.row(1)) + x1 * B.row(0) + x2 * B.row(1).
        //      = \sum xi B(Xi)
        // mesh() holds the undeformed mesh.
        const Vector2r X0 = mesh().vertex(mesh().element(e_idx)(0));
        const Vector2r X1 = mesh().vertex(mesh().element(e_idx)(1));
        const Vector2r X2 = mesh().vertex(mesh().element(e_idx)(2));
        Matrix2r B_inv;
        B_inv.col(0) = X1 - X0;
        B_inv.col(1) = X2 - X0;
        const Matrix2r B = B_inv.inverse();
        grad_undeformed_sample_weight.row(0) = -B.row(0) - B.row(1);
        grad_undeformed_sample_weight.row(1) = B.row(0);
        grad_undeformed_sample_weight.row(2) = B.row(1);

        auto& e = finite_element_samples_[e_idx];
        e[0].Initialize(undeformed_sample, grad_undeformed_sample_weight);
    }
}