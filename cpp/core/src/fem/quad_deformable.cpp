#include "fem/quad_deformable.h"

void QuadDeformable::InitializeFiniteElementSamples() {
    const int element_num = mesh().NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    finite_element_samples_.clear();
    finite_element_samples_.resize(element_num, std::vector<FiniteElementSample<2, 4>>(sample_num));

    const real r = ToReal(1 / std::sqrt(3));
    const real dx = mesh().dx();
    const real inv_dx = ToReal(1) / dx;

    Eigen::Matrix<real, 2, 4> undeformed_samples;
    undeformed_samples << 0, 0, 1, 1,
                        0, 1, 0, 1;
    undeformed_samples -= Eigen::Matrix<real, 2, 4>::Ones() / 2;
    undeformed_samples *= r;
    undeformed_samples += Eigen::Matrix<real, 2, 4>::Ones() / 2;
    undeformed_samples *= dx;

    for (auto& e : finite_element_samples_) {
        // Initialize grad_undeformed_sample_weight.
        for (int s = 0; s < 4; ++s) {
            Eigen::Matrix<real, 4, 2> grad_undeformed_sample_weight;
            grad_undeformed_sample_weight.setZero();
            // N00(X) = (1 - x / dx) * (1 - y / dx).
            // N01(X) = (1 - x / dx) * y / dx.
            // N10(X) = x / dx * (1 - y / dx).
            // N11(X) = x / dx * y / dx.

            const real x = undeformed_samples(0, s), y = undeformed_samples(1, s);
            const real nx = x * inv_dx, ny = y * inv_dx;
            const real cnx = 1 - nx, cny = 1 - ny;
            grad_undeformed_sample_weight.row(0) = Vector2r(-inv_dx * cny, cnx * -inv_dx);
            grad_undeformed_sample_weight.row(1) = Vector2r(-inv_dx * ny, cnx * inv_dx);
            grad_undeformed_sample_weight.row(2) = Vector2r(inv_dx * cny, nx * -inv_dx);
            grad_undeformed_sample_weight.row(3) = Vector2r(inv_dx * ny, nx * inv_dx);

            e[s].Initialize(undeformed_samples.col(s), grad_undeformed_sample_weight);
        }
    }
}