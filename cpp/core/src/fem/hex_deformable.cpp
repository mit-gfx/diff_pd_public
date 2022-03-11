#include "fem/hex_deformable.h"

void HexDeformable::InitializeFiniteElementSamples() {
    const int element_num = mesh().NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    finite_element_samples_.clear();
    finite_element_samples_.resize(element_num, std::vector<FiniteElementSample<3, 8>>(sample_num));

    const real r = ToReal(1 / std::sqrt(3));
    const real dx = mesh().dx();
    const real inv_dx = ToReal(1) / dx;

    Eigen::Matrix<real, 3, 8> undeformed_samples;
    undeformed_samples << 0, 0, 0, 0, 1, 1, 1, 1,
                        0, 0, 1, 1, 0, 0, 1, 1,
                        0, 1, 0, 1, 0, 1, 0, 1;
    undeformed_samples -= Eigen::Matrix<real, 3, 8>::Ones() / 2;
    undeformed_samples *= r;
    undeformed_samples += Eigen::Matrix<real, 3, 8>::Ones() / 2;
    undeformed_samples *= dx;

    for (auto& e : finite_element_samples_) {
        // Initialize grad_undeformed_sample_weight.
        for (int s = 0; s < 8; ++s) {
            Eigen::Matrix<real, 8, 3> grad_undeformed_sample_weight;
            grad_undeformed_sample_weight.setZero();
            // N000(X) = (1 - x / dx) * (1 - y / dx) * (1 - z / dx).
            // N001(X) = (1 - x / dx) * (1 - y / dx) * z / dx.
            // N010(X) = (1 - x / dx) * y / dx * (1 - z / dx).
            // N011(X) = (1 - x / dx) * y / dx * z / dx.
            // N100(X) = x / dx * (1 - y / dx) * (1 - z / dx).
            // N101(X) = x / dx * (1 - y / dx) * z / dx.
            // N110(X) = x / dx * y / dx * (1 - z / dx).
            // N111(X) = x / dx * y / dx * z / dx.

            const real x = undeformed_samples(0, s), y = undeformed_samples(1, s), z = undeformed_samples(2, s);
            const real nx = x * inv_dx, ny = y * inv_dx, nz = z * inv_dx;
            const real cnx = 1 - nx, cny = 1 - ny, cnz = 1 - nz;
            grad_undeformed_sample_weight.row(0) = Vector3r(-inv_dx * cny * cnz, cnx * -inv_dx * cnz, cnx * cny * -inv_dx);
            grad_undeformed_sample_weight.row(1) = Vector3r(-inv_dx * cny * nz, cnx * -inv_dx * nz, cnx * cny * inv_dx);
            grad_undeformed_sample_weight.row(2) = Vector3r(-inv_dx * ny * cnz, cnx * inv_dx * cnz, cnx * ny * -inv_dx);
            grad_undeformed_sample_weight.row(3) = Vector3r(-inv_dx * ny * nz, cnx * inv_dx * nz, cnx * ny * inv_dx);
            grad_undeformed_sample_weight.row(4) = Vector3r(inv_dx * cny * cnz, nx * -inv_dx * cnz, nx * cny * -inv_dx);
            grad_undeformed_sample_weight.row(5) = Vector3r(inv_dx * cny * nz, nx * -inv_dx * nz, nx * cny * inv_dx);
            grad_undeformed_sample_weight.row(6) = Vector3r(inv_dx * ny * cnz, nx * inv_dx * cnz, nx * ny * -inv_dx);
            grad_undeformed_sample_weight.row(7) = Vector3r(inv_dx * ny * nz, nx * inv_dx * nz, nx * ny * inv_dx);

            e[s].Initialize(undeformed_samples.col(s), grad_undeformed_sample_weight);
        }
    }
}