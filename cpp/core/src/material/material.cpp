#include "material/material.h"

template<int dim>
Material<dim>::Material()
    : youngs_modulus_(0), poissons_ratio_(0), lambda_(0), mu_(0) {}

template<int dim>
void Material<dim>::Initialize(const real youngs_modulus, const real poissons_ratio) {
    youngs_modulus_ = youngs_modulus;
    poissons_ratio_ = poissons_ratio;
    const real k = youngs_modulus_;
    const real nu = poissons_ratio_;
    lambda_ = k * nu / ((1 + nu) * (1 - 2 * nu));
    mu_ = k / (2 * (1 + nu));
}

template class Material<2>;
template class Material<3>;