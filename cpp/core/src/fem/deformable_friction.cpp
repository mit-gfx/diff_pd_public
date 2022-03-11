#include "fem/deformable.h"
#include "common/common.h"
#include "friction/planar_frictional_boundary.h"
#include "friction/spherical_frictional_boundary.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetFrictionalBoundary(const std::string& boundary_type,
    const std::vector<real>& params, const std::vector<int> indices) {
    pd_solver_ready_ = false;
    if (boundary_type == "planar") {
        CheckError(static_cast<int>(params.size()) == vertex_dim + 1, "Incompatible parameter number.");
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal(i) = params[i];
        const real offset = params[vertex_dim];
        auto planar = std::make_shared<PlanarFrictionalBoundary<vertex_dim>>();
        planar->Initialize(normal, offset);
        frictional_boundary_ = planar;

        // Check if there are duplicated elements in indices;
        std::set<int> unique_indices;
        for (const int idx : indices) {
            unique_indices.insert(idx);
        }
        CheckError(unique_indices.size() == indices.size(), "Duplicated vertex elements.");
        frictional_boundary_vertex_indices_.clear();
        int cnt = 0;
        for (const int idx : unique_indices) {
            frictional_boundary_vertex_indices_[idx] = cnt;
            ++cnt;
        }
    } else if (boundary_type == "spherical") {
        CheckError(static_cast<int>(params.size()) == vertex_dim + 1, "Incompatible parameter number.");
        Eigen::Matrix<real, vertex_dim, 1> center;
        for (int i = 0; i < vertex_dim; ++i) center(i) = params[i];
        const real radius = params[vertex_dim];
        auto spherical = std::make_shared<SphericalFrictionalBoundary<vertex_dim>>();
        spherical->Initialize(center, radius);
        frictional_boundary_ = spherical;

        // Check if there are duplicated elements in indices;
        std::set<int> unique_indices;
        for (const int idx : indices) {
            unique_indices.insert(idx);
        }
        CheckError(unique_indices.size() == indices.size(), "Duplicated vertex elements.");
        frictional_boundary_vertex_indices_.clear();
        int cnt = 0;
        for (const int idx : unique_indices) {
            frictional_boundary_vertex_indices_[idx] = cnt;
            ++cnt;
        }
    } else {
        CheckError(false, "Unsupported frictional boundary type: " + boundary_type);
    }
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;