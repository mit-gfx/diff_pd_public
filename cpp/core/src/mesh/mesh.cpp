#include "mesh/mesh.h"
#include "common/file_helper.h"

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name) {
    std::ifstream fin(binary_file_name);
    const int v_dim = Load<int>(fin);
    const int e_dim = Load<int>(fin);
    CheckError(v_dim == vertex_dim && e_dim == element_dim, "Corrupted mesh file: " + binary_file_name);
    const MatrixXr vertices = Load<MatrixXr>(fin);
    const MatrixXi elements = Load<MatrixXi>(fin);
    CheckError(vertices.rows() == vertex_dim && elements.rows() == element_dim, "Inconsistent mesh matrix size.");
    vertices_ = vertices;
    elements_ = elements;

    // Compute element volumne.
    const int element_num = static_cast<int>(elements.cols());
    element_volume_.resize(element_num, 0);
    for (int e = 0; e < element_num; ++e) {
        Eigen::Matrix<real, vertex_dim, element_dim> element;
        for (int k = 0; k < element_dim; ++k) {
            element.col(k) = vertices.col(elements(k, e));
        }
        element_volume_[e] = ComputeElementVolume(element);
    }

    // Compute average element volume.
    average_element_volume_ = 0;
    for (const real& e : element_volume_) average_element_volume_ += e;
    average_element_volume_ /= element_num;
    dx_ = std::pow(average_element_volume_, ToReal(1) / vertex_dim);
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& elements) {
    vertices_ = vertices;
    elements_ = elements;

    // Compute element volumne.
    const int element_num = static_cast<int>(elements.cols());
    element_volume_.resize(element_num, 0);
    for (int e = 0; e < element_num; ++e) {
        Eigen::Matrix<real, vertex_dim, element_dim> element;
        for (int k = 0; k < element_dim; ++k) {
            element.col(k) = vertices.col(elements(k, e));
        }
        element_volume_[e] = ComputeElementVolume(element);
    }

    // Compute average element volume.
    average_element_volume_ = 0;
    for (const real& e : element_volume_) average_element_volume_ += e;
    average_element_volume_ /= element_num;
    dx_ = std::pow(average_element_volume_, ToReal(1) / vertex_dim);
}

template<int vertex_dim, int element_dim>
const real Mesh<vertex_dim, element_dim>::element_volume(const int element_idx) const {
    const int element_num = static_cast<int>(elements_.cols());
    CheckError(0 <= element_idx && element_idx < element_num, "Element index out of range.");
    return element_volume_.at(element_idx);
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::Scale(const real scale_factor) {
    vertices_ *= scale_factor;
    real factor = ToReal(1);
    for (int d = 0; d < vertex_dim; ++d) factor *= scale_factor;
    for (auto& v : element_volume_) {
        v *= factor;
    }
    dx_ *= scale_factor;
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::SaveToFile(const std::string& file_name) const {
    if (EndsWith(file_name, ".bin")) SaveToBinaryFile(file_name);
    else PrintError("Invalid save file name: " + file_name);
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::SaveToBinaryFile(const std::string& binary_file_name) const {
    PrepareToCreateFile(binary_file_name);
    std::ofstream fout(binary_file_name);
    Save<int>(fout, vertex_dim);
    Save<int>(fout, element_dim);
    Save<MatrixXr>(fout, vertices_);
    Save<MatrixXi>(fout, elements_);
}

template<>
const real Mesh<2, 3>::ComputeElementVolume(const Eigen::Matrix<real, 2, 3>& element) const {
    const Vector2r v0 = element.col(0), v1 = element.col(1), v2 = element.col(2);
    const Vector3r a(v1.x() - v0.x(), v1.y() - v0.y(), 0);
    const Vector3r b(v2.x() - v0.x(), v2.y() - v0.y(), 0);
    return a.cross(b).norm() / 2;
}

template<>
const real Mesh<2, 4>::ComputeElementVolume(const Eigen::Matrix<real, 2, 4>& element) const {
    // For this one, we have to assume the elements are ordered as (0, 0), (0, 1), (1, 0), (1, 1).
    // We also assume the quad must be convex.
    const Vector2r v00 = element.col(0), v01 = element.col(1), v10 = element.col(2), v11 = element.col(3);
    // Compute the triangle (v00, v01, v11).
    // Compute the triangle (v00, v10, v11).
    const Vector3r a(v11.x() - v00.x(), v11.y() - v00.y(), 0);
    const Vector3r b(v01.x() - v00.x(), v01.y() - v00.y(), 0);
    const Vector3r c(v10.x() - v00.x(), v10.y() - v00.y(), 0);
    return (a.cross(b).norm() + a.cross(c).norm()) / 2;
}

template<>
const real Mesh<3, 4>::ComputeElementVolume(const Eigen::Matrix<real, 3, 4>& element) const {
    const Vector3r v0 = element.col(0), v1 = element.col(1), v2 = element.col(2), v3 = element.col(3);
    const Vector3r a = v1 - v0, b = v2 - v0, c = v3 - v0;
    return std::fabs(a.dot(b.cross(c)) / 6);
}

template<>
const real Mesh<3, 8>::ComputeElementVolume(const Eigen::Matrix<real, 3, 8>& element) const {
    // For this one, we have to assume the element is a reqular cuboid.
    const Vector3r v000 = element.col(0), v001 = element.col(1), v010 = element.col(2), v100 = element.col(4);
    Matrix3r A;
    A.col(0) = v001 - v000;
    A.col(1) = v010 - v000;
    A.col(2) = v100 - v000;
    return std::fabs(A.determinant());
}

template<>
const real Mesh<2, 3>::dx() const {
    CheckError(false, "dx is not defined for triangle meshes.");
    return 0;
}

template<>
const real Mesh<2, 4>::dx() const {
    return dx_;
}

template<>
const real Mesh<3, 4>::dx() const {
    CheckError(false, "dx is not defined for tet meshes.");
    return 0;
}

template<>
const real Mesh<3, 8>::dx() const {
    return dx_;
}

template<>
const int Mesh<2, 3>::GetNumOfVerticesInFace() { return 2; }

template<>
const int Mesh<2, 4>::GetNumOfVerticesInFace() { return 2; }

template<>
const int Mesh<3, 4>::GetNumOfVerticesInFace() { return 3; }

template<>
const int Mesh<3, 8>::GetNumOfVerticesInFace() { return 4; }

template class Mesh<2, 3>;  // Triangle mesh.
template class Mesh<2, 4>;  // Quad mesh.
template class Mesh<3, 4>;  // Tet mesh.
template class Mesh<3, 8>;  // Hex mesh.