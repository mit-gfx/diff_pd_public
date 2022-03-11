#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "solver/matrix_op.h"
#include "material/linear.h"
#include "material/corotated.h"
#include "material/neohookean.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
Deformable<vertex_dim, element_dim>::Deformable()
    : mesh_(), density_(0), element_volume_(0), material_(nullptr), dofs_(0), pd_solver_ready_(false), act_dofs_(0),
    frictional_boundary_(nullptr) {}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(binary_file_name);
    InitializeAfterMesh(density, material_type, youngs_modulus, poissons_ratio);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& elements, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, elements);
    InitializeAfterMesh(density, material_type, youngs_modulus, poissons_ratio);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::InitializeAfterMesh(const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    density_ = density;
    element_volume_ = mesh_.average_element_volume();
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
    InitializeFiniteElementSamples();
    pd_solver_ready_ = false;
}

template<int vertex_dim, int element_dim>
const std::shared_ptr<Material<vertex_dim>> Deformable<vertex_dim, element_dim>::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material<vertex_dim>> material(nullptr);
    if (material_type == "linear") {
        material = std::make_shared<LinearMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "neohookean") {
        material = std::make_shared<NeohookeanMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "none") {
        material = nullptr;
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Forward(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
    std::vector<int>& active_contact_idx) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "pd")) ForwardProjectiveDynamics(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "newton")) ForwardNewton(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else PrintError("Unsupported forward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
    VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const {
    if (method == "semi_implicit")
        BackwardSemiImplicit(q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else if (BeginsWith(method, "pd"))
        BackwardProjectiveDynamics(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else if (BeginsWith(method, "newton"))
        BackwardNewton(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else
        PrintError("Unsupported backward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent q size. " + std::to_string(q.size())
        + " != " + std::to_string(dofs_));
    Mesh<vertex_dim, element_dim> mesh;
    mesh.Initialize(Eigen::Map<const MatrixXr>(q.data(), vertex_dim, dofs_ / vertex_dim), mesh_.elements());
    mesh.SaveToFile(obj_file_name);
}

// For Python binding.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
    std::vector<real>& q_next, std::vector<real>& v_next, std::vector<int>& active_contact_idx) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, options, q_next_eig, v_next_eig,
        active_contact_idx);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next,
    const std::vector<real>& v_next, const std::vector<int>& active_contact_idx,
    const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
    const std::map<std::string, real>& options,
    std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_da, std::vector<real>& dl_df_ext,
    std::vector<real>& dl_dmat_w, std::vector<real>& dl_dact_w, std::vector<real>& dl_dstate_p) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dmat_w_eig, dl_dact_w_eig, dl_dstate_p_eig;
    Backward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, ToEigenVector(q_next),
        ToEigenVector(v_next), active_contact_idx, ToEigenVector(dl_dq_next), ToEigenVector(dl_dv_next), options,
        dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dmat_w_eig, dl_dact_w_eig, dl_dstate_p_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_da = ToStdVector(dl_da_eig);
    dl_df_ext = ToStdVector(dl_df_ext_eig);
    dl_dmat_w = ToStdVector(dl_dmat_w_eig);
    dl_dact_w = ToStdVector(dl_dact_w_eig);
    dl_dstate_p = ToStdVector(dl_dstate_p_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyElasticEnergy(const std::vector<real>& q) const {
    return ElasticEnergy(ToEigenVector(q));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForce(const std::vector<real>& q) const {
    return ToStdVector(ElasticForce(ToEigenVector(q)));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q, const std::vector<real>& dq) const {
    return ToStdVector(ElasticForceDifferential(ToEigenVector(q), ToEigenVector(dq)));
}

template<int vertex_dim, int element_dim>
const std::vector<std::vector<real>> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q) const {
    PrintWarning("PyElasticForceDifferential should only be used for small-scale problems and for testing purposes.");
    const SparseMatrixElements nonzeros = ElasticForceDifferential(ToEigenVector(q));
    std::vector<std::vector<real>> K(dofs_, std::vector<real>(dofs_, 0));
    for (const auto& triplet : nonzeros) {
        K[triplet.row()][triplet.col()] += triplet.value();
    }
    return K;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ElasticEnergy(const VectorXr& q) const {
    if (!material_) return 0;

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<real> element_energy(element_num, 0);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            element_energy[i] += material_->EnergyDensity(F) * element_volume_ / sample_num;
        }
    }
    real energy = 0;
    for (const real e : element_energy) energy += e;
    return energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForce(const VectorXr& q) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<Eigen::Matrix<real, vertex_dim, 1>> f_ints(element_num * element_dim,
        Eigen::Matrix<real, vertex_dim, 1>::Zero());

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            const auto P = material_->StressTensor(F);
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> f_kd =
                -Flatten(P).transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k) {
                f_ints[i * element_dim + k] += Eigen::Matrix<real, vertex_dim, 1>(f_kd.segment(k * vertex_dim, vertex_dim));
            }
        }
    }

    VectorXr f_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_num; ++i) {
        const auto vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                f_int(vi(j) * vertex_dim + k) += f_ints[i * element_dim + j](k);
            }
        }
    }

    return f_int;
}

template<int vertex_dim, int element_dim>
const SparseMatrixElements Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q) const {
    if (!material_) return SparseMatrixElements();

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    // The sequential version:
    // SparseMatrixElements nonzeros;
    SparseMatrixElements nonzeros(element_num * sample_num * element_dim * vertex_dim * element_dim * vertex_dim);

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
            for (int s = 0; s < element_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(t) * finite_element_samples_[i][j].grad_undeformed_sample_weight().row(s);
                    dF.col(s * vertex_dim + t) += Flatten(dF_single);
                }
            const auto dP = material_->StressTensorDifferential(F) * dF;
            const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kd =
                -dP.transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    for (int s = 0; s < element_dim; ++s)
                        for (int t = 0; t < vertex_dim; ++t)
                            nonzeros[i * sample_num * element_dim * vertex_dim * element_dim * vertex_dim
                                + j * element_dim * vertex_dim * element_dim * vertex_dim
                                + k * vertex_dim * element_dim * vertex_dim
                                + d * element_dim * vertex_dim
                                + s * vertex_dim
                                + t] = Eigen::Triplet<real>(vertex_dim * vi(k) + d, vertex_dim * vi(s) + t,
                                    df_kd(s * vertex_dim + t, k * vertex_dim + d));
                            // Below is the sequential version:
                            // nonzeros.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                            //     vertex_dim * vi(s) + t, df_kd(s * vertex_dim + t, k * vertex_dim + d)));
        }
    }
    return nonzeros;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<Eigen::Matrix<real, vertex_dim, 1>> df_ints(element_num * element_dim,
        Eigen::Matrix<real, vertex_dim, 1>::Zero());

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        const auto ddeformed = ScatterToElement(dq, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            const auto dF = DeformationGradient(i, ddeformed, j);
            const auto dP = material_->StressTensorDifferential(F, dF);
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> df_kd =
                -Flatten(dP).transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k) {
                df_ints[i * element_dim + k] += Eigen::Matrix<real, vertex_dim, 1>(df_kd.segment(k * vertex_dim, vertex_dim));
            }
        }
    }

    VectorXr df_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_num; ++i) {
        const auto vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                df_int(vi(j) * vertex_dim + k) += df_ints[i * element_dim + j](k);
            }
        }
    }

    return df_int;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::GetUndeformedShape() const {
    VectorXr q = VectorXr::Zero(dofs_);
    const int vertex_num = mesh_.NumOfVertices();
    for (int i = 0; i < vertex_num; ++i) q.segment(vertex_dim * i, vertex_dim) = mesh_.vertex(i);
    for (const auto& pair : dirichlet_) {
        q(pair.first) = pair.second;
    }
    return q;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, element_dim> Deformable<vertex_dim, element_dim>::ScatterToElement(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    Eigen::Matrix<real, vertex_dim, element_dim> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.col(j) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim * element_dim, 1> Deformable<vertex_dim, element_dim>::ScatterToElementFlattened(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.segment(j * vertex_dim, vertex_dim) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> Deformable<vertex_dim, element_dim>::DeformationGradient(
    const int element_idx, const Eigen::Matrix<real, vertex_dim, element_dim>& q, const int sample_idx) const {
    return q * finite_element_samples_[element_idx][sample_idx].grad_undeformed_sample_weight();
}

template<int vertex_dim, int element_dim>
const bool Deformable<vertex_dim, element_dim>::HasFlippedElement(const VectorXr& q) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent number of elements.");
    const int sample_num = element_dim;
    for (int i = 0; i < mesh_.NumOfElements(); ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            if (F.determinant() < std::numeric_limits<real>::epsilon()) return true;
        }
    }
    return false;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AssignToGlobalDeformable() const {
    global_deformable = this;
    global_vertex_dim = vertex_dim;
    global_element_dim = element_dim;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ClearGlobalDeformable() const {
    global_deformable = nullptr;
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

// Initialize the global variable used for the preconditioner.
const void* global_deformable = nullptr;
int global_vertex_dim = 0;
int global_element_dim = 0;
std::map<int, real> global_additional_dirichlet_boundary = std::map<int, real>();
std::string global_pd_backward_method = "";