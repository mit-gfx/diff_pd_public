#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::GetQuasiStaticState(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
    CheckError(frictional_boundary_vertex_indices_.empty(), "Frictional boundary is not supported in the quasi-static solver.");
    if (BeginsWith(method, "newton")) QuasiStaticStateNewton(method, a, f_ext, options, q);
    else PrintError("Unsupport quasi-static method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::QuasiStaticStateNewton(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
    CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
    CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
    const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_newton_iter > 0, "Invalid max_newton_iter: " + std::to_string(max_newton_iter));
    CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));

    omp_set_num_threads(thread_ct);
    // Governing equation: f_ela(q_next) + f_pd(q_next) + f_act(q_next, u) = -f_ext - f_state(q, v).
    const VectorXr rhs = -f_ext - ForwardStateForce(q, VectorXr::Zero(dofs_));

    VectorXr selected = VectorXr::Ones(dofs_);
    VectorXr q_sol = GetUndeformedShape();
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) {
        selected(pair.first) = 0;
        q_sol(pair.first) = pair.second;
    }
    VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, false) + ActuationForce(q_sol, a);
    // Now q_sol is our initial guess.

    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        VectorXr new_rhs = rhs - force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet_) new_rhs(pair.first) = 0;
        VectorXr dq = VectorXr::Zero(dofs_);
        // Solve for the search direction.
        if (method == "newton_pcg") {
            // Matrix-free operator is only compatible with IdentityPreconditioner and it seems that
            // matrix operators are more accurate.
            Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<real>> cg;
            const SparseMatrix op = QuasiStaticMatrix(q_sol, a);
            cg.compute(op);
            dq = cg.solve(new_rhs);
            CheckError(cg.info() == Eigen::Success, "CG solver failed.");
        } else if (method == "newton_cholesky") {
            // Cholesky.
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            const SparseMatrix op = QuasiStaticMatrix(q_sol, a);
            cholesky.compute(op);
            dq = cholesky.solve(new_rhs);
            CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
        } else {
            // Should never happen.
        }
        if (verbose_level > 0) std::cout << "|dq| = " << dq.norm() << std::endl;

        // Line search.
        real step_size = 1;
        VectorXr q_sol_next = q_sol + step_size * dq;
        VectorXr force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next, false) + ActuationForce(q_sol_next, a);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!HasFlippedElement(q_sol_next) && !force_next.hasNaN()) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next, false) + ActuationForce(q_sol_next, a);
            if (verbose_level > 1) std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
            PrintWarning("Newton's method is using < 1 step size: " + std::to_string(step_size));
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        const VectorXr lhs = force_next;
        const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            q = q_sol_next;
            return;
        }

        // Update.
        q_sol = q_sol_next;
        force_sol = force_next;
    }
    PrintError("Newton's method fails to converge.");
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyGetQuasiStaticState(const std::string& method, const std::vector<real>& a,
    const std::vector<real>& f_ext, const std::map<std::string, real>& options, std::vector<real>& q) const {
    VectorXr q_eig;
    GetQuasiStaticState(method, ToEigenVector(a), ToEigenVector(f_ext), options, q_eig);
    q = ToStdVector(q_eig);
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& a, const VectorXr& dq) const {
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet_) dq_w_bonudary(pair.first) = 0;
    const int mat_w_dofs = NumOfPdElementEnergies();
    const int act_w_dofs = NumOfPdMuscleEnergies();
    VectorXr ret = ElasticForceDifferential(q, dq_w_bonudary) + PdEnergyForceDifferential(q, dq_w_bonudary, VectorXr::Zero(mat_w_dofs))
        + ActuationForceDifferential(q, a, dq_w_bonudary, VectorXr::Zero(act_dofs_), VectorXr::Zero(act_w_dofs));
    for (const auto& pair : dirichlet_) ret(pair.first) = dq(pair.first);
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::QuasiStaticMatrix(const VectorXr& q, const VectorXr& a) const {
    SparseMatrixElements nonzeros = ElasticForceDifferential(q);
    SparseMatrixElements nonzeros_pd, nonzeros_dummy;
    PdEnergyForceDifferential(q, true, false, false, nonzeros_pd, nonzeros_dummy);
    SparseMatrixElements nonzeros_act_dq, nonzeros_act_da, nonzeros_act_dw;
    ActuationForceDifferential(q, a, nonzeros_act_dq, nonzeros_act_da, nonzeros_act_dw);
    nonzeros.insert(nonzeros.end(), nonzeros_pd.begin(), nonzeros_pd.end());
    nonzeros.insert(nonzeros.end(), nonzeros_act_dq.begin(), nonzeros_act_dq.end());
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_.find(row) != dirichlet_.end() || dirichlet_.find(col) != dirichlet_.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, val));
    }
    for (const auto& pair : dirichlet_) nonzeros_new.push_back(Eigen::Triplet<real>(pair.first, pair.first, 1));
    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;