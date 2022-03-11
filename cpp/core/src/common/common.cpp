#include "common/common.h"
#include "common/exception_with_call_stack.h"
#include "common/file_helper.h"

const real ToReal(const double v) {
    return static_cast<real>(v);
}

const double ToDouble(const real v) {
    return static_cast<double>(v);
}

const std::string GreenHead() {
    return "\x1b[6;30;92m";
}

const std::string RedHead() {
    return "\x1b[6;30;91m";
}

const std::string YellowHead() {
    return "\x1b[6;30;93m";
}

const std::string CyanHead() {
    return "\x1b[6;30;96m";
}

const std::string GreenTail() {
    return "\x1b[0m";
}

const std::string RedTail() {
    return "\x1b[0m";
}

const std::string YellowTail() {
    return "\x1b[0m";
}

const std::string CyanTail() {
    return "\x1b[0m";
}

void PrintError(const std::string& message, const int return_code) {
#if PRINT_LEVEL >= PRINT_ERROR
    std::cerr << RedHead() << message << RedTail() << std::endl;
    throw return_code;
#endif
}

void PrintWarning(const std::string& message) {
#if PRINT_LEVEL >= PRINT_ERROR_AND_WARNING
    std::cout << YellowHead() << message << YellowTail() << std::endl;
#endif
}

void PrintInfo(const std::string& message) {
#if PRINT_LEVEL >= PRINT_ALL
    std::cout << CyanHead() << message << CyanTail() << std::endl;
#endif
}

void PrintSuccess(const std::string& message) {
    std::cout << GreenHead() << message << GreenTail() << std::endl;
}

// Timing.
static std::stack<timeval> t_begins;

void Tic() {
    timeval t_begin;
    gettimeofday(&t_begin, nullptr);
    t_begins.push(t_begin);
}

void Toc(const std::string& message) {
    timeval t_end;
    gettimeofday(&t_end, nullptr);
    timeval t_begin = t_begins.top();
    const real t_interval = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_usec - t_begin.tv_usec) / 1e6;
    std::cout << CyanHead() << "[Timing] " << message << ": " << t_interval << "s"
              << CyanTail() << std::endl;
    t_begins.pop();
}

void CheckError(const bool condition, const std::string& error_message) {
#if PRINT_LEVEL >= PRINT_ERROR
    if (!condition) {
        std::cerr << RedHead() << error_message << RedTail() << std::endl;
        throw ExceptionWithCallStack((RedHead() + error_message + RedTail()).c_str());
    }
#endif
}

// Debugging.
void PrintNumpyStyleMatrix(const MatrixXr& mat) {
    if (!mat.size()) {
        std::cout << "mat = np.array([[]])" << std::endl;
        return;
    }
    const int n_row = static_cast<int>(mat.rows());
    const int n_col = static_cast<int>(mat.cols());
    std::cout << "mat = np.array([" << std::endl;
    for (int i = 0; i < n_row; ++i) {
        std::cout << "\t\t\t[";
        for (int j = 0; j < n_col; ++j) {
            std::cout << mat(i, j) << (j == n_col - 1 ? "" : ", ");
        }
        std::cout << (i == n_row - 1 ? "]" : "],") << std::endl;
    }
    std::cout << "])" << std::endl;
}

void PrintNumpyStyleVector(const VectorXr& vec) {
    std::cout << "vec = np.array([";
    const int n = static_cast<int>(vec.size());
    for (int i = 0; i < n; ++i) {
        std::cout << vec(i) << (i == n - 1 ? "" : ", ");
    }
    std::cout << "])" << std::endl;
}

const real Clip(const real val, const real min, const real max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

const real ClipWithGradient(const real val, const real min, const real max, real& grad) {
    if (val < min) {
        grad = 0.0;
        return min;
    }
    if (val > max) {
        grad = 0.0;
        return max;
    }
    grad = 1.0;
    return val;
}

const real Pi() {
    return ToReal(3.1415926535897932384626);
}

const std::vector<real> ToStdVector(const VectorXr& v) {
    return std::vector<real>(v.data(), v.data() + v.size());
}

const VectorXr ToEigenVector(const std::vector<real>& v) {
    return Eigen::Map<const VectorXr>(v.data(), v.size());
}

const bool BeginsWith(const std::string& full, const std::string& beginning) {
    return full.length() >= beginning.length() &&
        full.compare(0, beginning.length(), beginning) == 0;
}

const bool EndsWith(const std::string& full, const std::string& ending) {
    return full.length() >= ending.length() &&
        full.compare(full.length() - ending.length(), ending.length(), ending) == 0;
}

const std::set<int> VectorToSet(const std::vector<int>& v) {
    return std::set<int>(v.begin(), v.end());
}

const std::vector<int> SetToVector(const std::set<int>& s) {
    return std::vector<int>(s.begin(), s.end());
}

const bool SameSet(const std::set<int>& a, const std::set<int>& b) {
    if (a.size() != b.size()) return false;
    for (const int e : a)
        if (b.find(e) == b.end()) return false;
    return true;
}

const bool ProposeNewSet(const std::set<int>& a, const std::set<int>& b, std::set<int>& new_set) {
    std::set<int> a_b_inter, a_comp, b_comp;
    // a_b_inter = a /\ b.
    // a_comp = a - a_b_inter.
    // b_comp = b - a_b_inter.
    for (const int e : a) {
        if (b.find(e) != b.end()) a_b_inter.insert(e);
        else a_comp.insert(e);
    }
    for (const int e : b) {
        if (a_b_inter.find(e) == a_b_inter.end()) b_comp.insert(e);
    }
    // Propose a new set by a_b_inter \/ subset of a_comp \/ subset of b_comp.
    const int a_comp_size = static_cast<int>(a_comp.size());
    const int b_comp_size = static_cast<int>(b_comp.size());
    new_set.clear();
    for (const int e : a_b_inter) new_set.insert(e);
    if (a_comp_size <= 1 && b_comp_size <= 1) return false;
    const int a_half_comp_size = a_comp_size / 2;
    const int b_half_comp_size = b_comp_size / 2;
    int idx = 0;
    for (const int e : a_comp) {
        if (idx == a_half_comp_size) break;
        else {
            new_set.insert(e);
            ++idx;
        }
    }
    idx = 0;
    for (const int e : b_comp) {
        if (idx == b_half_comp_size) break;
        else {
            new_set.insert(e);
            ++idx;
        }
    }
    return true;
}

const SparseMatrixElements FromSparseMatrix(const SparseMatrix& A) {
    SparseMatrixElements nonzeros;
    for (int k = 0; k < A.outerSize(); ++k)
        for (SparseMatrix::InnerIterator it(A, k); it; ++it)
            nonzeros.push_back(Eigen::Triplet<real>(it.row(), it.col(), it.value()));
    return nonzeros;
}

const SparseMatrix ToSparseMatrix(const int row, const int col, const SparseMatrixElements& nonzeros) {
    SparseMatrix A(row, col);
    A.setFromTriplets(nonzeros.begin(), nonzeros.end());
    return A;
}

void SaveSparseMatrixToBinaryFile(const SparseMatrix& A, const std::string& file_name) {
    std::ofstream fout(file_name);
    const SparseMatrixElements nonzeros = FromSparseMatrix(A);
    Save<int>(fout, static_cast<int>(A.rows()));
    Save<int>(fout, static_cast<int>(A.cols()));
    const int nonzeros_num = static_cast<int>(nonzeros.size());
    Save<int>(fout, nonzeros_num);
    for (const auto& triplet : nonzeros) {
        Save<int>(fout, triplet.row());
        Save<int>(fout, triplet.col());
        Save<real>(fout, triplet.value());
    }
}

const SparseMatrix LoadSparseMatrixFromBinaryFile(const std::string& file_name) {
    std::ifstream fin(file_name);
    const int rows = Load<int>(fin);
    const int cols = Load<int>(fin);
    const int nonzeros_num = Load<int>(fin);
    SparseMatrixElements nonzeros(nonzeros_num);
    for (int i = 0; i < nonzeros_num; ++i) {
        const int row = Load<int>(fin);
        const int col = Load<int>(fin);
        const real value = Load<real>(fin);
        nonzeros[i] = Eigen::Triplet<real>(row, col, value);
    }
    return ToSparseMatrix(rows, cols, nonzeros);
}

void SaveEigenVectorToBinaryFile(const VectorXr& v, const std::string& file_name) {
    std::ofstream fout(file_name);
    const int n = static_cast<int>(v.size());
    Save<int>(fout, n);
    for (int i = 0; i < n; ++i) Save<real>(fout, v(i));
}

const VectorXr LoadEigenVectorFromBinaryFile(const std::string& file_name) {
    std::ifstream fin(file_name);
    const int n = Load<int>(fin);
    VectorXr v = VectorXr::Zero(n);
    for (int i = 0; i < n; ++i) v(i) = Load<real>(fin);
    return v;
}

void SaveEigenMatrixToBinaryFile(const MatrixXr& A, const std::string& file_name) {
    std::ofstream fout(file_name);
    const int row_num = static_cast<int>(A.rows());
    const int col_num = static_cast<int>(A.cols());
    Save<int>(fout, row_num);
    Save<int>(fout, col_num);
    for (int i = 0; i < row_num; ++i)
        for (int j = 0; j < col_num; ++j)
            Save<double>(fout, ToDouble(A(i, j)));
}

const MatrixXr LoadEigenMatrixFromBinaryFile(const std::string& file_name) {
    std::ifstream fin(file_name);
    const int row_num = Load<int>(fin);
    const int col_num = Load<int>(fin);
    MatrixXr A = MatrixXr::Zero(row_num, col_num);
    for (int i = 0; i < row_num; ++i)
        for (int j = 0; j < col_num; ++j)
            A(i, j) = ToReal(Load<double>(fin));
    return A;
}

const VectorXr VectorSparseMatrixProduct(const VectorXr& v,
    const int row, const int col, const SparseMatrixElements& A) {
    VectorXr vA = VectorXr::Zero(col);
    const int nonzeros = static_cast<int>(A.size());
    for (int i = 0; i < nonzeros; ++i) {
        const int r = A[i].row();
        const int c = A[i].col();
        const real a = A[i].value();
        vA[c] += v(r) * a;
    }
    return vA;
}

const MatrixXr SparseMatrixMatrixProduct(const int row, const int col,
    const SparseMatrixElements& A, const MatrixXr& B) {
    const int b_col = static_cast<int>(B.cols());
    MatrixXr AB = MatrixXr::Zero(row, b_col);
    #pragma omp parallel for
    for (int j = 0; j < b_col; ++j) {
        for (const auto& triplet : A) {
            AB(triplet.row(), j) += triplet.value() * B(triplet.col(), j);
        }
    }
    return AB;
}

const MatrixXr MatrixMatrixProduct(const MatrixXr& A, const MatrixXr& B) {
    const int A_rows = static_cast<int>(A.rows());
    const int B_cols = static_cast<int>(B.cols());
    MatrixXr AB(A_rows, B_cols);
    AB.setZero();
    // Parallelize B_cols.
    #pragma omp parallel for
    for (int j = 0; j < B_cols; ++j) AB.col(j) = A * B.col(j);
    return AB;
}

const int GetNonzeros(const MatrixXr& A, const real eps) {
    const int row_num = static_cast<int>(A.rows());
    const int col_num = static_cast<int>(A.cols());
    int cnt = 0;
    for (int i = 0; i < row_num; ++i)
        for (int j = 0; j < col_num; ++j) {
            if (A(i, j) < -eps || A(i, j) > eps) {
                ++cnt;
            }
        }
    return cnt;
}