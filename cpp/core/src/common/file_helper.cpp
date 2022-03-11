#include "common/file_helper.h"
#include <sys/stat.h>
#include "common/common.h"

template<>
void Save<int>(std::ofstream& f, const int& val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(int));
}

template<>
void Save<real>(std::ofstream& f, const real& val) {
    const double val_db = ToDouble(val);
    f.write(reinterpret_cast<const char*>(&val_db), sizeof(double));
}

template<>
void Save<bool>(std::ofstream& f, const bool& val) {
    const char ch = static_cast<char>(val);
    f.write(reinterpret_cast<const char*>(&ch), sizeof(char));
}

template<>
void Save<Vector2r>(std::ofstream& f, const Vector2r& val) {
    Save<real>(f, val[0]); Save<real>(f, val[1]);
}

template<>
void Save<Vector3r>(std::ofstream& f, const Vector3r& val) {
    Save<real>(f, val[0]); Save<real>(f, val[1]); Save<real>(f, val[2]);
}

template<>
void Save<Vector2i>(std::ofstream& f, const Vector2i& val) {
    Save<int>(f, val[0]); Save<int>(f, val[1]);
}

template<>
void Save<Vector3i>(std::ofstream& f, const Vector3i& val) {
    Save<int>(f, val[0]); Save<int>(f, val[1]); Save<int>(f, val[2]);
}

template<>
void Save<Matrix2r>(std::ofstream& f, const Matrix2r& val) {
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) Save<real>(f, val(i, j));
}

template<>
void Save<Matrix3r>(std::ofstream& f, const Matrix3r& val) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Save<real>(f, val(i, j));
}

template<>
void Save<std::array<real, 2>>(std::ofstream& f, const std::array<real, 2>& val) {
    for (int i = 0; i < 2; ++i) Save<real>(f, val[i]);
}

template<>
void Save<std::array<real, 3>>(std::ofstream& f, const std::array<real, 3>& val) {
    for (int i = 0; i < 3; ++i) Save<real>(f, val[i]);
}

template<>
void Save<std::array<std::array<real, 2>, 2>>(std::ofstream& f, const std::array<std::array<real, 2>, 2>& val) {
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) Save<real>(f, val[i][j]);
}

template<>
void Save<std::array<std::array<real, 3>, 3>>(std::ofstream& f, const std::array<std::array<real, 3>, 3>& val) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Save<real>(f, val[i][j]);
}

template<>
void Save<MatrixXr>(std::ofstream& f, const MatrixXr& val) {
    const int row = static_cast<int>(val.rows());
    const int col = static_cast<int>(val.cols());
    Save<int>(f, row);
    Save<int>(f, col);
    for (int i = 0; i < row; ++i) for (int j = 0; j < col; ++j) Save<real>(f, val(i, j));
}

template<>
void Save<MatrixXi>(std::ofstream& f, const MatrixXi& val) {
    const int row = static_cast<int>(val.rows());
    const int col = static_cast<int>(val.cols());
    Save<int>(f, row);
    Save<int>(f, col);
    for (int i = 0; i < row; ++i) for (int j = 0; j < col; ++j) Save<int>(f, val(i, j));
}

template<>
void Save<VectorXr>(std::ofstream& f, const VectorXr& val) {
    const int len = static_cast<int>(val.size());
    Save<int>(f, len);
    for (int i = 0; i < len; ++i) Save<real>(f, val(i));
}

template<>
const int Load<int>(std::ifstream& f) {
    int val = 0;
    f.read(reinterpret_cast<char*>(&val), sizeof(int));
    return val;
}

template<>
const real Load<real>(std::ifstream& f) {
    double val = 0;
    f.read(reinterpret_cast<char*>(&val), sizeof(double));
    return ToReal(val);
}

template<>
const bool Load<bool>(std::ifstream& f) {
    char val = 0;
    f.read(reinterpret_cast<char*>(&val), sizeof(char));
    return static_cast<bool>(val);
}

template<>
const Vector2r Load<Vector2r>(std::ifstream& f) {
    Vector2r val; val[0] = Load<real>(f); val[1] = Load<real>(f); return val;
}

template<>
const Vector3r Load<Vector3r>(std::ifstream& f) {
    Vector3r val; val[0] = Load<real>(f); val[1] = Load<real>(f); val[2] = Load<real>(f); return val;
}

template<>
const Vector2i Load<Vector2i>(std::ifstream& f) {
    Vector2i val; val[0] = Load<int>(f); val[1] = Load<int>(f); return val;
}

template<>
const Vector3i Load<Vector3i>(std::ifstream& f) {
    Vector3i val; val[0] = Load<int>(f); val[1] = Load<int>(f); val[2] = Load<int>(f); return val;
}

template<>
const Matrix2r Load<Matrix2r>(std::ifstream& f) {
    Matrix2r val; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) val(i, j) = Load<real>(f); return val;
}

template<>
const Matrix3r Load<Matrix3r>(std::ifstream& f) {
    Matrix3r val; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) val(i, j) = Load<real>(f); return val;
}

template<>
const std::array<real, 2> Load<std::array<real, 2>>(std::ifstream& f) {
    std::array<real, 2> val; for (int i = 0; i < 2; ++i) val[i] = Load<real>(f); return val;
}

template<>
const std::array<real, 3> Load<std::array<real, 3>>(std::ifstream& f) {
    std::array<real, 3> val; for (int i = 0; i < 3; ++i) val[i] = Load<real>(f); return val;
}

template<>
const std::array<std::array<real, 2>, 2> Load<std::array<std::array<real, 2>, 2>>(std::ifstream& f) {
    std::array<std::array<real, 2>, 2> val; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) val[i][j] = Load<real>(f); return val;
}

template<>
const std::array<std::array<real, 3>, 3> Load<std::array<std::array<real, 3>, 3>>(std::ifstream& f) {
    std::array<std::array<real, 3>, 3> val; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) val[i][j] = Load<real>(f); return val;
}

template<>
const MatrixXr Load<MatrixXr>(std::ifstream& f) {
    const int row = Load<int>(f);
    const int col = Load<int>(f);
    MatrixXr val(row, col);
    for (int i = 0; i < row; ++i) for (int j = 0; j < col; ++j) val(i, j) = Load<real>(f);
    return val;
}

template<>
const MatrixXi Load<MatrixXi>(std::ifstream& f) {
    const int row = Load<int>(f);
    const int col = Load<int>(f);
    MatrixXi val(row, col);
    for (int i = 0; i < row; ++i) for (int j = 0; j < col; ++j) val(i, j) = Load<int>(f);
    return val;
}

template<>
const VectorXr Load<VectorXr>(std::ifstream& f) {
    const int len = Load<int>(f);
    VectorXr val(len);
    for (int i = 0; i < len; ++i) val(i) = Load<real>(f);
    return val;
}

const std::string RegularizeFilePath(const std::string& path) {
    std::string new_path = "";
    bool in_slash = false;
    for (const char ch : path) {
        const bool is_slash = (ch == '\\' || ch == '/');
        if (!is_slash) {
            new_path += ch;
            in_slash = false;
        } else if (!in_slash) {
            new_path += '/';
            in_slash = true;
        }
    }
    return new_path;
}

const std::string AppendFileToPath(const std::string& folder, const std::string& file_name) {
    return RegularizeFilePath(folder + "/" + file_name);
}

const std::string AppendFolderToPath(const std::string& folder, const std::string& subfolder) {
    return RegularizeFilePath(folder + "/" + subfolder);
}

const std::string GetParentFolder(const std::string path) {
    std::string reg_path = RegularizeFilePath(path);
    // Stop if it is already the root.
    if (reg_path == "/") return reg_path;
    // Ignore the trailing slash.
    if (reg_path.back() == '/')
        reg_path = reg_path.substr(0, reg_path.size() - 1);
    const auto idx = reg_path.find_last_of('/');
    if (idx == std::string::npos) return "./";
    else return reg_path.substr(0, idx);
}

void PrepareToCreateFile(const std::string& file_path) {
    const std::string reg_file_path = RegularizeFilePath(file_path);
    const std::size_t found = reg_file_path.rfind("/");
    if (found != std::string::npos) {
        const std::string folder_name = reg_file_path.substr(0, found + 1);
        size_t pos = 0;
        do {
            pos = folder_name.find_first_of('/', pos + 1);
            mkdir(folder_name.substr(0, pos).c_str(), S_IRWXU);
        } while (pos != std::string::npos);
    }
    std::ofstream fout(reg_file_path, std::ios::out);
    if (!fout.is_open()) {
        std::cerr << RedHead() << "PrepareToCreateFile: did not create file " << reg_file_path
                  << " successfully." << RedTail() << std::endl;
        exit(-1);
    }
    fout.close();
}

const bool FileExist(const std::string& file_path) {
    std::ifstream fin;
    fin.open(file_path);
    const bool exists = fin.good();
    fin.close();
    return exists;
}