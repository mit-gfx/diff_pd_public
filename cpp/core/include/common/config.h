#ifndef COMMON_CONFIG_H
#define COMMON_CONFIG_H

// Common headers.
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// Flags for timing.
#include <sys/time.h>
// OpenMP
#include <omp.h>

#include "Eigen/Dense"
#include "Eigen/Sparse"

using real = double;

// PRINT_LEVEL:
// 0: silence;
// 1: print error only;
// 2: print error and warning;
// 3: print all.
#define PRINT_NOTHING           0
#define PRINT_ERROR             1
#define PRINT_ERROR_AND_WARNING 2
#define PRINT_ALL               3
#define PRINT_LEVEL             PRINT_ALL

using Vector2i = Eigen::Matrix<int, 2, 1>;
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Vector4i = Eigen::Matrix<int, 4, 1>;
using VectorXi = Eigen::Matrix<int, -1, 1>;
using Vector2r = Eigen::Matrix<real, 2, 1>;
using Vector3r = Eigen::Matrix<real, 3, 1>;
using Vector4r = Eigen::Matrix<real, 4, 1>;
using Vector5r = Eigen::Matrix<real, 5, 1>;
using Vector6r = Eigen::Matrix<real, 6, 1>;
using Vector7r = Eigen::Matrix<real, 7, 1>;
using Vector8r = Eigen::Matrix<real, 8, 1>;
using Vector9r = Eigen::Matrix<real, 9, 1>;
using Vector24r = Eigen::Matrix<real, 24, 1>;
using VectorXr = Eigen::Matrix<real, -1, 1>;
using Matrix2r = Eigen::Matrix<real, 2, 2>;
using Matrix3r = Eigen::Matrix<real, 3, 3>;
using Matrix4r = Eigen::Matrix<real, 4, 4>;
using Matrix5r = Eigen::Matrix<real, 5, 5>;
using Matrix6r = Eigen::Matrix<real, 6, 6>;
using Matrix7r = Eigen::Matrix<real, 7, 7>;
using Matrix8r = Eigen::Matrix<real, 8, 8>;
using Matrix9r = Eigen::Matrix<real, 9, 9>;
using Matrix24r = Eigen::Matrix<real, 24, 24>;
using MatrixXr = Eigen::Matrix<real, -1, -1>;
using MatrixX2r = Eigen::Matrix<real, -1, 2>;
using MatrixX3r = Eigen::Matrix<real, -1, 3>;
using Matrix2Xr = Eigen::Matrix<real, 2, -1>;
using Matrix3Xr = Eigen::Matrix<real, 3, -1>;
using MatrixXi = Eigen::Matrix<int, -1, -1>;
using Matrix4Xi = Eigen::Matrix<int, 4, -1>;
using Matrix8Xi = Eigen::Matrix<int, 8, -1>;

using SparseVector = Eigen::SparseVector<real>;
using SparseMatrix = Eigen::SparseMatrix<real>;
using SparseMatrixElements = std::vector<Eigen::Triplet<real>>;

using RowVectorXr = Eigen::Matrix<real, 1, -1>;
using RowVector2r = Eigen::Matrix<real, 1, 2>;
using RowVector3r = Eigen::Matrix<real, 1, 3>;
using RowVector4r = Eigen::Matrix<real, 1, 4>;

#endif