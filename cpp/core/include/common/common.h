#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H

#include "common/config.h"

const real ToReal(const double v);
const double ToDouble(const real v);

// Colorful print.
const std::string GreenHead();
const std::string RedHead();
const std::string YellowHead();
const std::string CyanHead();
const std::string GreenTail();
const std::string RedTail();
const std::string YellowTail();
const std::string CyanTail();
// Use return_code = -1 unless you want to customize it.
void PrintError(const std::string& message, const int return_code = -1);
void PrintWarning(const std::string& message);
void PrintInfo(const std::string& message);
void PrintSuccess(const std::string& message);

// Timing.
void Tic();
void Toc(const std::string& message);

// Error checking.
void CheckError(const bool condition, const std::string& error_message);

// Debugging.
void PrintNumpyStyleMatrix(const MatrixXr& mat);
void PrintNumpyStyleVector(const VectorXr& vec);

const real Clip(const real val, const real min, const real max);
const real ClipWithGradient(const real val, const real min, const real max, real& grad);

const real Pi();

// Eigen to std::vector.
const std::vector<real> ToStdVector(const VectorXr& v);
const VectorXr ToEigenVector(const std::vector<real>& v);

// String functions.
const bool BeginsWith(const std::string& full, const std::string& beginning);
const bool EndsWith(const std::string& full, const std::string& ending);

// Set/Vector functions.
const std::set<int> VectorToSet(const std::vector<int>& v);
const std::vector<int> SetToVector(const std::set<int>& s);
const bool SameSet(const std::set<int>& a, const std::set<int>& b);
// Used by the boundary condition algorithm.
const bool ProposeNewSet(const std::set<int>& a, const std::set<int>& b, std::set<int>& new_set);

// SparseMatrix.
const SparseMatrixElements FromSparseMatrix(const SparseMatrix& A);
const SparseMatrix ToSparseMatrix(const int row, const int col, const SparseMatrixElements& nonzeros);
void SaveSparseMatrixToBinaryFile(const SparseMatrix& A, const std::string& file_name);
const SparseMatrix LoadSparseMatrixFromBinaryFile(const std::string& file_name);
void SaveEigenVectorToBinaryFile(const VectorXr& v, const std::string& file_name);
const VectorXr LoadEigenVectorFromBinaryFile(const std::string& file_name);
void SaveEigenMatrixToBinaryFile(const MatrixXr& A, const std::string& file_name);
const MatrixXr LoadEigenMatrixFromBinaryFile(const std::string& file_name);
const VectorXr VectorSparseMatrixProduct(const VectorXr& v, const int row, const int col, const SparseMatrixElements& A);
const MatrixXr SparseMatrixMatrixProduct(const int row, const int col, const SparseMatrixElements& A, const MatrixXr& B);
const MatrixXr MatrixMatrixProduct(const MatrixXr& A, const MatrixXr& B);

const int GetNonzeros(const MatrixXr& A, const real eps);

#endif