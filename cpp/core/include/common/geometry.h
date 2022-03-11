#ifndef COMMON_GEOMETRY_H
#define COMMON_GEOMETRY_H

#include "common/config.h"
#include "common/common.h"

void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S);
void PolarDecomposition(const Matrix3r& F, Matrix3r& R, Matrix3r& S);
const Matrix2r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF);
const Matrix3r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S, const Matrix3r& dF);
const Matrix4r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S);
const Matrix9r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S);

// F = U * sig * Vt.
void Svd(const Matrix2r& F, Matrix2r& U, Vector2r& sig, Matrix2r& V);
void Svd(const Matrix3r& F, Matrix3r& U, Vector3r& sig, Matrix3r& V);
void dSvd(const Matrix2r& F, const Matrix2r& U, const Vector2r& sig, const Matrix2r& V, const Matrix2r& dF,
    Matrix2r& dU, Vector2r& dsig, Matrix2r& dV);
void dSvd(const Matrix3r& F, const Matrix3r& U, const Vector3r& sig, const Matrix3r& V, const Matrix3r& dF,
    Matrix3r& dU, Vector3r& dsig, Matrix3r& dV);

const Vector4r Flatten(const Matrix2r& A);
const Vector9r Flatten(const Matrix3r& A);
const Matrix2r Unflatten(const Vector4r& a);
const Matrix3r Unflatten(const Vector9r& a);

const Matrix3r SkewSymmetricMatrix(const Vector3r& w);

#endif