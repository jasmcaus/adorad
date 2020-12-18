#pragma once
#define OPENBLAS_CONST
#include "ops_dot_blas.hpp"
#include "tensor.hpp"
#include <cblas.h>

namespace ts {

auto dot(Matrix A, Vector X) -> Vector
{
    Vector Y(A.shape()[0]);
    cblas_sgemv(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, A.shape()[0],
                A.shape()[1], 1.0f, A.data(), A.shape()[1], X.data(), 1, 0.0f, Y.data(), 1);

    return Y;
}

auto dot(Matrix A, Matrix B) -> Matrix
{
    Matrix C(A.shape()[0], B.shape()[1]);
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans, A.shape()[0], B.shape()[1], A.shape()[1], 1.0f,
                A.data(), A.shape()[1], B.data(), B.shape()[1], 0.0f, C.data(), C.shape()[1]);
    return C;
}

}