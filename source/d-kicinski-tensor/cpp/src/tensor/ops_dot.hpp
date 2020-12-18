#pragma once
#include "tensor_forward.hpp"
#include <vector>

namespace ts {

auto dot(Matrix const &, Vector const &) -> Vector;

auto dot(Matrix const & A, Matrix const & B, bool A_T=false, bool B_T=false) -> Matrix ;

auto dot(Tensor<float, 3> const & A, Matrix const & B) -> Tensor<float, 3>;

}