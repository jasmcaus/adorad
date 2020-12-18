#pragma once
#include "tensor_forward.hpp"

namespace ts {

auto dot(Matrix, Vector) -> Vector;

auto dot(Matrix, Matrix) -> Matrix ;

}