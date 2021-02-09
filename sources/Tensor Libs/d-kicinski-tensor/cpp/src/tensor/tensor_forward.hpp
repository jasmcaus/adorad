#pragma once

namespace ts {

template <typename Element, int Dim> class Tensor;

// Convenient typedefs
typedef Tensor<float, 2> Matrix;
typedef Tensor<float, 1> Vector;

}
