# tensor
[![Build Status](https://travis-ci.org/d-kicinski/tensor.svg?branch=master)](https://travis-ci.org/d-kicinski/tensor)
[![codecov](https://codecov.io/gh/d-kicinski/tensor/branch/master/graph/badge.svg)](https://codecov.io/gh/d-kicinski/tensor)


This library provides a two main features:
- A class for interacting with multidimensional arrays (For backend library uses BLAS/LAPACK libraries with fallback to
own naive implementations).
- Deep neural networks.

The design goal is to create a numpy/pytorch alike interface for interacting
with multidimensional arrays packaged in a simple, relatively lightweight, library with limited external dependencies that
could be used on platforms like android phones and microcontrollers.

#### How to use in your project
If you're using cmake see [tensor-example](https://github.com/dawidkski/tensor-example) for example usage.

#### Usage of `tensor/nn` module
For example usage jump to [nn-planar-data example](https://github.com/d-kicinski/tensor/tree/master/examples/nn-planar-data)

#### Usage of `tensor` module in embedded application
See this [repository](https://github.com/d-kicinski/tensor-example-embedded)

#### Example usages of `tensor` module
Create multidimensional tensor:
```c++
// Create an array for storing bitmaps with variadic constructor
int constexpr width = 1024;
int constexpr height = 720;
int constexpr channels = 3;

ts::Tensor<int, 3> image(width, height, channels);
image.shape();      // std::array{1024, 720, 3});
image.data_size();  // 1024 * 720 * 3 = 2211840
```

Multiply matrices:
```c++
// Construct 2D arrays via std::initializer_list
using Matrix = ts::Tensor<float, 2>;
Matrix A = {
    {3, 1, 3},
    {1, 5, 9},
};
Matrix B = {
    {3, 1},
    {1, 5},
    {2, 6}
};
// Multiply via free function
Matrix C = ts::dot(A, B);
// C = 
//    / 16 26 \
//    \ 26 80 /
```
