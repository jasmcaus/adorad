#pragma once

#include <tensor/tensor.hpp>

namespace ts {

class FeedForward {
  public:
    FeedForward(int dim_in, int dim_out, bool activation=false, float alpha=1e-3);

    auto operator()(Matrix const &) -> Matrix;

    auto forward(Matrix const &) -> Matrix;

    auto backward(Matrix) -> Matrix;

    auto update(float step_size) -> void;

    auto weights() -> Matrix;

  private:
    Matrix _x;
    Matrix _y;
    Matrix _weights;
    Matrix _d_weights;
    Vector _bias;
    Vector _d_bias;
    float _alpha;
    bool _activation;

};

}

