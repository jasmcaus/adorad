#pragma once

#include <tensor/tensor.hpp>
#include <utility>

namespace ts {

class CrossEntropyLoss {
  public:
    CrossEntropyLoss(std::vector<Matrix> weights, float alpha=1e-3)
        : _weights(std::move(weights)), _alpha(alpha) {}

    auto operator()(Matrix const & probs, Tensor<int, 1> const &labels) -> float;

    auto forward(Matrix const & probs, Tensor<int, 1> const &labels) -> float;

    auto backward(Matrix const &scores) -> Matrix;

  private:
    std::vector<Matrix> _weights;
    float _alpha;
    Tensor<int, 1> _labels;

    auto _calculate_regularization_loss() -> float;

};

}

