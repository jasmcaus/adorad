#pragma once

#include "Tensor.h"

namespace xtorch {
// ATM I could have just used a function, but making the loss a class allows for optimizations later.
// Prevent reallocation of the MSELossNode and also store the topological order computed in Tensor::backward()?
class MSELoss {
  public:
    Tensor operator()(const Tensor& x, const Tensor& y) { return {opNode<MSELossNode>({x.node, y.node})}; }
};

class BCELoss {
  public:
    Tensor operator()(const Tensor& x, const Tensor& y) { return {opNode<BCELossNode>({x.node, y.node})}; }
};

class BCEWithLogitsLoss {
  public:
    Tensor operator()(const Tensor& x, const Tensor& y) { return {opNode<BCEWithLogitsLossNode>({x.node, y.node})}; }
};

} // namespace xtorch
