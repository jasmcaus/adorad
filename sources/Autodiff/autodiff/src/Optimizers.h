#pragma once
#include <vector>

namespace xtorch {
using std::vector;

class SGD {
    vector<Tensor> parameters;
    double lr;

  public:
    SGD(const vector<Tensor>& parameters, const double lr = .01) : parameters(parameters), lr(lr) {}
    void zeroGrad() {
        for (const auto& param : parameters) {
            param.node->grad.fill(0.);
        }
    }
    void step() {
        for (const auto& param : parameters) {
            param.node->value += (-lr) * param.node->grad;
        }
    }
};

} // namespace xtorch
