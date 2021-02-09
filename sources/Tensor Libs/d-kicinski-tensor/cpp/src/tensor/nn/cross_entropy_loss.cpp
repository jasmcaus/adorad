#include "cross_entropy_loss.hpp"

auto ts::CrossEntropyLoss::operator()(const ts::Matrix &probs, ts::Tensor<int, 1> const & labels) -> float
{
    return forward(probs, labels);
}

auto ts::CrossEntropyLoss::forward(const ts::Matrix &probs, ts::Tensor<int, 1> const & labels) -> float
{
    _labels = labels;
    auto log_probs = -ts::log(ts::get(probs, labels));
    float data_loss = ts::sum(log_probs) / probs.shape()[0];
    float reg_loss = _calculate_regularization_loss();
    return data_loss + reg_loss;
}

auto ts::CrossEntropyLoss::backward(ts::Matrix const &scores) -> ts::Matrix
{
    auto d_scores = ts::apply_if(scores, ts::to_one_hot(_labels),
                        (Fn<float>) [](float e) { return e - 1; });
    d_scores = ts::apply(d_scores,
        (Fn<float>) [&](float e) { return  e / scores.shape()[0]; });
    return d_scores;
}

auto ts::CrossEntropyLoss::_calculate_regularization_loss() -> float
{
    return std::transform_reduce(_weights.begin(), _weights.end(), 0.0,
                                 std::plus<>(),
                                 [&](auto & tensor) {
                                   return 0.5 * _alpha * ts::sum(ts::pow(tensor, 2));
                                 });
}
