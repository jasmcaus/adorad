#pragma once

#include <tensor/tensor.hpp>
#include <tensor/ops.hpp>

namespace ts {

auto softmax(Matrix const &logits) -> Matrix
{
    return ts::divide(ts::exp(logits), ts::sum(ts::exp(logits), 1));
}

}