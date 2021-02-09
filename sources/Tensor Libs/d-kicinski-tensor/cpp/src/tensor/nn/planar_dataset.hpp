#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "tensor/tensor.hpp"
#include "tensor/nn/dataset_iterator.hpp"

namespace ts {

class PlanarDataset {

  private:
    ts::Tensor<float, 2> _inputs;
    ts::Tensor<int, 1> _labels;
    int _batch_size;

  public:
    auto inputs() -> ts::Tensor<float, 2>;

    auto labels() -> ts::Tensor<int, 1>;

    auto size() -> int;

    auto begin() -> DatasetIterator;

    auto end() -> DatasetIterator;

    explicit PlanarDataset(std::string const &path, bool header_line = false, int batch_size = 16);

};

} // namespace ts
