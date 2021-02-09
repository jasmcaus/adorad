#pragma once

#include "tensor/tensor.hpp"

namespace ts {

class DatasetIterator {

  private:
    ts::Tensor<float, 2> _inputs;
    ts::Tensor<int, 1> _labels;
    int _batch_size;
    int _index = 0;

  public:
    using return_type = std::pair<ts::Tensor<float, 2>, ts::Tensor<int, 1>>;

    typedef std::forward_iterator_tag iterator_category;

    DatasetIterator(ts::Tensor<float, 2> const &inputs, ts::Tensor<int, 1> const &labels,
                    int batch_size, int index = 0);

    auto operator++() -> DatasetIterator &;

    auto operator+(int value) -> DatasetIterator &;

    auto operator++(int) -> DatasetIterator;

    auto operator*() -> return_type;

    auto operator*() const -> return_type { return make_pair(); }

    auto operator->() -> return_type { return make_pair(); }

    auto make_pair() -> return_type;

    auto make_pair() const -> return_type;

    friend auto operator==(DatasetIterator const &lhs, DatasetIterator const &rhs) -> bool;

    friend auto operator!=(DatasetIterator const &lhs, DatasetIterator const &rhs) -> bool;
};

}
