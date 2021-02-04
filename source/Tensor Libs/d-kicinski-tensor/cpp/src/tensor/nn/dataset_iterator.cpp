#include "dataset_iterator.hpp"

ts::DatasetIterator::DatasetIterator(const ts::Tensor<float, 2> &inputs,
                                     const ts::Tensor<int, 1> &labels, int batch_size, int index)
    : _inputs(inputs), _labels(labels), _batch_size(batch_size), _index(index) { }

auto ts::DatasetIterator::operator++() -> ts::DatasetIterator &
{
    _index += _batch_size;
    return *this;
}

auto ts::DatasetIterator::operator+(int value) -> ts::DatasetIterator &
{
    _index += value;
    return *this;
}

auto ts::DatasetIterator::operator++(int) -> ts::DatasetIterator
{
    std::size_t temp = _index;
    (*this) + _batch_size;
    return DatasetIterator(_inputs, _labels, _batch_size, temp);
}

auto ts::DatasetIterator::operator*() -> ts::DatasetIterator::return_type
{
    return std::make_pair(ts::slice(_inputs, _index, _index + _batch_size),
                          ts::slice(_labels, _index, _index + _batch_size));
}

auto ts::DatasetIterator::make_pair() -> ts::DatasetIterator::return_type
{
    return std::make_pair(ts::slice(_inputs, _index, _index + _batch_size),
                          ts::slice(_labels, _index, _index + _batch_size));
}

auto ts::DatasetIterator::make_pair() const -> ts::DatasetIterator::return_type
{
    return std::make_pair(ts::slice(_inputs, _index, _index + _batch_size),
                          ts::slice(_labels, _index, _index + _batch_size));
}

auto ts::operator==(const ts::DatasetIterator &lhs, const ts::DatasetIterator &rhs) -> bool
{
    return lhs._index >= rhs._inputs.shape()[0];
}

auto ts::operator!=(const ts::DatasetIterator &lhs, const ts::DatasetIterator &rhs) -> bool
{
    return !(lhs == rhs);
}
