#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "ops.hpp"
#include "tensor_forward.hpp"

namespace ts {

/**
 *
 * @tparam Element is the type of array element
 * @tparam Dim is the number of dimensions
 */

template <typename Element, int Dim> class Tensor {

  public:
    using size_type = int;
    using vector_t = std::vector<Element>;
    using data_t = std::shared_ptr<vector_t>;
    using iterator = typename vector_t::iterator;

    auto data() const -> data_t { return _data; };
    auto shape() const -> std::array<size_type, Dim> { return _dimensions; }
    [[nodiscard]] auto shape(size_type index) const -> size_type { return _dimensions[index]; }
    [[nodiscard]] auto data_size() const -> size_type { return _data_size; }

    auto begin() -> iterator { return _begin; }
    auto end() -> iterator { return _end; }
    auto begin() const -> iterator { return _begin; }
    auto end() const -> iterator { return _end; }

    Tensor();

    Tensor(std::initializer_list<Element> list);

    Tensor(std::initializer_list<Tensor<Element, Dim - 1>> list);

    Tensor(std::vector<Tensor<Element, Dim - 1>> list);

    Tensor(std::array<size_type, Dim> const &shape);

    template <typename... Sizes> Tensor(size_type first, Sizes... rest);

    Tensor(Tensor const &tensor);

    Tensor(Tensor<Element, Dim + 1> const &tensor, size_type index);

    template <typename... Indices>
    auto operator()(size_type first, Indices... rest) -> decltype(auto);

    template <typename... Indices>
    auto operator()(size_type first, Indices... rest) const -> decltype(auto);

    auto operator[](size_type i) -> decltype(auto);

    auto operator==(Tensor<Element, Dim> const &other) const -> bool;

    auto operator!=(Tensor<Element, Dim> const &other) -> bool;

    auto operator=(Tensor const &tensor) -> Tensor &;

    auto operator<(Element const &value) -> Tensor<bool, Dim>;

    auto operator<=(Element const &value) -> Tensor<bool, Dim>;

    auto operator>(Element const &value) -> Tensor<bool, Dim>;

    auto operator>=(Element const &value) -> Tensor<bool, Dim>;

    auto operator==(Element const &value) -> Tensor<bool, Dim>;

    auto operator+=(Tensor const &tensor) -> Tensor &;

    auto operator-() -> Tensor &;

    auto static randn(std::vector<int> const &shape) -> Tensor;

  private:
    size_type _data_size;
    std::array<size_type, Dim> _dimensions;
    std::shared_ptr<std::vector<Element>> _data;
    iterator _begin;
    iterator _end;

    template <typename... Sizes> auto set_sizes(int pos, size_type first, Sizes... rest) -> void;

    template <typename... Indices>
    auto get_index(int pos, size_type prev_index, size_type first, Indices... rest) const
        -> size_type;
};

template <typename Element, int Dim> Tensor<Element, Dim>::Tensor()
{
    _data = nullptr;
    _data_size = 0;
}

template <typename Element, int Dim>
template <typename... Sizes>
Tensor<Element, Dim>::Tensor(Tensor::size_type first, Sizes... rest)
{
    set_sizes(0, first, rest...);
    _data = std::make_shared<vector_t>(_data_size);
    _begin = _data->begin();
    _end = _data->end();
}

template <typename Element, int Dim> Tensor<Element, Dim>::Tensor(Tensor const &tensor)
{
    _data_size = tensor.data_size();
    _dimensions = tensor.shape();
    _data = std::make_shared<vector_t>(*tensor.data());
    _begin = _data->begin();
    _end = _data->end();
}

template <typename Element, int Dim>
Tensor<Element, Dim>::Tensor(Tensor<Element, Dim + 1> const &tensor, Tensor::size_type index)
{
    std::copy(tensor.shape().begin() + 1, tensor.shape().end(), _dimensions.begin());
    _data_size = tensor.data_size() / tensor.shape()[0];
    _data = tensor.data();
    _begin = _data->begin() + index * _data_size;
    _end = _begin + _data_size; // TODO: I added +1 and nothing tests didn't break :/
}

template <typename Element, int Dim>
Tensor<Element, Dim>::Tensor(const std::array<size_type, Dim> &shape)
{
    std::copy(shape.begin(), shape.end(), _dimensions.begin());
    _data_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<>());
    _data = std::make_shared<vector_t>(_data_size);
    _begin = _data->begin();
    _end = _data->end();
}

template <typename Element, int Dim>
template <typename... Indices>
auto Tensor<Element, Dim>::operator()(Tensor::size_type first, Indices... rest) -> decltype(auto)
{
    if constexpr (sizeof...(Indices) == Dim - 1) {
        return _begin[get_index(0, 0, first, rest...)];
    } else if constexpr (Dim >= 2 && sizeof...(Indices) == 0) {
        return Tensor<Element, Dim - 1>(*this, first);
    } else if constexpr (Dim >= 2 && sizeof...(Indices) < Dim - 1) {
        return Tensor<Element, Dim - 1>(*this, first)(rest...);
    } else {
        assert(false);
    }
}

template <typename Element, int Dim>
template <typename... Indices>
auto Tensor<Element, Dim>::operator()(Tensor::size_type first, Indices... rest) const
    -> decltype(auto)
{
    if constexpr (sizeof...(Indices) == Dim - 1) {
        return _begin[get_index(0, 0, first, rest...)];
    } else if constexpr (Dim >= 2 && sizeof...(Indices) == 0) {
        return Tensor<Element, Dim - 1>(*this, first);
    } else if constexpr (Dim >= 2 && sizeof...(Indices) < Dim - 1) {
        return Tensor<Element, Dim - 1>(*this, first)(rest...);
    } else {
        assert(false);
    }
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator==(const Tensor<Element, Dim> &other) const -> bool
{
    if (_data_size != other.data_size()) {
        return false;
    }
    return std::equal(other.begin(), other.end(), _begin);
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator[](Tensor::size_type i) -> decltype(auto)
{
    if constexpr (Dim == 1) {
        return _begin[i];
    } else {
        return Tensor<Element, Dim - 1>(*this, i);
    }
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator!=(const Tensor<Element, Dim> &other) -> bool
{
    return !(*this == other);
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator=(Tensor const &tensor) -> Tensor &
{
    if (this == &tensor)
        return *this;

    _data = tensor.data();
    _data_size = tensor.data_size();
    _dimensions = tensor.shape();
    _begin = tensor.begin();
    _end = tensor.end();

    return *this;
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::randn(const std::vector<int> &shape) -> Tensor
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<Element> dist{0.0};

    std::array<int, Dim> array_shape;
    // TODO: this is weird solution :P
    std::copy(shape.begin(), shape.end(), array_shape.begin());
    Tensor<Element, Dim> tensor(array_shape);
    std::generate(tensor.begin(), tensor.end(), [&]() { return dist(mt); });
    return tensor;
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator<(const Element &value) -> Tensor<bool, Dim>
{
    return ts::mask<Element, Dim>(*this, [&](Element e) { return e < value; });
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator<=(const Element &value) -> Tensor<bool, Dim>
{
    return ts::mask<Element, Dim>(*this, [&](Element e) { return e <= value; });
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator>(const Element &value) -> Tensor<bool, Dim>
{
    return ts::mask<Element, Dim>(*this, [&](Element e) { return e > value; });
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator>=(const Element &value) -> Tensor<bool, Dim>
{
    return ts::mask<Element, Dim>(*this, [&](Element e) { return e >= value; });
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator==(const Element &value) -> Tensor<bool, Dim>
{
    return ts::mask<Element, Dim>(*this, [&](Element e) { return e == value; });
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator+=(Tensor const &tensor) -> Tensor &
{
    std::transform(_begin, _end, tensor.begin(), _begin, std::plus());
    return *this;
}

template <typename Element, int Dim>
template <typename... Sizes>
auto Tensor<Element, Dim>::set_sizes(int pos, Tensor::size_type first, Sizes... rest) -> void
{
    if (pos == 0) {
        _data_size = 1;
    }
    _dimensions[pos] = first;
    _data_size *= first;

    if constexpr (sizeof...(rest) > 0) {
        set_sizes(pos + 1, rest...);
    }
}

template <typename Element, int Dim>
template <typename... Indices>
auto Tensor<Element, Dim>::get_index(int pos, Tensor::size_type prev_index, Tensor::size_type first,
                                     Indices... rest) const -> Tensor::size_type
{
    size_type index = (prev_index * _dimensions[pos]) + first;
    if constexpr (sizeof...(rest) > 0) {
        return get_index(pos + 1, index, rest...);
    } else {
        return index;
    }
}

template <typename Element, int Dim>
Tensor<Element, Dim>::Tensor(std::initializer_list<Tensor<Element, Dim - 1>> list)
{
    if (list.size() == 0) {
        _data_size = 0;
        _data = nullptr;
        return;
    }
    _data_size = 0;
    _dimensions[0] = list.size();
    for (auto const &tensor : list) {
        if (_data_size == 0) {
            for (int i = 0; i < tensor.shape().size(); ++i) {
                _dimensions[i + 1] = tensor.shape()[i];
            }
        }
        _data_size += tensor.data_size();
    }

    _data = std::make_shared<vector_t>(_data_size);
    auto data_end = _data->begin();
    for (auto const &tensor : list) {
        data_end = std::copy(tensor.begin(), tensor.end(), data_end);
    }
    _begin = _data->begin();
    _end = data_end;
}

template <typename Element, int Dim>
Tensor<Element, Dim>::Tensor(std::vector<Tensor<Element, Dim - 1>> list)
{
    if (list.size() == 0) {
        _data_size = 0;
        _data = nullptr;
        return;
    }
    _data_size = 0;
    _dimensions[0] = list.size();
    for (auto const &tensor : list) {
        if (_data_size == 0) {
            for (int i = 0; i < tensor.shape().size(); ++i) {
                _dimensions[i + 1] = tensor.shape()[i];
            }
        }
        _data_size += tensor.data_size();
    }

    _data = std::make_shared<vector_t>(_data_size);
    auto data_end = _data->begin();
    for (auto const &tensor : list) {
        data_end = std::copy(tensor.begin(), tensor.end(), data_end);
    }
    _begin = _data->begin();
    _end = data_end;
}

template <typename Element, int Dim>
Tensor<Element, Dim>::Tensor(std::initializer_list<Element> list)
{
    if (list.size() == 0) {
        _data_size = 0;
        _data = nullptr;
        return;
    }
    _data_size = list.size();
    _dimensions[0] = _data_size;
    _data = std::make_shared<vector_t>(list.begin(), list.end());
    _begin = _data->begin();
    _end = _data->end();
}

template <typename Element, int Dim>
auto Tensor<Element, Dim>::operator-() -> Tensor &
{
    std::transform(_begin, _end, _begin, [](Element const &e) { return -e;});
    return *this;
}

} // namespace ts
