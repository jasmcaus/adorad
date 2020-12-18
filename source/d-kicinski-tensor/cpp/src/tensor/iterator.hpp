#pragma once

namespace ts {

template <typename Element> class IteratorClass {
    Element *_element;
    std::size_t _index;

  public:
    typedef std::forward_iterator_tag iterator_category;

    IteratorClass(Element *element, std::size_t index) : _element(element), _index(index) {}

    auto operator++() -> IteratorClass &
    {
        ++_index;
        return *this;
    }

    auto operator++(int) -> IteratorClass
    {
        std::size_t temp = _index;
        ++(*this);
        return IteratorClass(_element, temp);
    }

    auto operator*() -> Element & { return _element[_index]; }

    auto operator*() const -> Element const & { return _element[_index]; }

    auto operator->() -> Element * { return &_element[_index]; }

    friend auto operator==(IteratorClass const & lhs, IteratorClass const & rhs) -> bool {
        // comparing addresses because I can't assume Element will have operator== implemented
        return &(lhs._element[lhs._index]) == &(rhs._element[rhs._index]);
    }

    friend auto operator!=(IteratorClass const & lhs, IteratorClass const & rhs) -> bool {
        return &(lhs._element[lhs._index]) != &(rhs._element[rhs._index]);
    }

    static auto begin(Element * element, std::size_t length) -> IteratorClass {
        return IteratorClass(element, 0);
    }

    static auto end(Element * element, std::size_t length) -> IteratorClass {
        return IteratorClass(element, length);
    }
};

}
