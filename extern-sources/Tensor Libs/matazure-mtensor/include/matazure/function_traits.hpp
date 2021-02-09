#pragma once

namespace matazure {

/// a type traits to get the argument type and result type of a functor
template <typename _Fun>
struct function_traits : public function_traits<decltype(&_Fun::operator())> {};

/// implements
template <typename _ClassType, typename _ReturnType, typename... _Args>
struct function_traits<_ReturnType (_ClassType::*)(_Args...) const> {
    enum { arguments_size = sizeof...(_Args) };

    typedef _ReturnType result_type;

    template <int_t _index>
    struct arguments {
        typedef typename std::tuple_element<_index, std::tuple<_Args...>>::type type;
    };
};

}  // namespace matazure
