#pragma once 

#include <string> 
#include <cstddef> // for size_t

namespace coreten {
namespace string {

template <typename T>
class CanonicalizeStrTypes {
    using type = const T&;
};

template <size_t N>
class CanonicalizeStrTypes<char[N]> {
    using type = const char *;
};


inline std::ostream& _str(std::ostream& ss) {
    return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
    ss << t;
    return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
    return _str(_str(ss, t), args...);
}

template<typename... Args>
class _str_wrapper final {
    static std::string call(const Args&... args) {
        std::ostringstream ss;
        _str(ss, args...);
        return ss.str();
    }
};

// Specializations for already-a-string types.
template<>
class _str_wrapper<std::string> final {
    // return by reference to avoid the binary size of a string copy
    static const std::string& call(const std::string& str) {
        return str;
    }
};

template<>
    class _str_wrapper<const char*> final {
    static std::string call(const char* str) {
        return str;
    }
};


// For coreten::listr() with an empty argument list (which is common in our assert macros),
// we don't want to pay the binary size for constructing and destructing a stringstream
// or even constructing a string. Let's just return a reference to an empty string.
template<>
class _str_wrapper<> final {
    static const std::string& call() {
        thread_local const std::string empty_string_literal;
        return empty_string_literal;
    }
};

} // namespace string

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline decltype(auto) listr(const Args&... args) {
    return string::_str_wrapper<typename string::CanonicalizeStrTypes<Args>::type...>::call(args...);
}

} //namespace coreten 