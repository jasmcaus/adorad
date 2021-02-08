#pragma once 

#define CORETEN_ASSERT
#define CORETEN_ASSERT_VOID
#define CORETEN_ASSERT_PTR

#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif


namespace coreten {

/// The primary Error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`.
/// Don't throw this directly
///
/// NB: coreten::Error is handled specially  to suppress the backtrace

class Error : public std::exception {
    // The actual error message.
    std::string msg_;

    public:

    // Default error message
    Error(
        const char* file,
        const uint32_t line,
        const char* condition,
        const std::string& msg);

    // Base constructor
    Error(std::string msg);

    // Add some new context to the message stack.  The last added context
    // will be formatted at the end of the context list upon printing.
    // WARNING: This method is O(n) in the size of the stack, so don't go
    // wild adding a ridiculous amount of context to error messages.
    void add_context(std::string msg);

    const std::string& msg() const { return msg_; }

};


// Used for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).  These turn into
// IndexError when they cross to Python.
class IndexError : public Error {
    using Error::Error;
};

// Used for invalid values.  These turn into
// ValueError when they cross to Python.
class ValueError : public Error {
    using Error::Error;
};

// Used for invalid types.  These turn into
// TypeError when they cross to Python.
class TypeError : public Error {
    using Error::Error;
};

// Used for non finite indices.  These turn into
// ExitException when they cross to Python.
class EnforceFiniteError : public Error {
    using Error::Error;
};



namespace detail {

// Return x if it is non-empty; otherwise return y.
inline std::string if_empty(const std::string& x, const std::string& y) {
    if (x.empty()) { return y; }
    else { return x; }
}


// A utility macro to make it easier to test for error conditions from user
// input.  Like CORETEN_INTERNAL_ASSERT, it supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the error
// message using operator<< (e.g., you can pass any object which has
// operator<< defined.  Most objects in PyTorch have these definitions!)
//
// Usage:
//    CORETEN_CHECK(should_be_true); // A default error message will be provided
//                                 // in this case; but we recommend writing an
//                                 // explicit error message, as it is more
//                                 // user friendly.
//    CORETEN_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception.  If this exception propagates
// to Python, it will convert into a Python RuntimeError.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike CHECK() from glog.)
//
[[noreturn]] void coretenCheckFail(const char* func, const char* file, uint32_t line, const std::string& msg);

} //namespace detail 
} // namespace coreten



#define CORETEN_CHECK_MSG(cond, type, ...)                              \
    ::coreten::detail::if_empty(                                       \
        ::coreten::str(__VA_ARGS__),                                        \
        "Expected " #cond " to be true, but got false.  "               \
    )

#define CORETEN_CHECK_WITH_MSG(error_t, cond, type, ...)                \
    if (CORETEN_UNLIKELY_OR_CONST(!(cond))) {                               \
        CORETEN_THROW_ERROR(error_t, CORETEN_CHECK_MSG(cond, type, __VA_ARGS__)); \
    }


#define CORETEN_CHECK(cond, ...)                                          \
    if (CORETEN_UNLIKELY_OR_CONST(!(cond))) {                                 \
        ::coreten::detail::coretenCheckFail(                                      \
            __func__, __FILE__, static_cast<uint32_t>(__LINE__),            \
            CORETEN_CHECK_MSG(cond, "", __VA_ARGS__));                        \
    }



// Like CORETEN_CHECK, but raises IndexErrors instead of Errors.
#define CORETEN_CHECK_INDEX(cond, ...) \
    CORETEN_CHECK_WITH_MSG(IndexError, cond, "INDEX", __VA_ARGS__)

// Like CORETEN_CHECK, but raises ValueErrors instead of Errors.
#define CORETEN_CHECK_VALUE(cond, ...) \
    CORETEN_CHECK_WITH_MSG(ValueError, cond, "VALUE", __VA_ARGS__)

// Like CORETEN_CHECK, but raises TypeErrors instead of Errors.
#define CORETEN_CHECK_TYPE(cond, ...) \
    CORETEN_CHECK_WITH_MSG(TypeError, cond, "TYPE", __VA_ARGS__)



// Private helper macro for implementing TORCH_INTERNAL_ASSERT and TORCH_CHECK
//
// Note: In the debug build With MSVC, __LINE__ might be of long type (a.k.a int32_t),
// which is different from the definition of `SourceLocation` that requires
// unsigned int (a.k.a uint32_t) and may cause a compile error with the message:
// error C2397: conversion from 'long' to 'uint32_t' requires a narrowing conversion
// Here the static cast is used to pass the build.
// if this is used inside a lambda the __func__ macro expands to operator(),
// which isn't very useful, but hard to fix in a macro so suppressing the warning.
#define CORETEN_THROW_ERROR(err_type, msg) \
    throw ::coreten::err_type({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)