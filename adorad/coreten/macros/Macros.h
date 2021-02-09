#pragma once 

// Based off of PyTorch 

// CORETEN_LIKELY/CORETEN_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if CORETEN_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect takes a long argument, which means you may
// trigger the wrong conversion without it.
//

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define CORETEN_LIKELY(expr)    (__builtin_expect(static_cast<bool>(expr), 1))
#define CORETEN_UNLIKELY(expr)  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define CORETEN_LIKELY(expr)    (expr)
#define CORETEN_UNLIKELY(expr)  (expr)
#endif

#define CORETEN_UNLIKELY_OR_CONST(e) CORETEN_UNLIKELY(e)