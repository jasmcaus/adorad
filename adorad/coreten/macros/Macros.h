#pragma once 


#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr)    (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr)  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr)    (expr)
#define C10_UNLIKELY(expr)  (expr)
#endif

#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)