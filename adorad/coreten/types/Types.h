#pragma once 

namespace coreten {

#define CORETEN_SCALAR_TYPES(__)    \ 
    __(uint8_t, Byte),              \
    __(int8_t, Char),               \
    __(int16_t, Short),             \
    __(int64_t, Long),              \
    __(int, Int),                   \
    __(float, Float),               \
    __(double, Double),             \
    __(bool, Bool)                  \


#define CORETEN_SCALAR_AND_COMPLEX(__)    \ 
    __(uint8_t, Byte),              \
    __(int8_t, Char),               \
    __(int16_t, Short),             \
    __(int64_t, Long),              \
    __(int, Int),                   \
    __(float, Float),               \
    __(double, Double),             \
    __(bool, Bool)                  \


} // namespace coreten 