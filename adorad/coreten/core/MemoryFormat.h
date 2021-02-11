#pragma once 

#include <iostream>

#include <adorad/coreten/Checks.h>
#include <adorad/coreten/String/StringUtil.h>


// Memory format is NOT the property of a Tensor. It is simply a way to tell an operator how the result should be organized 
// in memory. Nothing more. 
// By this logic, memory format should NEVER be used as a return value for any Tensor state (either internally or
// externally)


// Possible options are:
//  Preserve:
//    If any of the input Tensor is in the ``channels_last`` format, the operator output should be in the
//    ``channels_last`` format
//
//  ChannelsLast:
//    Regardless of the input Tensor format, the output should be in ``channels_last`` format.
// 
//  Contiguous:
//    Regardless of the input Tensor format, the output should be a contiguous Tensor.
//

namespace coreten {

enum class MemoryFormat : int8_t {
    Preserve, 
    ChannelsLast, 
    Contiguous
};

inline MemoryFormat get_contiguous_memory_format() {
    return MemoryFormat::Contiguous;
}

inline MemoryFormat get_preserve_memory_format() {
    return MemoryFormat::Preserve;
}

inline MemoryFormat get_channelslast_memory_format() {
    return MemoryFormat::ChannelsLast;
}

inline std::ostream& operator<<(std::ostream& stream, MemoryFormat memory_format) {
    switch (memory_format) {
        case MemoryFormat::Preserve:
            return stream << "Preserve";

        case MemoryFormat::Contiguous:
            return stream << "Contiguous";

        case MemoryFormat::ChannelsLast:
            return stream << "ChannelsLast";

        default:
            // Hard-coding false, 
            CORETEN_CHECK(false, "Unknown memory format");
    }
}

} // namespace coreten 