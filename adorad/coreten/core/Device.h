#pragma once 

#include <adorad/coreten/Checks.h>


// coreten::Device is an enum which holds the current device (CPU/CUDA) that a Tensor is defined on
// Like MemoryFormat, it is NOT intended to be used as a return value for any function, internal or external 

// Possible options are:
//  CPU:
//     This is the default device 
// 
// CUDA:
//     If you have an CUDA-supported device, this should be used 
namespace coreten {

enum class Device : int8_t {
    CPU,
    CUDA
};

inline Device get_cpu_device() {
    return Device::CPU;
}

inline Device get_cuda_device() {
    return Device::CUDA;
}

inline std::ostream& operator<<(std::ostream& stream, Device device_format) {
    switch (device_format) {
        case Device::CPU:
            return stream << "Device: CPU";

        case Device::CUDA:
            return stream << "Device: CUDA";

        default:
            // Hard-coding false, 
            CORETEN_CHECK(false, "Unknown Device format");
    }
}

} // namespace coreten 