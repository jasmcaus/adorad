#pragma once
#include <iostream>
using std::cout;
using std::endl;

namespace xtorch {
void printShape(const auto& x) {
    auto s = x.shape();
    cout << "[";
    for (int i = 0; i + 1 < s.size(); ++i)
        cout << s[i] << ",";
    if (s.size()) {
        cout << s.back();
    }
    cout << "]";
}
} // namespace xtorch

#define XTENSOR_USE_XSIMD
#define XTENSOR_USE_OPENMP
#define XTENSOR_DISABLE_EXCEPTIONS
// #define XTENSOR_ENABLE_ASSERT

#include "Losses.h"
#include "Modules.h"
#include "Optimizers.h"
#include "Tensor.h"

