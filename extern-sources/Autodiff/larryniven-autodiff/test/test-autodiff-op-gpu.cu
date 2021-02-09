#include "ebt/ebt.h"
#include <vector>
#include <functional>
#include "la/la-gpu.h"
#include "autodiff/autodiff-op-gpu.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests = {
    {"test-vec-logistic", []() {
        la::vector<double> ha {0, 1, 2};
        la::gpu::vector<double> da {ha};
        la::gpu::vector<double> db;
        db.resize(da.size());
        autodiff::op::gpu::logistic(db, da);
        la::vector<double> hb = to_host(db);
        ebt::assert_equals(0.5, hb(0));
        ebt::assert_equals(0.731059, hb(1), 1e-5);
        ebt::assert_equals(0.880797, hb(2), 1e-5);
    }},
};

int main()
{
    for (auto& t: tests) {
        std::cout << t.first << std::endl;
        t.second();
    }

    return 0;
}
