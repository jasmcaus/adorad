#pragma once
#include "xtorch.h"
using namespace xtorch;

#include <xtensor-io/ximage.hpp>

#include <iostream>
using std::cout;
using std::endl;
#include <filesystem>
namespace fs = std::filesystem;

const int imgSz = 64;
const int imgDepth = 3;

struct DataLoader {
    vector<vector<std::string>> batches;
    const int batchSize;

    DataLoader(const int batchSize) : batchSize(batchSize) {
        std::string folder = "/home/xdaimon/projects/ID/IntrinsicDimensionality/Datasets/CelebAResized";
        vector<std::string> batch;
        for (const auto& entry : fs::directory_iterator(folder)) {
            batch.push_back(entry.path());
            if (batch.size() == batchSize) {
                batches.push_back(batch);
                batch.clear();
            }
        }
    }

    int size() { return batches.size(); }

    Tensor loadBatch(int i) {
        vector<std::string>& batch = batches[i];
        xt::xarray<double> ret = xt::zeros<double>({batchSize, imgSz, imgSz, imgDepth});
        for (int j = 0; j < batchSize; ++j) {
            auto img = xt::load_image(batch[j]);
            xt::view(ret, j) = 2. * (img / 255.) - 1.;
        }
        return {xt::transpose(ret, {0, 3, 1, 2})};
    }
};

void dump_img(auto x, auto name) {
    xt::xarray<double> x0 = ((x + 1.) / 2.) * 256.;
    xt::xarray<int> x1 = xt::minimum(xt::cast<int>(xt::transpose(x0, {1, 2, 0})), 255);
    xt::xarray<int> x2 = xt::maximum(x1, 0);
    xt::dump_image(name, x2);
}
