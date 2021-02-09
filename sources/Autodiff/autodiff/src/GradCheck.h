#pragma once

#include "xtorch.h"

namespace xtorch {

void gradCheckModule(vector<Tensor> params, auto lossFn, auto input, bool show = false, bool wrtInput = true) {
    // Do a forward pass to build the graph
    auto loss0 = lossFn();

    // Analytically compute gradients of weights and inputs
    loss0.backward();

    const double eps = .000935913915;

    cout << ":::Weights:::" << endl;
    // Compute gradients of weights by difference quotient
    for (Tensor w : params) {
        auto data = xt::flatten(w.node->value);
        auto analyticGrad = xt::flatten(w.node->grad);
        for (int i = 0; i < data.size(); ++i) {
            const double orig = data[i];
            data[i] += eps;
            auto loss1 = lossFn();
            data[i] = orig - eps;
            auto loss2 = lossFn();
            data[i] = orig;
            const auto dwi = ((loss1 - loss2) / (2 * eps)).getValue();

            if (std::abs(dwi[0] - analyticGrad[i]) > .001 or show) {
                cout << "diff(" << dwi << ") \t\t grad(" << analyticGrad[i] << ") \t\t " << analyticGrad[i] / dwi[0] << " \t\t shape("
                     << xt::adapt(w.shape()) << ")" << endl;
            }
        }
    }
    if (!wrtInput)
        return;

    cout << ":::Inputs:::" << endl;
    // Compute gradients of input by difference quotient
    auto data = xt::flatten(input.node->value);
    auto analyticGrad = xt::flatten(input.node->grad);
    for (int i = 0; i < data.size(); ++i) {
        const double orig = data[i];
        data[i] += eps;
        auto loss1 = lossFn();
        data[i] = orig - eps;
        auto loss2 = lossFn();
        data[i] = orig;
        const auto dwi = ((loss1 - loss2) / (2 * eps)).getValue();

        if (std::abs(dwi[0] - analyticGrad[i]) > .001 or show) {
            cout << "diff(" << dwi << ") \t\t grad(" << analyticGrad[i] << ") \t\t " << analyticGrad[i] / dwi[0] << " \t\t shape("
                 << xt::adapt(input.shape()) << ")" << endl;
        }
    }
}

void gradCheck() {
    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}}};
        // Tensor target{xt::xarray<double>{{1, 0, -1, 0}}};
        // Linear m{3, 4};
        // auto crit = MSELoss();
        // auto lossFn = [&]() { return crit(m(input), target); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {.99, 2.3, -.4}}};
        // Tensor target{xt::xarray<double>{{1, 0, -1, 0}, {-.5, .3, .4, 1}}};
        // Linear m{3, 4};
        // auto crit = MSELoss();
        // auto lossFn = [&]() { return crit(m(input), target); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {.99, 2.3, -.4}}};
        // Tensor target{xt::xarray<double>{{1, 0, -1, 0}, {-.5, .3, .4, 1}}};
        // Linear m{3, 4};
        // auto crit = MSELoss();
        // auto lossFn = [&]() { return crit(m(input), target); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}}};
        // Linear m{3, 4};
        // auto lossFn = [&]() { return m(input).sum(); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {3, 2, -6}}};
        // Linear m{3, 4};
        // auto lossFn = [&]() { return m(input).sum(); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {3, 2, -6}}};
        // Linear m{3, 4};
        // auto lossFn = [&]() {
        //     auto y = m(input).sum();
        //     return (y * y - y * .5).sin();
        // };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, -2}, {3, 2, -6, 0}}};
        // Tensor A{xt::random::randn<double>({4, 4})};
        // Tensor b{xt::random::randn<double>({2, 4})};
        // auto lossFn = [&]() {
        //     auto z = x.dot(A) + b;
        //     auto y = z.dot(A) + b;
        //     return (y.sin().square() * 3 * z).sum();
        // };
        // gradCheckModule({A, b}, lossFn, x);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {3, 2, -6}}};
        // Linear m{3, 4};
        // auto lossFn = [&]() {
        //     auto y = m(input).sum();
        //     return y / ((y * y).relu() + y.relu() * 3.141592).sin();
        // };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {3, 2, -6}}};
        // Linear m{3, 4};
        // auto lossFn = [&]() { return m(input).sum({0}).sum({0}); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::xarray<double>{{.5, .3, .1}, {3, 2, -6}}};
        // Linear m{3, 4};
        // Linear m1{4, 3};
        // auto lossFn = [&]() { return m1(m(input).relu(.1)).relu(.1).sum(); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor input{xt::random::randn<double>({1, 2, 4, 4})};
        // Conv2d conv(2, 2, 4, 2, 1, true);
        // auto lossFn = [&]() { return conv(input).relu(.1).sum(); };
        // gradCheckModule(conv.parameters(), lossFn, input, true);
    }

    {
        // const int inDepth = 2, outDepth = 3, filterSize = 5, pad = 2, stride = 2, inSz = 7, batchSize = 1;
        // const int outSz = (inSz - filterSize + 2 * pad) / stride + 1;
        // const int flatSize = outDepth * (outSz * outSz);
        //
        // Tensor input{xt::random::randn<double>({batchSize, inDepth, inSz, inSz})};
        //
        // auto m = Sequential(Conv2d{inDepth, outDepth, filterSize, stride, pad}, FlattenBatch{}, Linear{flatSize, 10});
        // auto lossFn = [&]() { return m(input).sum(); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // const int inDepth = 2, outDepth = 3, filterSize = 4, pad = 1, stride = 2, outSz = 8, batchSize = 1;
        // Tensor input{xt::random::randn<double>({batchSize, outDepth, outSz, outSz})};
        // auto m = Sequential(ConvTranspose2d{inDepth, outDepth, filterSize, stride, pad, false}, ReLU{.1});
        // auto lossFn = [&]() { return m(input).sum(); };
        // gradCheckModule(m.parameters(), lossFn, input);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, -2}, {.2342, .1, -.06, 0}}};
        // auto lossFn = [&]() { return x.sigmoid().sum(); };
        // gradCheckModule({}, lossFn, x);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, -2}, {.2342, .1, -.06, 0}}};
        // auto lossFn = [&]() { return x.tanh().sum(); };
        // gradCheckModule({}, lossFn, x);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, 2}, {.2342, .1, .06, 0}}};
        // auto lossFn = [&]() { return x.pow(3.141592).sum().square(); };
        // gradCheckModule({}, lossFn, x, true);
    }

    {
        const int numFeatures = 3, outSz = 4, batchSize = 128;
        // Tensor x{(xt::random::randn<double>({batchSize, numFeatures, outSz, outSz}) * 3.1415) * 2. + .1};
        // Tensor x{(xt::random::randn<double>({batchSize, numFeatures, outSz, outSz})) + .1};
        Tensor x{(xt::random::randn<double>({batchSize, numFeatures, outSz, outSz})) + 2.};
        BatchNorm2d b(numFeatures, 1e-5, .1, true, false);
        auto lossFn = [&]() {
            // return b(x).sum();
            // return b(x).square().sum();
            return b(x).relu().sum();
            // return b(x).sigmoid().sum();
        };
        gradCheckModule(b.parameters(), lossFn, x, true);
    }

    {
        // const int numFeatures = 3, outSz = 64, batchSize = 8;
        // Tensor x{(xt::random::randn<double>({batchSize, numFeatures, outSz, outSz})) + 2.};
        // Sequential netD{Conv2d{3, 64, 4, 2, 1, false},           ReLU{.2}, Conv2d{64, 128, 4, 2, 1, false},
        //                 BatchNorm2d{128, 1e-5, .1, true, false}, ReLU{.2}, Conv2d{128, 256, 4, 2, 1, false},
        //                 BatchNorm2d{256, 1e-5, .1, true, false}, ReLU{.2}, Conv2d{256, 512, 4, 2, 1, false},
        //                 BatchNorm2d{512, 1e-5, .1, true, false}, ReLU{.2}, Conv2d{512, 1, 4, 1, 0, false}};
        // auto lossFn = [&]() { return netD(x).sum().sigmoid(); };
        // gradCheckModule({m.parameters()[10]}, lossFn, x, true);
    }

    {
        // const int batchSize = 8;
        // Tensor x{(xt::random::randn<double>({batchSize, 100, 1, 1})) + 2.};
        // Sequential netG{ConvTranspose2d{100, 512, 4, 1, 0, false}, ReLU{.2}, ConvTranspose2d{512, 256, 4, 2, 1, false},
        //                 BatchNorm2d{512, 1e-5, .1, true, false},   ReLU{.2}, ConvTranspose2d{128, 256, 4, 2, 1, false},
        //                 BatchNorm2d{256, 1e-5, .1, true, false},   ReLU{.2}, ConvTranspose2d{256, 512, 4, 2, 1, false},
        //                 BatchNorm2d{512, 1e-5, .1, true, false},   ReLU{.2}, ConvTranspose2d{512, 1, 4, 1, 0, false}};
        // auto lossFn = [&]() { return netD(x).sum().sigmoid(); };
        // gradCheckModule({m.parameters()[10]}, lossFn, x, true);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, -2}, {.2342, .1, -.06, 0}}};
        // Tensor y{xt::xarray<double>{1, 0}};
        // Linear l{4, 1};
        // auto crit = BCEWithLogitsLoss();
        // auto lossFn = [&]() { return crit(l(x).reshape({2}), y); };
        // gradCheckModule(l.parameters(), lossFn, x, true);
    }

    {
        // Tensor x{xt::xarray<double>{{.5, .3, .1, -2}, {.2342, .1, -.06, 0}}};
        // Tensor y{xt::xarray<double>{1, 0}};
        // Linear l{4, 1};
        // auto crit = BCELoss();
        // auto lossFn = [&]() {
        //     return crit(l(x).sigmoid().reshape({2}), y);
        // };
        // gradCheckModule(l.parameters(), lossFn, x, true);
    }
}
}; // namespace xtorch

