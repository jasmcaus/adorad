#pragma once
#include <queue>
#include <functional>

#include "Nodes.h"
namespace xtorch {

// TODO handle multiple inputs/outputs?
class Module {
    shared_ptr<Node> input;
    shared_ptr<Node> output;
    vector<shared_ptr<Node>> parentsOfInputNode; // nodes that depend on the value of the input node
    bool hasBuiltGraph = false;

  public:
    Tensor operator()(const Tensor& x) { // TODO should only be called from root module? this probably doesn't work as expected
        if (!hasBuiltGraph) {
            hasBuiltGraph = true;
            input = x.node;
            const auto& ret = forward(x);
            output = ret.node;
            parentsOfInputNode = getParentsOfX(output, x.node);
            return ret;
        }
        auto inputNode = x.node;
        if (output == inputNode) {
            inputNode = x.detach().node;
        }

        // Replace previous input node with new input node
        // If both are ParamNodes then this basically just sets node->value
        for (auto& parent : parentsOfInputNode) {
            parent->replaceChild(input, inputNode);
        }
        input = inputNode;
        output->recompute(input); // recompute but do not descend into tree rooted at input
        return Tensor(output);
    }

    virtual Tensor forward(const Tensor& x) = 0;

    // Returns immediate submodules
    virtual vector<Module*> submodules() { return {}; }

    virtual void apply(std::function<void(Module&)> Fn) {
        for (Module* module : allModules()) {
            Fn(*module);
        }
    }

    vector<Tensor> parameters() {
        vector<Tensor> params;
        for (Module* module : allModules()) {
            const auto& moduleParameters = module->getParameters();
            params.insert(params.end(), moduleParameters.begin(), moduleParameters.end());
        }
        return params;
    }

    virtual std::string name() const { return ""; }

    virtual ~Module() {}

  protected:
    virtual vector<Tensor> getParameters() const { return {}; }
    vector<Module*> allModules() {
        std::queue<Module*> q;
        q.push(this);
        vector<Module*> modules;
        while (!q.empty()) {
            Module* module = q.front();
            q.pop();
            modules.push_back(module);
            for (Module* submodule : module->submodules()) {
                q.push(submodule);
            }
        }
        return modules;
    }
};

class Linear : public Module {
    Tensor W, b;

  public:
    Linear(int in, int out) : W{xt::random::randn<double>({in, out})}, b{xt::zeros<double>({out})} {}
    virtual Tensor forward(const Tensor& x) { return x.dot(W) + b; }

  protected:
    virtual vector<Tensor> getParameters() const { return {W, b}; }
};

class FlattenBatch : public Module {
  public:
    virtual Tensor forward(const Tensor& x) { return x.reshape({int(x.shape(0)), -1}); }
};

class Flatten : public Module {
  public:
    virtual Tensor forward(const Tensor& x) { return x.reshape({-1}); }
};

class ReLU : public Module {
    const double negativeSlope;

  public:
    ReLU(const double negativeSlope = 0.) : negativeSlope(negativeSlope) {}
    virtual Tensor forward(const Tensor& x) { return x.relu(negativeSlope); }
};

class Sin : public Module {
  public:
    virtual Tensor forward(const Tensor& x) { return x.sin(); }
};

class Sigmoid : public Module {
  public:
    virtual Tensor forward(const Tensor& x) { return x.sigmoid(); }
};

class Tanh : public Module {
  public:
    virtual Tensor forward(const Tensor& x) { return x.tanh(); }
};

// TODO it'd be better to have type(modules) == vector<Module*>
template <typename... Args> class Sequential : public Module {
    std::tuple<Args...> modules;

  public:
    Sequential(Args... args) : modules(args...) {}
    virtual Tensor forward(const Tensor& x) {
        Tensor temp = x;
        return std::apply(
            [&](Args&... t) {
                ((temp = t.forward(temp)), ...);
                return temp;
            },
            modules);
    }
    virtual vector<Module*> submodules() {
        return std::apply([](Args&... t) { return vector<Module*>{{(&t)...}}; }, modules);
    }
};

// TODO assumes filter is same size in each direction
class Conv2d : public Module {
    const int inDepth, outDepth, filterHeight, filterWidth, padding, stride;
    const bool bias;
    Tensor W, b;

  public:
    Conv2d(int inDepth, int outDepth, int filterSize, int stride, int padding, bool bias = true)
        : inDepth(inDepth), outDepth(outDepth), filterHeight(filterSize), filterWidth(filterSize), stride(stride), padding(padding),
          bias(bias), W{xt::random::randn<double>({inDepth * filterHeight * filterWidth, outDepth})}, b{xt::zeros<double>({outDepth})} {}
    virtual Tensor forward(const Tensor& x) {
        const int batchSize = x.shape(0);
        const int inHeight = x.shape(2) + 2 * padding;
        const int inWidth = x.shape(3) + 2 * padding;
        const int outHeight = (inHeight - filterHeight) / stride + 1;
        const int outWidth = (inWidth - filterWidth) / stride + 1;

        auto y = x;
        if (padding) {
            y = y.pad(padding);
        }
        y = y.im2row(inDepth, filterHeight, stride).dot(W);
        if (bias) {
            y = y + b;
        }
        return y.reshape({batchSize, outHeight, outWidth, outDepth}).transpose({0, 3, 1, 2});
    }

    virtual std::string name() const { return "Conv2d"; }

  protected:
    virtual vector<Tensor> getParameters() const {
        if (bias)
            return {W, b};
        return {W};
    }
};

// This assumes that stride divides (I - 2 * P + K).
// See the theano convolution arithmetic article for details.
// ConvTranspose2d is a sort of inverse to Conv2d w.r.t. shape.
// But Conv2d outputs the same shape for multiple input shapes.
// So, ConvTranspose2d returns the smallest among all the shapes
// that Conv2d sends to the same shape.
// TODO ignoring bias
class ConvTranspose2d : public Module {
    const int inDepth, outDepth, filterHeight, filterWidth, padding, stride;
    const bool bias;
    Tensor W, b;

  public:
    ConvTranspose2d(int outDepth, int inDepth, int filterSize, int stride, int padding, bool bias = true)
        : inDepth(inDepth), outDepth(outDepth), filterHeight(filterSize), filterWidth(filterSize), padding(padding), stride(stride),
          bias(bias), W{xt::random::randn<double>({outDepth, inDepth * filterHeight * filterWidth})}, b{xt::zeros<double>({inDepth})} {}
    virtual Tensor forward(const Tensor& x) {
        // compute the shape of the tensor we will return
        auto outShape = x.shape();
        auto inShape = outShape;
        inShape[0] = outShape[0];
        inShape[1] = inDepth;
        // size of input after dilating and padding in conv impl of transposed conv
        inShape[2] = (outShape[2] - 1) * stride + 1 + (filterHeight - 1);
        // also can be seen as just solving for inWidth in the outWidth computation in Conv2d
        inShape[3] = (outShape[3] - 1) * stride + filterWidth;

        // x == (N,K,P,Q)
        auto y = x.transpose({0, 2, 3, 1}); // (N,P,Q,K)
        y = y.reshape({-1, outDepth});      // (NPQ,K)
        // y = y.dot(W) + b;                                   // (NPQ,CRS)
        y = y.dot(W);                                          // (NPQ,CRS)
        y = y.row2im(inShape, outShape, filterHeight, stride); // (N,C,H+2*pad,W+2*pad)
        if (padding)
            return y.unpad(padding); // height = ((P - 1) * stride + 1) + (K - 1) - 2*pad = (P-1)*stride + K - 2 * pad // (N,C,H,W)
        else
            return y;
    }
    virtual std::string name() const { return "ConvTranspose2d"; }

  protected:
    virtual vector<Tensor> getParameters() const {
        if (bias)
            return {W, b};
        return {W};
    }
};

class BatchNorm2d : public Module {
    const int numFeatures;
    const double eps;
    const double momentum;
    const bool affine;
    const bool trackRunningStats;
    bool inTrainMode = true;

    Tensor gamma, beta, runningMean, runningVar;

  public:
    BatchNorm2d(int numFeatures, double eps = 1e-05, double momentum = 0.1, bool affine = true, bool trackRunningStats = false)
        : numFeatures(numFeatures), eps(eps), momentum(momentum), trackRunningStats(trackRunningStats), affine(affine),
          gamma(xt::random::randn<double>({1, numFeatures, 1, 1})), beta(xt::zeros<double>({1, numFeatures, 1, 1})),
          runningMean(xt::zeros<double>({1, numFeatures, 1, 1})), runningVar(xt::zeros<double>({1, numFeatures, 1, 1})) {}
    virtual Tensor forward(const Tensor& x) {
        const vector<shared_ptr<Node>>& children = {x.node, gamma.node, beta.node, runningMean.node, runningVar.node};
        return {make_shared<BatchNorm2dNode>(children, inTrainMode, eps, momentum, affine, trackRunningStats)};
    }

    virtual std::string name() const { return "BatchNorm2d"; }

  protected:
    virtual vector<Tensor> getParameters() const {
        if (affine)
            return {gamma, beta};
        return {};
    }
};

} // namespace xtorch
