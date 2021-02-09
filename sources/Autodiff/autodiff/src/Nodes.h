#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <xadapt.hpp>
#include <xlinalg.hpp>
#include <xmanipulation.hpp>
#include <xnoalias.hpp>
#include <xpad.hpp>
#include <xrandom.hpp>
#include <xtensor.hpp>
#include <xmath.hpp>

namespace xtorch {
using std::make_shared;
using std::shared_ptr;
using std::vector;

// xt::noalias is used to avoid temporaries in expressions like b = a + c when a,b,c are preallocated xarrays/xtensors
// see https://xtensor.readthedocs.io/en/latest/container.html#aliasing-and-temporaries

struct Node {
    // the value the node represents
    xt::xarray<double> value;

    // the gradient of the loss w.r.t. this node's value
    xt::xarray<double> grad;

    vector<shared_ptr<Node>>& children = c; // for convenience

    // builds a node given it's value, only used when building a ParamNode
    Node(const xt::xarray<double>& value) : value(value) {}

    // builds a node given it's children
    Node(const vector<shared_ptr<Node>>& c) : c(c) {}

    bool isLeaf() const { return children.size() == 0; }

    // replace any edge from this to toReplace with an edge from this to replacement
    void replaceChild(const shared_ptr<Node>& toReplace, const shared_ptr<Node>& replacement) {
        for (auto& child : children) {
            if (toReplace == child) {
                child = replacement;
            }
        }
    }

    // computes value of this node given the value of its children
    virtual void forward() = 0;

    // computes the gradient of the loss w.r.t. the children of the node
    virtual void backward() = 0;

    // updates values for all nodes reachable from this
    void recompute(const shared_ptr<Node>& doNotDescend = nullptr) {
        for (const auto& child : children) {
            if (child == doNotDescend) {
                continue;
            }
            child->recompute(doNotDescend);
        }
        forward();
    }
    virtual ~Node() {}

  protected:
    vector<shared_ptr<Node>> c; // children
};

// A node whose value is given at construction instead of being computed from children nodes.
struct ParamNode : Node {
    using Node::Node;
    virtual void backward() {}
    virtual void forward() {}
};

struct AddNode : Node {
    const bool isBroadcast;
    xt::svector<size_t> lshape;
    xt::svector<size_t> rshape;

    bool lstretched = false;
    bool lpadded = false;

    bool rstretched = false;
    bool rpadded = false;

    xt::svector<size_t> lstretchedDims;
    xt::svector<size_t> rstretchedDims;

    xt::svector<size_t> lpaddedShape;
    xt::svector<size_t> rpaddedShape;
    AddNode(const vector<shared_ptr<Node>>& children) : Node(children), isBroadcast(c[0]->value.shape() != c[1]->value.shape()) {
        if (isBroadcast) {
            lshape = c[0]->value.shape();
            rshape = c[1]->value.shape();

            lpaddedShape = lshape;
            rpaddedShape = rshape;

            if (lshape.size() < rshape.size()) {
                lpadded = true;
                while (lpaddedShape.size() < rshape.size()) {
                    lpaddedShape.insert(lpaddedShape.begin(), 1u);
                }
            } else if (rshape.size() < lshape.size()) {
                rpadded = true;
                while (rpaddedShape.size() < lshape.size()) {
                    rpaddedShape.insert(rpaddedShape.begin(), 1u);
                }
            }
            assert(lpaddedShape.size() == rpaddedShape.size());
            for (int i = 0; i < lpaddedShape.size(); ++i) {
                if (lpaddedShape[i] < rpaddedShape[i] && lpaddedShape[i] == 1u) {
                    lstretched = true;
                    lstretchedDims.push_back(i);
                }
                if (rpaddedShape[i] < lpaddedShape[i] && rpaddedShape[i] == 1u) {
                    rstretched = true;
                    rstretchedDims.push_back(i);
                }
            }
        }
    }
    virtual void forward() { xt::noalias(value) = c[0]->value + c[1]->value; }
    virtual void backward() {
        if (lstretched && lpadded) {
            c[0]->grad += xt::reshape_view(xt::sum(grad, lstretchedDims), lshape);
        } else if (lstretched) {
            c[0]->grad += xt::sum(grad, lstretchedDims);
        } else if (lpadded) {
            c[0]->grad += xt::reshape_view(grad, lshape);
        } else {
            c[0]->grad += grad;
        }

        if (rstretched && rpadded) {
            c[1]->grad += xt::reshape_view(xt::sum(grad, rstretchedDims), rshape);
        } else if (rstretched) {
            c[1]->grad += xt::sum(grad, rstretchedDims);
        } else if (rpadded) {
            c[1]->grad += xt::reshape_view(grad, rshape);
        } else {
            c[1]->grad += grad;
        }
    }
};

struct SubtractNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = c[0]->value - c[1]->value; }
    virtual void backward() {
        c[0]->grad += grad;
        c[1]->grad -= grad;
    }
};

struct MultiplyNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = c[0]->value * c[1]->value; }
    virtual void backward() {
        c[0]->grad += grad * c[1]->value;
        c[1]->grad += grad * c[0]->value;
    }
};

struct DivideNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = c[0]->value / c[1]->value; }
    virtual void backward() {
        c[0]->grad += grad / c[1]->value;
        c[1]->grad -= grad * c[0]->value / (c[1]->value * c[1]->value);
    }
};

struct DotNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = xt::linalg::dot(c[0]->value, c[1]->value); }
    virtual void backward() {
        c[0]->grad += xt::linalg::dot(grad, xt::transpose(c[1]->value));
        c[1]->grad += xt::linalg::dot(xt::transpose(c[0]->value), grad);
    }
};

struct PowNode : Node {
    const double power;
    PowNode(const shared_ptr<Node>& child, const double p) : Node({child}), power(p) {}
    virtual void forward() { xt::noalias(value) = xt::pow(c[0]->value, power); }
    virtual void backward() { c[0]->grad += power * xt::pow(c[0]->value, power - 1) * grad; }
};

struct SquareNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = xt::square(c[0]->value); }
    virtual void backward() { c[0]->grad += 2 * c[0]->value * grad; }
};

struct ReLUNode : Node {
    const double negativeSlope;
    ReLUNode(const shared_ptr<Node>& child, const double negativeSlope) : Node({child}), negativeSlope(negativeSlope) {}
    virtual void forward() { xt::noalias(value) = xt::maximum(c[0]->value, negativeSlope * c[0]->value); }
    virtual void backward() { c[0]->grad += xt::where(value > 0, grad, negativeSlope * grad); }
};

struct SinNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = xt::sin(c[0]->value); }
    virtual void backward() { c[0]->grad += grad * xt::cos(c[0]->value); }
};

struct SigmoidNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = 1. / (1. + xt::exp(-c[0]->value)); }
    virtual void backward() { c[0]->grad += grad * value * (1. - value); }
};

struct TanhNode : Node {
    using Node::Node;
    virtual void forward() { xt::noalias(value) = 2. / (1. + xt::exp(-2. * c[0]->value)) - 1.; }
    virtual void backward() { c[0]->grad += grad * (1. - xt::square(value)); }
};

// TODO Im2Row assumes filter is square and strides are equal in each axis
struct Im2RowNode : Node {
    using Node::Node;
    size_t stride, batchSize, inHeight, inWidth, outHeight, outWidth, inDepth, filterHeight, filterWidth;
    Im2RowNode(const shared_ptr<Node>& child, size_t inDepth, size_t filterHeight, size_t stride)
        : Node({child}), inDepth(inDepth), filterHeight(filterHeight), filterWidth(filterHeight), stride(stride) {
        auto xshape = c[0]->value.shape();
        batchSize = xshape[0];
        inHeight = xshape[2];
        inWidth = xshape[3];
        outHeight = (inHeight - filterHeight) / stride + 1;
        outWidth = (inWidth - filterWidth) / stride + 1;
        value = xt::zeros<double>({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
        grad = xt::zeros<double>({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
    }
    virtual void forward() {
        value.reshape({batchSize, outHeight, outWidth, inDepth, filterHeight, filterWidth});
        for (auto i = 0; i <= inHeight - filterHeight; i += stride) {
            for (auto j = 0; j <= inWidth - filterWidth; j += stride) {
                auto x = i / stride;
                auto y = j / stride;
                xt::view(value, xt::all(), x, y) =
                    xt::view(c[0]->value, xt::all(), xt::all(), xt::range(i, i + filterHeight), xt::range(j, j + filterWidth));
            }
        }
        value.reshape({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
    }
    virtual void backward() {
        grad.reshape({batchSize, outHeight, outWidth, inDepth, filterHeight, filterWidth});
        for (auto i = 0; i <= inHeight - filterHeight; i += stride) {
            for (auto j = 0; j <= inWidth - filterWidth; j += stride) {
                auto x = i / stride;
                auto y = j / stride;
                xt::view(c[0]->grad, xt::all(), xt::all(), xt::range(i, i + filterHeight), xt::range(j, j + filterWidth)) +=
                    xt::view(grad, xt::all(), x, y);
            }
        }
        grad.reshape({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
    }
};

struct Row2ImNode : Node {
    using Node::Node;
    size_t stride, batchSize, inHeight, inWidth, outHeight, outWidth, inDepth, filterHeight, filterWidth;
    // c[0]->value : (NPQ,CRS)
    // value       : (N,C,H,W)
    Row2ImNode(const shared_ptr<Node>& child, auto inShape, auto outShape, size_t filterHeight, size_t stride)
        : Node({child}), filterHeight(filterHeight), filterWidth(filterHeight), stride(stride) {
        // inShape = [N,C,H,W]
        // outShape = [N,K,P,Q]
        batchSize = inShape[0];
        inDepth = inShape[1];
        inHeight = inShape[2];
        inWidth = inShape[3];
        outHeight = outShape[2];
        outWidth = outShape[3];
        value = xt::zeros<double>({batchSize, inDepth, inHeight, inWidth});
        grad = xt::zeros<double>({batchSize, inDepth, inHeight, inWidth});
    }
    virtual void backward() {
        c[0]->grad.reshape({batchSize, outHeight, outWidth, inDepth, filterHeight, filterWidth});
        for (auto i = 0; i <= inHeight - filterHeight; i += stride) {
            for (auto j = 0; j <= inWidth - filterWidth; j += stride) {
                auto x = i / stride;
                auto y = j / stride;
                xt::view(c[0]->grad, xt::all(), x, y) +=
                    xt::view(grad, xt::all(), xt::all(), xt::range(i, i + filterHeight), xt::range(j, j + filterWidth));
            }
        }
        c[0]->grad.reshape({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
    }
    virtual void forward() {
        value.fill(0.);
        c[0]->value.reshape({batchSize, outHeight, outWidth, inDepth, filterHeight, filterWidth});
        for (auto i = 0; i <= inHeight - filterHeight; i += stride) {
            for (auto j = 0; j <= inWidth - filterWidth; j += stride) {
                auto x = i / stride;
                auto y = j / stride;
                xt::view(value, xt::all(), xt::all(), xt::range(i, i + filterHeight), xt::range(j, j + filterWidth)) +=
                    xt::view(c[0]->value, xt::all(), x, y);
            }
        }
        c[0]->value.reshape({batchSize * outHeight * outWidth, inDepth * filterHeight * filterWidth});
    }
};

struct ReshapeNode : Node {
    xt::svector<size_t> inShape;
    xt::svector<int> outShape;
    ReshapeNode(const shared_ptr<Node>& child, const vector<int>& outShape)
        : Node({child}), outShape(outShape), inShape(child->value.shape()) {}
    virtual void forward() {
        c[0]->value.reshape(outShape);
        value = c[0]->value;
        c[0]->value.reshape(inShape);
    }
    virtual void backward() {
        grad.reshape(inShape);
        c[0]->grad += grad;
        grad.reshape(outShape);
    }
};

struct TransposeNode : Node {
    const vector<size_t> perm;
    vector<size_t> invPerm;
    TransposeNode(const shared_ptr<Node>& child, const vector<size_t>& perm) : Node({child}), perm(perm) {
        xt::noalias(value) = xt::transpose(child->value, perm);
        // Think of the permutation as a directed bipartite graph.
        // Then to get back to the arrangement you started at just reverse the edges
        // perm[i] is the to node on the edge (i,perm[i])
        invPerm.resize(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            invPerm[perm[i]] = i;
        }
    }
    virtual void forward() { xt::noalias(value) = xt::transpose(c[0]->value, perm); }
    virtual void backward() { c[0]->grad += xt::transpose(grad, invPerm); }
};

struct ImagePadNode : Node {
    size_t pad;
    ImagePadNode(const shared_ptr<Node>& child, size_t pad) : Node({child}), pad(pad) {
        auto shape = c[0]->value.shape();
        shape[2] += 2 * pad;
        shape[3] += 2 * pad;
        value = xt::zeros<double>(shape);
    }

    virtual void forward() { xt::view(value, xt::all(), xt::all(), xt::range(pad, -pad), xt::range(pad, -pad)) = c[0]->value; }
    virtual void backward() { c[0]->grad += xt::view(grad, xt::all(), xt::all(), xt::range(pad, -pad), xt::range(pad, -pad)); }
};

struct ImageUnPadNode : Node {
    size_t pad;
    ImageUnPadNode(const shared_ptr<Node>& child, size_t pad) : Node({child}), pad(pad) {
        auto shape = c[0]->value.shape();
        shape[2] -= 2 * pad;
        shape[3] -= 2 * pad;
        value = xt::zeros<double>(shape);
    }

    virtual void forward() { value = xt::view(c[0]->value, xt::all(), xt::all(), xt::range(pad, -pad), xt::range(pad, -pad)); }
    virtual void backward() { xt::view(c[0]->grad, xt::all(), xt::all(), xt::range(pad, -pad), xt::range(pad, -pad)) += grad; }
};

struct SumNode : Node {
    const vector<int> axis;
    xt::svector<size_t> unsqueezed;
    xt::svector<size_t> squeezed;
    SumNode(const shared_ptr<Node>& child, vector<int> axis) : Node({child}), axis(axis) {
        if (axis.size()) {
            value = xt::zeros<double>(xt::sum(c[0]->value, axis).shape());
            grad = xt::zeros<double>(xt::sum(c[0]->value, axis).shape());
            unsqueezed = c[0]->value.shape();
            for (int i = 0; i < axis.size(); ++i) {
                unsqueezed[axis[i]] = 1;
            }
            squeezed = grad.shape();
        } else {
            value = 0.;
            grad = 0.;
        }
    }
    virtual void forward() {
        if (axis.size()) {
            xt::noalias(value) = xt::sum(c[0]->value, axis);
        } else {
            xt::noalias(value) = xt::sum(c[0]->value);
        }
    }
    virtual void backward() {
        if (axis.size()) {
            grad.reshape(unsqueezed);
            c[0]->grad += grad;
            grad.reshape(squeezed);
        } else {
            c[0]->grad += grad[0];
        }
    }
};

// TODO ignoring runningStats right now because I am not using them anywhere else
struct BatchNorm2dNode : Node {
    const double eps;
    const double momentum;
    const bool affine;
    const bool trackRunningStats;
    const int batchSize;
    bool& inTrainMode;

    xt::xarray<double> xNorm; // size == x.size
    xt::xarray<double> std;   // size == gamma.size
    BatchNorm2dNode(const vector<shared_ptr<Node>>& children, bool& inTrainMode, double eps = 1e-05, double momentum = 0.1,
                    bool affine = true, bool trackRunningStats = true)
        : Node(children), eps(eps), momentum(momentum), trackRunningStats(trackRunningStats), affine(affine),
          batchSize(c[0]->value.shape(0)), inTrainMode(inTrainMode) {}
    virtual void forward() {
        // x = c[0]
        // gamma = c[1]
        // beta = c[2]
        // runningMean = c[3]
        // runningVar = c[4]
        if (inTrainMode || !trackRunningStats) {

            // calculate mean
            xt::noalias(std) = xt::view(xt::mean(c[0]->value, {0, 2, 3}), xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());
            if (trackRunningStats) {
                c[3]->value += momentum * (std - c[3]->value);
            }
            // calculate std
            xt::noalias(xNorm) = c[0]->value - std;
            xt::noalias(std) = xt::view(xt::mean(xt::square(xNorm), {0, 2, 3}), xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());
            if (trackRunningStats) {
                c[4]->value += momentum * (std - c[4]->value);
            }
            // normalize
            std = xt::sqrt(std + eps);
            xNorm /= std;
            if (affine) {
                xt::noalias(value) = c[1]->value * xNorm + c[2]->value;
            } else {
                xt::noalias(value) = xNorm;
            }

        } else {
            xNorm = (c[0]->value - c[3]->value) / xt::sqrt(c[4]->value + eps);
            if (affine) {
                xt::noalias(value) = c[1]->value * xNorm + c[2]->value;
            } else {
                xt::noalias(value) = xNorm;
            }
        }
    }
    // costapt.github.io/2016/07/09/batch-norm-alt/
    virtual void backward() {
        xt::noalias(c[1]->grad) = xt::view(xt::sum(grad * xNorm, {0, 2, 3}), xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());
        xt::noalias(c[2]->grad) = xt::view(xt::sum(grad, {0, 2, 3}), xt::newaxis(), xt::all(), xt::newaxis(), xt::newaxis());
        const double N = (batchSize * xNorm.shape(2) * xNorm.shape(3));
        xt::noalias(c[0]->grad) += (c[1]->value / std) * (grad - xNorm * (c[1]->grad / N) - (c[2]->grad / N));
    }
};

struct MSELossNode : Node {
    size_t batchSize;
    MSELossNode(const vector<shared_ptr<Node>>& children) : Node(children), batchSize(c[0]->value.shape(0)) {}
    virtual void forward() {
        const auto& diff = c[0]->value - c[1]->value;
        xt::noalias(value) = (.5 / batchSize) * xt::sum(diff * diff);
    }
    virtual void backward() {
        const auto& res = grad * (c[0]->value - c[1]->value) / batchSize;
        c[0]->grad += res;
        c[1]->grad -= res;
    }
};

// https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
// https://gombru.github.io/2018/05/23/cross_entropy_loss/
// TODO not accumulating grad into labels. but that's probably ok for this project.
struct BCEWithLogitsLossNode : Node {
    size_t batchSize;
    BCEWithLogitsLossNode(const vector<shared_ptr<Node>>& children) : Node(children) {
        batchSize = c[0]->value.shape(0);
        assert(batchSize == c[1]->value.shape(0));
    }
    virtual void forward() {
        const auto& z = c[0]->value; // logit
        const auto& y = c[1]->value;
        xt::noalias(value) = (1. / batchSize) * xt::sum(xt::maximum(z, 0) - z * y + xt::log1p(xt::exp(-xt::abs(z))));
    }
    virtual void backward() {
        auto sig = 1 / (1 + xt::exp(-c[0]->value));
        xt::noalias(c[0]->grad) += grad / batchSize * (sig - c[1]->value);
    }
};

struct BCELossNode : Node {
    size_t batchSize;
    BCELossNode(const vector<shared_ptr<Node>>& children) : Node(children), batchSize(c[0]->value.shape(0)) {}
    virtual void forward() {
        const auto& x = c[0]->value;
        const auto& y = c[1]->value;
        xt::noalias(value) = (-1. / batchSize) * xt::sum(y * xt::log(x) + (1 - y) * xt::log(1 - x));
    }
    virtual void backward() { c[0]->grad += grad / (-batchSize) * (c[1]->value / c[0]->value - (1 - c[1]->value) / (1 - c[0]->value)); }
};

template <typename T> static shared_ptr<Node> opNode(const vector<shared_ptr<Node>>& children) { return make_shared<T>(children); }

} // namespace xtorch
