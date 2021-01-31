#pragma once

#include "Graph.h"
#include "Nodes.h"

namespace xtorch {
using std::make_shared;
using std::queue;
using std::shared_ptr;
using std::vector;

class Tensor {
  public:
    shared_ptr<Node> node;
    Tensor(const xt::xarray<double>& value) : node(make_shared<ParamNode>(value)) { node->grad = xt::zeros<double>(node->value.shape()); }

    Tensor relu(const double negativeSlope = 0.) const { return {make_shared<ReLUNode>(node, negativeSlope)}; }
    Tensor sin() const { return {opNode<SinNode>({node})}; }
    Tensor sigmoid() const { return {opNode<SigmoidNode>({node})}; }
    Tensor tanh() const { return {opNode<TanhNode>({node})}; }
    Tensor dot(const Tensor& x) const { return {opNode<DotNode>({node, x.node})}; }
    Tensor pow(const double p) const { return {make_shared<PowNode>(node, p)}; }
    Tensor square() const { return {opNode<SquareNode>({node})}; }
    Tensor sum(vector<int> axis = {}) const { return {make_shared<SumNode>(node, axis)}; }
    Tensor pad(int padding) const { return {make_shared<ImagePadNode>(node, padding)}; }
    Tensor unpad(int padding) const { return {make_shared<ImageUnPadNode>(node, padding)}; }
    Tensor reshape(const vector<int>& outShape) const { return {make_shared<ReshapeNode>(node, outShape)}; }
    Tensor transpose(const vector<size_t>& perm) const { return {make_shared<TransposeNode>(node, perm)}; }
    Tensor im2row(int inDepth, int filterSize, int stride) const { return {make_shared<Im2RowNode>(node, inDepth, filterSize, stride)}; }
    Tensor row2im(auto inShape, auto outShape, int filterSize, int stride) const {
        return {make_shared<Row2ImNode>(node, inShape, outShape, filterSize, stride)};
    }

    Tensor operator+(const Tensor& rhs) const { return {opNode<AddNode>({node, rhs.node})}; }
    Tensor operator+(const double c) const { return *this + Tensor(c); }
    Tensor operator-(const Tensor& rhs) const { return {opNode<SubtractNode>({node, rhs.node})}; }
    Tensor operator-(const double c) const { return *this - Tensor(c); }
    Tensor operator*(const Tensor& rhs) const { return {opNode<MultiplyNode>({node, rhs.node})}; }
    Tensor operator*(const double c) const { return *this * Tensor(c); }
    Tensor operator/(const Tensor& rhs) const { return {opNode<DivideNode>({node, rhs.node})}; }
    Tensor operator/(const double c) const { return *this * Tensor(1. / c); }

    xt::xarray<double> getValue() const { return node->value; }
    void setValue(xt::xarray<double> value) { node->value = value; }
    void fill(const double d) { node->value.fill(d); }

    auto shape() const { return node->value.shape(); }
    auto shape(int i) const { return node->value.shape(i); }

    double item() {
        assert(shape().size() == 0);
        return node->value[0];
    }

    // returns a new leaf node that has the same value as node. this copies (pytorch does not).
    Tensor detach() const { return {node->value}; }

    // TODO gradient does not need to flow down every path in the network.
    void backward() {
        assert(shape().size() == 0);

        // the root's grad is 1 since droot/droot = 1
        node->grad = 1.;

        vector<shared_ptr<Node>> nodesInTopologicalOrder = topologicalSort(node);
        for (const auto& u : nodesInTopologicalOrder) {
            u->backward();
            // we are done with this decendant, so clear its gradient if it is not a leaf
            if (!u->isLeaf()) {
                u->grad.fill(0.);
            }
        }
    }

  private:
    Tensor(const shared_ptr<Node>& node) : node(node) {
        node->forward();
        node->grad = xt::zeros<double>(node->value.shape());
    }

    void recompute() { node->recompute(); }

    friend Tensor relu(const Tensor& x);
    friend Tensor sin(const Tensor& x);
    friend class MSELoss;
    friend class SGD;
    friend class Module;
    friend class Conv2d;
    friend class ConvTranspose2d;
    friend class BatchNorm2d;
    friend class BCEWithLogitsLoss;
    friend class BCELoss;
};

Tensor operator*(const double c, const Tensor& t) { return Tensor(c) * t; }
Tensor operator/(const double c, const Tensor& t) { return Tensor(c) / t; }
std::ostream& operator<<(std::ostream& os, const xtorch::Tensor& x) { return os << x.getValue(); }
}; // namespace xtorch
