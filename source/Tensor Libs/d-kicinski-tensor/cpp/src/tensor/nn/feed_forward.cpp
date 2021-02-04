#include "feed_forward.hpp"

namespace ts {

FeedForward::FeedForward(int dim_in, int dim_out, bool activation, float alpha)
    : _alpha(alpha), _activation(activation)
{
   _weights = ts::Matrix::randn({dim_in, dim_out});
   _bias = ts::Vector(dim_out);
}

auto FeedForward::operator()(Matrix const &inputs) -> Matrix { return forward(inputs); }

auto FeedForward::forward(Matrix const &inputs) -> Matrix
{
    _x = inputs;
    _y = ts::add(ts::dot(_x, _weights), _bias);
    if (_activation) {
        _y = ts::maximum(0.0f, _y);  // np.maximum(0, _y)
    }
    return _y;
}

auto FeedForward::backward(Matrix d_y) -> Matrix
{
    if (_activation) {
        d_y = ts::assign_if(d_y, _y <= 0, 0.0f);  // d_y[_y <= 0] = 0;
    }
    _d_weights = ts::dot(_x, d_y, true);
    _d_bias = ts::sum(d_y, 0);

    return ts::dot(d_y, _weights, false, true);
}

auto FeedForward::update(float step_size) -> void
{
    _d_weights += ts::multiply(_weights, _alpha);
    _weights +=  ts::multiply(_d_weights, -step_size);
    _bias += ts::multiply(_d_bias, -step_size);
}

auto FeedForward::weights() -> Matrix { return _weights; }

}
