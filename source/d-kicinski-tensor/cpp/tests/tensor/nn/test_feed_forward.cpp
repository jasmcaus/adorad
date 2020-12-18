#include <catch2/catch.hpp>

#include <tensor/nn/cross_entropy_loss.hpp>
#include <tensor/nn/feed_forward.hpp>

using namespace ts;

TEST_CASE("Create FeedForward layer")
{
    FeedForward layer(2, 100);
    REQUIRE(true);
}

TEST_CASE("FeedForward: forward, backward")
{
    FeedForward layer(2, 3);
    Matrix input(32, 2);
    auto y = layer(input);
    {
        std::array<int, 2> expected_shape = {32, 3};
        REQUIRE(y.shape() == expected_shape);
    }
    auto d_y =layer.backward(y);
    {
        std::array<int, 2> expected_shape = {32, 2};
        REQUIRE(d_y.shape() == expected_shape);
    }
    layer.update(0.001);
}
