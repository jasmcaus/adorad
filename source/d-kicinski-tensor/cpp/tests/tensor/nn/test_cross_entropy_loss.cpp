#include <catch2/catch.hpp>

#include <tensor/nn/cross_entropy_loss.hpp>

using namespace ts;

TEST_CASE("CrossEntropyLoss:: forward, backward")
{
    Matrix w1(2, 100);
    Matrix w2(100, 3);
    CrossEntropyLoss loss({w1, w2}, 16);

    Matrix probabilities = ts::Matrix::randn({300, 3});
    Tensor<int, 1> labels = ts::randint<1>(0, 2, {300});

    float loss_value = loss.forward(probabilities, labels);
    Matrix d_probabilities = loss.backward(probabilities);

    REQUIRE(loss_value != 0.0);
}

TEST_CASE("sanity-check: scores[range(batch_size), labels] -= 1")
{
    // This operation is done in loss computation and I want to be sure it is correct
    Matrix scores = {{1, 1},
                     {1, 1},
                     {1, 1}};
    Tensor<int, 1> labels = {0, 1, 0};
    Matrix expected = {{0, 1},
                       {1, 0},
                       {0, 1}};
    auto result = ts::apply_if(scores, ts::to_one_hot(labels),
                                 (Fn<float>) [](float e) { return e - 1; });
    REQUIRE(result == expected);
}

TEST_CASE("sanity-check: ts::sum(ts::pow(tensor, 2)")
{
   Matrix matrix = {{2, 2},
                    {2, 2}};
   float result = ts::sum(ts::pow(matrix, 2));
   REQUIRE(result == 16.0);
}