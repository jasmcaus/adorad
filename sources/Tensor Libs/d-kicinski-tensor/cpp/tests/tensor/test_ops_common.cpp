#include <functional>
#include <cmath>

#include <catch2/catch.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>


using namespace ts;
using namespace std::placeholders;

TEST_CASE("multiply: Matrix[2, 3] X scalar")
{
    Tensor<float, 2> matrix = {{1, 1, 1},
                               {1, 1, 1}};
    float scalar = 1337;

    Tensor<float, 2> expected = {{1337, 1337, 1337},
                                 {1337, 1337, 1337}};
    auto result = multiply(matrix, scalar);

    REQUIRE(result == expected);
}

TEST_CASE("add: Matrix[2, 3] x Matrix[2, 3]")
{
    Matrix t1 = {{1, 1, 1},
                 {1, 1, 1}};
    Matrix t2 = {{1, 1, 1},
                 {1, 1, 1}};
    Matrix expected = {{2, 2, 2},
                       {2, 2, 2}};
    auto result = ts::add(t1, t2);

    REQUIRE(result == expected);
}

TEST_CASE("add(Matrix, Vector, axis=0)")
{
    Matrix matrix = {{1, 1, 1},
                     {0, 0, 0}};
    Vector vector = {3, 3, 3};
    Matrix expected = {{4, 4, 4},
                       {3, 3, 3}};
    auto result = ts::add(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("divide(Matrix, Vector, axis=0)")
{
    Matrix matrix = {{3, 6, 9},
                     {3, 6, 9}};
    Vector vector = {3, 3};
    Matrix expected = {{1, 2, 3},
                       {1, 2, 3}};
    auto result = ts::divide(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("transpose")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};

    Matrix expected = {{1, 1},
                       {1, 1},
                       {1, 1}};
    auto result = ts::transpose(matrix);

    REQUIRE(result == expected);
}

TEST_CASE("maximum(scalar, Matrix)")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Matrix expected = {{1, 0, 1},
                       {1, 0, 1}};
    auto result = ts::maximum(0.0f, matrix);

    REQUIRE(result == expected);
}

TEST_CASE("mask from tensor")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Tensor<bool, 2> expected = {{true, false, true},
                                {true, false, true}};

    auto mask = ts::mask<float>(matrix, [](float e) { return e >= 0; });

    REQUIRE(mask == expected);
}

TEST_CASE("assign_if")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Matrix expected = {{1, 1337, 1},
                       {1, 1337, 1}};

    auto result = assign_if(matrix, matrix < 0, 1337.0f);

    REQUIRE(result == expected);
}

TEST_CASE("sum(Matrix, Vector, axis=0)")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};
    Vector expected = {2, 2, 2};

    auto result = ts::sum(matrix, 0);

    REQUIRE(result == expected);
}

TEST_CASE("sum(Matrix, Vector, axis=1)")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};
    Vector expected = {3, 3};

    auto result = ts::sum(matrix, 1);

    REQUIRE(result == expected);
}

TEST_CASE("sum(Tensor)")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};
    auto result = ts::sum(matrix);
    REQUIRE(result == 6);
}

TEST_CASE("to_one_hot")
{
    Tensor<int, 1> vector = {2, 0, 1};
    Matrix matrix = {{1, 2, 3},
                     {1, 2, 3},
                     {1, 2, 3}};

    Tensor<bool, 2> expected = {{false, false, true},
                                {true, false, false},
                                {false, true, false}
    };
    Tensor<bool, 2> one_hot = ts::to_one_hot(vector);

    REQUIRE(one_hot == expected);
}

TEST_CASE("apply_if")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Matrix expected = {{1, 1, 1},
                       {1, 1, 1}};

    auto result = ts::apply_if(matrix, matrix < 0,
                               (Fn<float>)[](float e) { return std::abs(e); });

    REQUIRE(result == expected);
}

TEST_CASE("get(Matrix, Vector)")
{
    Matrix matrix = {{1, 2, 3},
                     {1, 2, 3}};
    Tensor<int, 1> vector = {2, 0};
    Vector expected = {3, 1};

    auto result = ts::get(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("apply")
{
    Matrix matrix = {{1, 2, 3},
                     {1, 2, 3}};

    Matrix expected = {{1, 4, 9},
                       {1, 4, 9}};

    Tensor<float, 2> result = ts::apply<float, 2>(matrix, (Fn<float>)[](float e){ return std::pow(e, 2); });

    REQUIRE(result == expected);
}

TEST_CASE("ts::log")
{
    Matrix matrix = {{1, 2, 3},
                     {1, 2, 3}};
    Tensor<float, 2> result = ts::log(matrix);
}

TEST_CASE("randint")
{
    auto tensor = ts::randint<2>(0, 2, {300, 3});
    for (auto const &e: tensor) {
        REQUIRE((e >= 0 and e <= 2));
    }
}

TEST_CASE("from_vector")
{
   std::vector<float> std_v = {1,2,3,4,5};
   Vector vector = ts::from_vector(std_v);

   REQUIRE(std::equal(std_v.begin(), std_v.end(), vector.begin()));
   REQUIRE(vector.shape() == std::array{5});
}

TEST_CASE("concatenate vectors with axis = 1")
{
   Vector v0 = {0, 0, 0};
   Vector v1 = {1, 1, 1};
   Vector v2 = {2, 2, 2};
   Matrix expected = {{0, 1, 2},
                      {0, 1, 2},
                      {0, 1, 2}};

   Tensor<float, 2> result = ts::concatenate<float, 1>({v0, v1, v2});

   REQUIRE(result == expected);
}

TEST_CASE("concatenate vectors with axis = 0")
{
    Vector v0 = {0, 0, 0};
    Vector v1 = {1, 1, 1};
    Vector v2 = {2, 2, 2};
    Vector expected = {0, 0, 0, 1, 1, 1, 2, 2, 2};

    Tensor<float, 1> result = ts::concatenate<float, 0>({v0, v1, v2});

    REQUIRE(result == expected);
}

TEST_CASE("concatenate vectors with axis = 0 and different sizes")
{
    Vector v0 = {0, 0, 0};
    Vector v1 = {1, 1};
    Vector v2 = {2};
    Vector expected = {0, 0, 0, 1, 1, 2};

    Tensor<float, 1> result = ts::concatenate<float, 0>({v0, v1, v2});

    REQUIRE(result == expected);
}

TEST_CASE("slice")
{
    Matrix matrix = {{0, 0, 0},
                     {1, 1, 1},
                     {2, 2, 2}};

    {
        Matrix slice = ts::slice(matrix, 0, 2);
        Matrix expected = {{0, 0, 0},
                           {1, 1, 1}};
        REQUIRE(slice == expected);
    }

    {
        Matrix slice = ts::slice(matrix, 1, 3);
        Matrix expected = {{1, 1, 1},
                           {2, 2, 2}};
        REQUIRE(slice == expected);
    }
}

TEST_CASE("argmax(Matrix)")
{
    Tensor<float, 2> matrix = {{0, 1, 0},
                               {3, 2, 1},
                               {0, 0, 1}};
    Tensor<int, 1> expected = {1, 0, 2};
    Tensor<int, 1> indexes = ts::argmax(matrix);

    REQUIRE(indexes == expected);
}
