#include <catch2/catch.hpp>
#include <tensor/nn/planar_dataset.hpp>

constexpr auto DATASET_PATH = "resources/example_planar_data.tsv";

TEST_CASE("load dataset")
{
    auto dataset = ts::PlanarDataset(DATASET_PATH, true);

    REQUIRE(dataset.size() == 6);
    REQUIRE(dataset.inputs().shape() == std::array{6, 2});
    REQUIRE(dataset.labels().shape() == std::array{6});
}

TEST_CASE("dataset iterator")
{
    auto dataset = ts::PlanarDataset(DATASET_PATH, true, 2);
    float i = 0;
    for (auto [inputs, labels] : dataset) {
        REQUIRE(inputs.shape() == std::array{2, 2});
        REQUIRE(labels.shape() == std::array{2});

        ts::Tensor<int, 1> expected_labels = {(int)i, (int)i};
        ts::Tensor<float, 2> expected_inputs = {{i+2, i+1},
                                                {i+2, i+1}};

        REQUIRE(inputs == expected_inputs);
        REQUIRE(labels == expected_labels);
        ++i;
    }
    REQUIRE(i == 3);
}