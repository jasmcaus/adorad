/**
 * @file tests/split_data_test.cpp
 * @author Tham Ngap Wei
 *
 * Test the SplitData method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

/**
 * Compare the data after train test split.  This assumes that the labels
 * correspond to each column, so that we can easily check each point against its
 * original.
 *
 * @param inputData The original data set before split.
 * @param compareData The data want to compare with the inputData;
 *   it could be train data or test data.
 * @param inputLabel The labels of each point in compareData.
 */
void CompareData(const mat& inputData,
                 const mat& compareData,
                 const Row<size_t>& inputLabel)
{
  for (size_t i = 0; i != compareData.n_cols; ++i)
  {
    const mat& lhsCol = inputData.col(inputLabel(i));
    const mat& rhsCol = compareData.col(i);
    for (size_t j = 0; j != lhsCol.n_rows; ++j)
    {
      if (std::abs(rhsCol(j)) < 1e-5)
        REQUIRE(lhsCol(j) == Approx(0.0).margin(1e-5));
      else
        REQUIRE(lhsCol(j) == Approx(rhsCol(j)).epsilon(1e-7));
    }
  }
}

void CheckMatEqual(const mat& inputData,
                   const mat& compareData)
{
  const mat& sortedInput = arma::sort(inputData, "ascend", 1);
  const mat& sortedCompare = arma::sort(compareData, "ascend", 1);
  for (size_t i = 0; i < sortedInput.n_cols; ++i)
  {
    const mat& lhsCol = sortedInput.col(i);
    const mat& rhsCol = sortedCompare.col(i);
    for (size_t j = 0; j < lhsCol.n_rows; ++j)
    {
      if (std::abs(rhsCol(j)) < 1e-5)
        REQUIRE(lhsCol(j) == Approx(0.0).margin(1e-5));
      else
        REQUIRE(lhsCol(j) == Approx(rhsCol(j)).epsilon(1e-7));
    }
  }
}

/**
 * Check that no labels have been duplicated.
 */
void CheckDuplication(const Row<size_t>& trainLabels,
                      const Row<size_t>& testLabels)
{
  // Assemble a vector that will hold the counts of each element.
  Row<size_t> counts(trainLabels.n_elem + testLabels.n_elem);
  counts.zeros();

  for (size_t i = 0; i < trainLabels.n_elem; ++i)
  {
    REQUIRE(trainLabels[i] < counts.n_elem);
    counts[trainLabels[i]]++;
  }
  for (size_t i = 0; i < testLabels.n_elem; ++i)
  {
    REQUIRE(testLabels[i] < counts.n_elem);
    counts[testLabels[i]]++;
  }

  // Now make sure each point has been used once.
  for (size_t i = 0; i < counts.n_elem; ++i)
    REQUIRE(counts[i] == 1);
}

TEST_CASE("SplitShuffleDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  size_t count = 0; // Counter for unique sequential values.
  input.imbue([&count] () { return ++count; });

  const auto value = Split(input, 0.2);
  REQUIRE(std::get<0>(value).n_cols == 8); // Train data.
  REQUIRE(std::get<1>(value).n_cols == 2); // Test data.

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  CheckMatEqual(input, concat);
}

TEST_CASE("SplitDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  size_t count = 0; // Counter for unique sequential values.
  input.imbue([&count] () { return ++count; });

  const auto value = Split(input, 0.2, false);
  REQUIRE(std::get<0>(value).n_cols == 8); // Train data.
  REQUIRE(std::get<1>(value).n_cols == 2); // Test data.

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("ZeroRatioSplitData", "[SplitDataTest]")
{
  mat input(2, 10);
  size_t count = 0; // Counter for unique sequential values.
  input.imbue([&count] () { return ++count; });

  const auto value = Split(input, 0, false);
  REQUIRE(std::get<0>(value).n_cols == 10); // Train data.
  REQUIRE(std::get<1>(value).n_cols == 0); // Test data.

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("TotalRatioSplitData", "[SplitDataTest]")
{
  mat input(2, 10);
  size_t count = 0; // Counter for unique sequential values.
  input.imbue([&count] () { return ++count; });

  const auto value = Split(input, 1, false);
  REQUIRE(std::get<0>(value).n_cols == 0); // Train data.
  REQUIRE(std::get<1>(value).n_cols == 10); // Test data.

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("SplitLabeledDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  input.randu();

  // Set the labels to the column ID, so that CompareData can compare the data
  // after Split is called.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  const auto value = Split(input, labels, 0.2);
  REQUIRE(std::get<0>(value).n_cols == 8);
  REQUIRE(std::get<1>(value).n_cols == 2);
  REQUIRE(std::get<2>(value).n_cols == 8);
  REQUIRE(std::get<3>(value).n_cols == 2);

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));

  // The last thing to check is that we aren't duplicating any points in the
  // train or test labels.
  CheckDuplication(std::get<2>(value), std::get<3>(value));
}

/**
 * The same test as above, but on a larger dataset.
 */
TEST_CASE("SplitDataLargerTest", "[SplitDataTest]")
{
  size_t count = 0;
  mat input(10, 497);
  input.imbue([&count] () { return ++count; });

  const auto value = Split(input, 0.3);
  REQUIRE(std::get<0>(value).n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(std::get<1>(value).n_cols == size_t(0.3 * 497));

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  CheckMatEqual(input, concat);
}

TEST_CASE("SplitLabeledDataLargerTest", "[SplitDataTest]")
{
  mat input(10, 497);
  input.randu();

  // Set the labels to the column ID.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  const auto value = Split(input, labels, 0.3);
  REQUIRE(std::get<0>(value).n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(std::get<1>(value).n_cols == size_t(0.3 * 497));
  REQUIRE(std::get<2>(value).n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(std::get<3>(value).n_cols == size_t(0.3 * 497));

  CompareData(input, std::get<0>(value), std::get<2>(value));
  CompareData(input, std::get<1>(value), std::get<3>(value));

  CheckDuplication(std::get<2>(value), std::get<3>(value));
}

/**
 * Check that test ratio of 0 results in a full train set for stratified split.
 */
TEST_CASE("ZeroRatioStratifiedSplitData", "[SplitDataTest]")
{
  mat input(2, 15);
  input.randu();

  // Set the labels to 5 0s and 10 1s.
  const Row<size_t> labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  const double test_ratio = 0;

  const auto value = Split(input, labels, test_ratio, false, true);
  REQUIRE(std::get<0>(value).n_cols == 15);
  REQUIRE(std::get<1>(value).n_cols == 0);
  REQUIRE(std::get<2>(value).n_cols == 15);
  REQUIRE(std::get<3>(value).n_cols == 0);
}

/**
 * Check that test ratio of 1 results in a full test set for stratified split.
 */
TEST_CASE("TotalRatioStratifiedSplitData", "[SplitDataTest]")
{
  mat input(2, 15);
  input.randu();

  // Set the labels to 5 0s and 10 1s.
  const Row<size_t> labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  const double test_ratio = 1;

  const auto value = Split(input, labels, test_ratio, false, true);
  REQUIRE(std::get<0>(value).n_cols == 0);
  REQUIRE(std::get<1>(value).n_cols == 15);
  REQUIRE(std::get<2>(value).n_cols == 0);
  REQUIRE(std::get<3>(value).n_cols == 15);
}

/**
 * Check if data is stratified according to labels.
 */
TEST_CASE("StratifiedSplitDataResultTest", "[SplitDataTest]")
{
  mat input(5, 24);
  input.randu();

  // Set the labels to 4 0s, 8 1s and 12 2s.
  const Row<size_t> labels = { 0, 0, 0, 0,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
  const double test_ratio = 0.25;

  const auto value = Split(input, labels, test_ratio, true, true);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 0)).n_rows == 3);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 1)).n_rows == 6);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 2)).n_rows == 9);

  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 0)).n_rows == 1);
  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 1)).n_rows == 2);
  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 2)).n_rows == 3);

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  CheckMatEqual(input, concat);
}

/**
 * Check if data is stratified according to labels on a larger data set.
 * Example calculation to find resultant number of samples in the train and
 * test set:
 *
 * Since there are 256 0s and the test ratio is 0.3,
 * Number of 0s in the test set = 76 ( floor(256 * 0.3) = floor(76.8) ).
 * Number of 0s in the train set = 180 ( 256 - 76 ).
 */
TEST_CASE("StratifiedSplitLargerDataResultTest", "[SplitDataTest]")
{
  mat input(3, 480);
  input.randu();

  // 256 0s, 128 1s, 64 2s and 32 3s.
  Row<size_t> zero_label(256);
  Row<size_t> one_label(128);
  Row<size_t> two_label(64);
  Row<size_t> three_label(32);

  zero_label.fill(0);
  one_label.fill(1);
  two_label.fill(2);
  three_label.fill(3);

  Row<size_t> labels = arma::join_rows(zero_label, one_label);
  labels = arma::join_rows(labels, two_label);
  labels = arma::join_rows(labels, three_label);
  const double test_ratio = 0.3;

  const auto value = Split(input, labels, test_ratio, false, true);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 0)).n_rows == 180);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 1)).n_rows == 90);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 2)).n_rows == 45);
  REQUIRE(static_cast<uvec>(find(std::get<2>(value) == 3)).n_rows == 23);

  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 0)).n_rows == 76);
  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 1)).n_rows == 38);
  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 2)).n_rows == 19);
  REQUIRE(static_cast<uvec>(find(std::get<3>(value) == 3)).n_rows == 9);

  mat concat = arma::join_rows(std::get<0>(value), std::get<1>(value));
  CheckMatEqual(input, concat);
}
