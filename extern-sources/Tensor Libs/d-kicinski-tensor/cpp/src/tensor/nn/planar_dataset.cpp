#include "planar_dataset.hpp"

ts::PlanarDataset::PlanarDataset(const std::string &path, bool header_line, int batch_size)
    : _batch_size(batch_size)
{
    if (std::ifstream input = std::ifstream(path)) {
        if (header_line) {
            std::string header;
            std::getline(input, header);
        }

        std::vector<float> inputs_x;
        std::vector<float> inputs_y;
        std::vector<int> labels;
        int line_num = 0;

        for (std::string line; std::getline(input, line); ++line_num) {
            // make a stream for the line itself
            std::istringstream in(line);

            int label;
            float x, y;
            in >> x >> y >> label;

            inputs_x.push_back(x);
            inputs_y.push_back(y);
            labels.push_back(label);
        }
        _inputs = ts::concatenate<float, 1>({ts::from_vector(inputs_x), ts::from_vector(inputs_y)});
        _labels = ts::from_vector(labels);
    } else {
        std::cerr << "Couldn't open file" << std::endl;
    }
}

auto ts::PlanarDataset::inputs() -> ts::Tensor<float, 2> { return _inputs; }

auto ts::PlanarDataset::labels() -> ts::Tensor<int, 1> { return _labels; }

auto ts::PlanarDataset::size() -> int { return _inputs.shape()[0]; }

auto ts::PlanarDataset::begin() -> ts::DatasetIterator
{
    return DatasetIterator(_inputs, _labels, _batch_size, 0);
}

auto ts::PlanarDataset::end() -> ts::DatasetIterator
{
    return DatasetIterator(_inputs, _labels, _batch_size, _inputs.shape()[0]);
}
