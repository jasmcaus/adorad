
#include <tensor/nn/cross_entropy_loss.hpp>
#include <tensor/nn/feed_forward.hpp>
#include <tensor/nn/planar_dataset.hpp>
#include <tensor/nn/softmax.hpp>

class Model {

  public:
    using MatrixF = ts::Tensor<float, 2>;
    using VectorI = ts::Tensor<int, 1>;

    auto predict(MatrixF const &inputs) -> VectorI { return ts::argmax(_forward(inputs)); }

    auto update(MatrixF const &inputs, VectorI const &labels, float learning_rate) -> float
    {
        // forward pass
        auto probabilities = _forward(inputs);
        float loss = _loss(probabilities, labels);

        // backward pass
        _layer1.backward(_layer2.backward(_loss.backward(probabilities)));

        // parameters update
        _layer1.update(learning_rate);
        _layer2.update(learning_rate);

        return loss;
    }

  private:
    ts::FeedForward _layer1 = ts::FeedForward(2, 100, true);
    ts::FeedForward _layer2 = ts::FeedForward(100, 3);
    ts::CrossEntropyLoss _loss = ts::CrossEntropyLoss({_layer1.weights(), _layer2.weights()});

    auto _forward(MatrixF const &inputs) -> MatrixF
    {
        auto logits = _layer2(_layer1(inputs));
        return ts::softmax(logits);
    }
};

auto train(Model &model, ts::PlanarDataset &dataset) -> float
{
    constexpr int epoch_num = 100; // Let's overfit to validate our code
    constexpr float learning_rate = 1e-0;
    float loss = std::numeric_limits<float>::max();

    for (int epoch_i = 0; epoch_i < epoch_num; ++epoch_i) {
        for (auto [inputs, labels] : dataset) {
            loss = model.update(inputs, labels, learning_rate);
            if (epoch_i % 10 == 0)
                std::cout << "epoch: " << epoch_i << "/" << epoch_num << " loss: " << loss
                          << std::endl;
        }
    }
    std::cout << "epoch: " << epoch_num << "/" << epoch_num << " loss: " << loss << std::endl;
    return loss;
}

auto label(Model &model, ts::PlanarDataset &dataset) -> Model::VectorI
{
    ts::Tensor<int, 1> labels;

    for (auto [inputs, _] : dataset) {
        auto batch_labels = model.predict(inputs);
        if (labels.data_size() == 0) {
            labels = batch_labels;
        }
        labels = ts::concatenate<int, 0>({labels, batch_labels});
    }
    return labels;
}

int main()
{

    ts::PlanarDataset dataset_train("resources/train_planar_data.tsv", true, 300);
    ts::PlanarDataset dataset_test("resources/test_planar_data.tsv", true, 1);
    Model model;

    std::cout << "Training... " << std::endl;
    float loss = train(model, dataset_train);
    std::cout << std::endl;

    std::cout << "Labeling test dataset... ";
    auto labels = label(model, dataset_test);
    std::cout << "done!" << std::endl;

    std::cout << "Dumping labels to 'resources/labels_planar_data.tsv'... ";
    if (std::ofstream output = std::ofstream("resources/labels_planar_data.tsv")) {
        for (auto label : labels) {
            output << label << std::endl;
        }
    }
    std::cout << "done!" << std::endl;
    return 0;
}