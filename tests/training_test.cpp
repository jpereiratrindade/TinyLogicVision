#include "tinyvision/dataset.hpp"
#include "tinyvision/model.hpp"

#include <iostream>

int main() {
    auto dataset = tinyvision::make_synthetic_dataset(20, 11);
    tinyvision::MLP model(dataset.width * dataset.height * dataset.channels, 24, 4, 7);

    for (std::size_t epoch = 0; epoch < 180; ++epoch) {
        for (const auto& sample : dataset.samples) {
            model.train_one(sample.rgb, sample.label, 0.025);
        }
    }

    std::size_t correct = 0;
    for (const auto& sample : dataset.samples) {
        correct += model.predict_class(sample.rgb) == sample.label ? 1 : 0;
    }
    const double accuracy = static_cast<double>(correct) / static_cast<double>(dataset.samples.size());
    std::cout << "training_accuracy=" << accuracy << '\n';
    return accuracy >= 0.98 ? 0 : 1;
}
