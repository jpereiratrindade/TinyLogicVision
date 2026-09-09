#include "tinyvision/dataset.hpp"
#include "tinyvision/model.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
    constexpr std::size_t hidden = 24;
    constexpr std::size_t classes = 4;
    auto dataset = tinyvision::make_synthetic_dataset(24, 11);
    const std::size_t input = dataset.width * dataset.height * dataset.channels;
    tinyvision::MLP model(input, hidden, classes, 7);

    constexpr std::size_t epochs = 180;
    constexpr double learning_rate = 0.025;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (const auto& sample : dataset.samples) {
            total_loss += model.train_one(sample.rgb, sample.label, learning_rate);
        }
        if ((epoch + 1) % 30 == 0) {
            std::size_t correct = 0;
            for (const auto& sample : dataset.samples) {
                correct += model.predict_class(sample.rgb) == sample.label ? 1 : 0;
            }
            std::cout << "epoch=" << std::setw(3) << (epoch + 1)
                      << " loss=" << std::fixed << std::setprecision(6)
                      << total_loss / static_cast<double>(dataset.samples.size())
                      << " accuracy=" << (100.0 * static_cast<double>(correct) /
                                            static_cast<double>(dataset.samples.size()))
                      << "%\n";
        }
    }

    std::cout << "\nTinyLogicVision TV-00 complete\n";
    return 0;
}
