#include "tinyvision/application.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tinyvision_evaluate MODEL.tlv SPLIT_DIRECTORY\n"
                  << "expected: SPLIT_DIRECTORY/CLASS/*.png|*.jpg\n";
        return 2;
    }

    try {
        auto model = tinyvision::load_application_model(argv[1]);
        const auto split = tinyvision::load_application_split(
            std::filesystem::path(argv[2]), model.class_names);
        const auto metrics = tinyvision::evaluate_application(model.network, split);

        std::cout << "samples=" << split.samples.size() << '\n'
                  << "loss=" << std::fixed << std::setprecision(6) << metrics.average_loss << '\n'
                  << "accuracy=" << std::setprecision(2) << 100.0 * metrics.accuracy << "%\n"
                  << "mean_true_probability=" << std::setprecision(6)
                  << metrics.mean_true_probability << '\n'
                  << "confusion rows=actual cols=predicted\n";
        std::cout << std::setw(16) << "";
        for (const auto& name : model.class_names) std::cout << ' ' << std::setw(14) << name;
        std::cout << '\n';
        for (std::size_t actual = 0; actual < model.class_names.size(); ++actual) {
            std::cout << std::setw(16) << model.class_names[actual];
            for (const auto count : metrics.confusion[actual]) {
                std::cout << ' ' << std::setw(14) << count;
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "tinyvision_evaluate: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
