#include "tinyvision/application.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

void print_metrics(const char* name, const tinyvision::ApplicationMetrics& metrics) {
    std::cout << name << "_loss=" << std::fixed << std::setprecision(6)
              << metrics.average_loss << '\n'
              << name << "_accuracy=" << std::setprecision(2)
              << 100.0 * metrics.accuracy << "%\n"
              << name << "_mean_true_probability=" << std::setprecision(6)
              << metrics.mean_true_probability << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tinyvision_train DATASET_ROOT MODEL.tlv\n"
                  << "expected: DATASET_ROOT/train/CLASS/*.png and DATASET_ROOT/dev/CLASS/*.png\n";
        return 2;
    }

    try {
        const std::filesystem::path root(argv[1]);
        const auto train = tinyvision::load_application_split(root / "train");
        const auto development = tinyvision::load_application_split(root / "dev", train.class_names);
        const tinyvision::ApplicationTrainingConfig config;
        auto result = tinyvision::train_application(train, development, config);

        std::cout << "TinyLogicVision TV-APP-00 training\n"
                  << "architecture=192->" << result.model.network.hidden_size() << "->"
                  << result.model.network.output_size() << '\n'
                  << "classes=" << result.model.class_names.size()
                  << " train_samples=" << train.samples.size()
                  << " dev_samples=" << development.samples.size() << '\n'
                  << "model_seed=" << config.model_seed
                  << " order_seed=" << config.order_seed
                  << " learning_rate=" << config.learning_rate
                  << " epochs=" << config.epochs << '\n';
        for (const auto& milestone : result.milestones) {
            std::cout << "epoch=" << std::setw(3) << milestone.epoch
                      << " train_acc=" << std::fixed << std::setprecision(2)
                      << 100.0 * milestone.train.accuracy << "%"
                      << " dev_acc=" << 100.0 * milestone.development.accuracy << "%\n";
        }
        print_metrics("final_train", result.final_train);
        print_metrics("final_dev", result.final_development);
        std::cout << "parameter_count=" << result.model.network.parameters().size() << '\n';
        tinyvision::save_application_model(result.model, argv[2]);
        std::cout << "model_saved=" << argv[2] << '\n'
                  << "application_probe_status=NOT_EVALUATED\n"
                  << "synthetic_test_status=NOT_EVALUATED\n";
    } catch (const std::exception& error) {
        std::cerr << "tinyvision_train: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
