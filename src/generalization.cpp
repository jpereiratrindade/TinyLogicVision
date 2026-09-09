#include "tinyvision/dataset.hpp"
#include "tinyvision/model.hpp"
#include "tinyvision/observation.hpp"

#include <array>
#include <iomanip>
#include <iostream>

namespace {

constexpr std::size_t hidden = 24;
constexpr std::size_t classes = 4;
constexpr std::size_t epochs = 180;
constexpr double learning_rate = 0.025;
constexpr std::array<std::size_t, 8> milestones{1, 10, 30, 60, 90, 120, 150, 180};

bool is_milestone(std::size_t epoch) {
    for (const auto value : milestones) if (value == epoch) return true;
    return false;
}

void print_observation(std::size_t epoch,
                       const tinyvision::Observation& train,
                       const tinyvision::Observation& validation) {
    std::cout << "epoch=" << std::setw(3) << epoch
              << " train_loss=" << std::fixed << std::setprecision(6) << train.average_loss
              << " train_acc=" << std::setprecision(2) << (100.0 * train.accuracy) << "%"
              << " val_loss=" << std::setprecision(6) << validation.average_loss
              << " val_acc=" << std::setprecision(2) << (100.0 * validation.accuracy) << "%"
              << " val_true_p=" << std::setprecision(6) << validation.mean_true_probability
              << " val_min_margin=" << validation.minimum_margin
              << '\n';
}

void print_confusion(const tinyvision::ConfusionMatrix& matrix) {
    std::cout << "validation_confusion rows=actual cols=predicted\n";
    std::cout << "             horizontal vertical checkerboard diagonal\n";
    for (std::size_t actual = 0; actual < tinyvision::kObservedClasses; ++actual) {
        std::cout << std::setw(12) << tinyvision::class_name(actual);
        for (std::size_t predicted = 0; predicted < tinyvision::kObservedClasses; ++predicted) {
            std::cout << ' ' << std::setw(8) << matrix[actual][predicted];
        }
        std::cout << '\n';
    }
}

} // namespace

int main() {
    const auto train = tinyvision::make_synthetic_dataset(24, 11);
    const auto validation = tinyvision::make_generalization_dataset(
        tinyvision::SyntheticSplit::Validation, 24, 101);
    const std::size_t input = train.width * train.height * train.channels;
    tinyvision::MLP model(input, hidden, classes, 7);

    auto previous_validation = tinyvision::observe(model, validation);
    std::size_t learning_events = 0;
    std::size_t forgetting_events = 0;
    double best_validation_accuracy = previous_validation.accuracy;
    std::size_t best_validation_epoch = 0;
    double final_window_minimum_accuracy = 1.0;

    std::cout << "TinyLogicVision TV-01A Generalization Observatory\n";
    std::cout << "baseline_commit=0fd5e71490021c935b9ba4715cc3e0e13a2287e0\n";
    std::cout << "architecture=192->24(tanh)->4 learning_rate=0.025 epochs=180 seed=7\n";
    std::cout << "train_samples=" << train.samples.size()
              << " validation_samples=" << validation.samples.size() << '\n';
    std::cout << "sealed_test_status=NOT_EVALUATED\n\n";

    for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& sample : train.samples) {
            model.train_one(sample.rgb, sample.label, learning_rate);
        }

        const auto train_observation = tinyvision::observe(model, train);
        const auto validation_observation = tinyvision::observe(model, validation);
        const auto transitions = tinyvision::compare_correctness(previous_validation, validation_observation);
        learning_events += transitions.learning_events;
        forgetting_events += transitions.forgetting_events;

        if (validation_observation.accuracy > best_validation_accuracy) {
            best_validation_accuracy = validation_observation.accuracy;
            best_validation_epoch = epoch;
        }
        if (epoch > epochs - 30) {
            final_window_minimum_accuracy = std::min(
                final_window_minimum_accuracy, validation_observation.accuracy);
        }
        if (is_milestone(epoch)) {
            print_observation(epoch, train_observation, validation_observation);
        }
        previous_validation = validation_observation;
    }

    const auto final_train = tinyvision::observe(model, train);
    const auto final_validation = tinyvision::observe(model, validation);
    std::cout << "\nsummary\n";
    std::cout << "final_train_accuracy=" << std::fixed << std::setprecision(2)
              << (100.0 * final_train.accuracy) << "%\n";
    std::cout << "final_validation_accuracy=" << (100.0 * final_validation.accuracy) << "%\n";
    std::cout << "best_validation_accuracy=" << (100.0 * best_validation_accuracy)
              << "% best_epoch=" << best_validation_epoch << '\n';
    std::cout << "validation_learning_events=" << learning_events << '\n';
    std::cout << "validation_forgetting_events=" << forgetting_events << '\n';
    std::cout << "final_30_epoch_min_validation_accuracy="
              << (100.0 * final_window_minimum_accuracy) << "%\n";
    std::cout << "final_validation_true_probability=" << std::setprecision(6)
              << final_validation.mean_true_probability << '\n';
    std::cout << "final_validation_min_margin=" << final_validation.minimum_margin << '\n';
    print_confusion(final_validation.confusion);

    std::cout << "\nREA boundary: observe only; no hyperparameter or architecture intervention in TV-01A.\n";
    return 0;
}
