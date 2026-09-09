#include "tinyvision/dataset.hpp"
#include "tinyvision/model.hpp"
#include "tinyvision/observation.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct RunResult {
    tinyvision::Observation train;
    tinyvision::Observation validation;
    std::vector<double> parameters;
    std::size_t learning_events{};
    std::size_t forgetting_events{};
};

bool same_dataset(const tinyvision::Dataset& a, const tinyvision::Dataset& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels ||
        a.samples.size() != b.samples.size()) return false;
    for (std::size_t i = 0; i < a.samples.size(); ++i) {
        if (a.samples[i].label != b.samples[i].label || a.samples[i].rgb != b.samples[i].rgb) return false;
    }
    return true;
}

RunResult run() {
    const auto train = tinyvision::make_synthetic_dataset(24, 11);
    const auto validation = tinyvision::make_generalization_dataset(
        tinyvision::SyntheticSplit::Validation, 24, 101);
    tinyvision::MLP model(train.width * train.height * train.channels, 24, 4, 7);
    auto previous = tinyvision::observe(model, validation);
    RunResult result;

    for (std::size_t epoch = 0; epoch < 180; ++epoch) {
        for (const auto& sample : train.samples) model.train_one(sample.rgb, sample.label, 0.025);
        const auto current = tinyvision::observe(model, validation);
        const auto transitions = tinyvision::compare_correctness(previous, current);
        result.learning_events += transitions.learning_events;
        result.forgetting_events += transitions.forgetting_events;
        previous = current;
    }

    result.train = tinyvision::observe(model, train);
    result.validation = tinyvision::observe(model, validation);
    result.parameters = model.parameters();
    return result;
}

} // namespace

int main() {
    const auto validation_a = tinyvision::make_generalization_dataset(
        tinyvision::SyntheticSplit::Validation, 24, 101);
    const auto validation_b = tinyvision::make_generalization_dataset(
        tinyvision::SyntheticSplit::Validation, 24, 101);
    const auto validation_other_seed = tinyvision::make_generalization_dataset(
        tinyvision::SyntheticSplit::Validation, 24, 102);

    if (!same_dataset(validation_a, validation_b)) {
        std::cerr << "validation dataset is not deterministic\n";
        return 1;
    }
    if (same_dataset(validation_a, validation_other_seed)) {
        std::cerr << "validation seed does not alter generated observations\n";
        return 1;
    }

    const auto first = run();
    const auto second = run();
    if (first.parameters != second.parameters ||
        first.train.average_loss != second.train.average_loss ||
        first.train.accuracy != second.train.accuracy ||
        first.validation.average_loss != second.validation.average_loss ||
        first.validation.accuracy != second.validation.accuracy ||
        first.validation.minimum_margin != second.validation.minimum_margin ||
        first.validation.mean_true_probability != second.validation.mean_true_probability ||
        first.validation.confusion != second.validation.confusion ||
        first.learning_events != second.learning_events ||
        first.forgetting_events != second.forgetting_events) {
        std::cerr << "TV-01A observation is not exactly reproducible\n";
        return 1;
    }

    if (first.train.accuracy < 0.98) {
        std::cerr << "TV-00 fitting gate regressed: " << first.train.accuracy << '\n';
        return 1;
    }
    if (!std::isfinite(first.validation.average_loss) ||
        !std::isfinite(first.validation.accuracy) ||
        !std::isfinite(first.validation.minimum_margin) ||
        !std::isfinite(first.validation.mean_true_probability)) {
        std::cerr << "non-finite validation observation\n";
        return 1;
    }

    std::cout << "TV-01A deterministic observatory PASS"
              << " train=" << first.train.accuracy
              << " validation=" << first.validation.accuracy
              << " learning_events=" << first.learning_events
              << " forgetting_events=" << first.forgetting_events << '\n';
    return 0;
}
