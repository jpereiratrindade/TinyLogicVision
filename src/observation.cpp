#include "tinyvision/observation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tinyvision {

Observation observe(const MLP& model, const Dataset& dataset) {
    if (dataset.samples.empty()) throw std::invalid_argument("cannot observe empty dataset");
    if (model.output_size() != kObservedClasses) {
        throw std::invalid_argument("TV-01A observation expects four output classes");
    }

    Observation result;
    result.minimum_margin = std::numeric_limits<double>::infinity();
    result.correct.reserve(dataset.samples.size());

    std::size_t correct_count = 0;
    for (const auto& sample : dataset.samples) {
        if (sample.label >= kObservedClasses) throw std::invalid_argument("sample label out of range");
        const auto probabilities = model.predict(sample.rgb);
        const auto predicted = static_cast<std::size_t>(std::distance(
            probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
        const bool is_correct = predicted == sample.label;
        correct_count += is_correct ? 1 : 0;
        result.correct.push_back(is_correct ? 1U : 0U);
        ++result.confusion[sample.label][predicted];
        result.average_loss += model.loss(sample.rgb, sample.label);
        result.mean_true_probability += probabilities[sample.label];

        double strongest_other = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < probabilities.size(); ++index) {
            if (index != sample.label) strongest_other = std::max(strongest_other, probabilities[index]);
        }
        result.minimum_margin = std::min(
            result.minimum_margin,
            probabilities[sample.label] - strongest_other);
    }

    const auto count = static_cast<double>(dataset.samples.size());
    result.average_loss /= count;
    result.accuracy = static_cast<double>(correct_count) / count;
    result.mean_true_probability /= count;

    if (!std::isfinite(result.average_loss) || !std::isfinite(result.accuracy) ||
        !std::isfinite(result.minimum_margin) || !std::isfinite(result.mean_true_probability)) {
        throw std::runtime_error("non-finite observation");
    }
    return result;
}

TransitionCounts compare_correctness(const Observation& before, const Observation& after) {
    if (before.correct.size() != after.correct.size()) {
        throw std::invalid_argument("observation sizes differ");
    }
    TransitionCounts result;
    for (std::size_t i = 0; i < before.correct.size(); ++i) {
        if (before.correct[i] == 0U && after.correct[i] != 0U) ++result.learning_events;
        if (before.correct[i] != 0U && after.correct[i] == 0U) ++result.forgetting_events;
    }
    return result;
}

} // namespace tinyvision
