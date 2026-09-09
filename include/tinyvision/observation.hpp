#pragma once

#include "tinyvision/dataset.hpp"
#include "tinyvision/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyvision {

constexpr std::size_t kObservedClasses = 4;
using ConfusionMatrix = std::array<std::array<std::size_t, kObservedClasses>, kObservedClasses>;

struct Observation {
    double average_loss{};
    double accuracy{};
    double minimum_margin{};
    double mean_true_probability{};
    ConfusionMatrix confusion{};
    std::vector<std::uint8_t> correct;
};

struct TransitionCounts {
    std::size_t learning_events{};
    std::size_t forgetting_events{};
};

Observation observe(const MLP& model, const Dataset& dataset);
TransitionCounts compare_correctness(const Observation& before, const Observation& after);

} // namespace tinyvision
