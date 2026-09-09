#pragma once

#include "tinyvision/dataset.hpp"
#include "tinyvision/observation.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace tinyvision {

struct MilestoneObservation {
    std::size_t epoch{};
    Observation train;
    Observation validation;
};

struct SpatialExperimentRun {
    TrainingSpatialCoverage coverage{};
    std::vector<MilestoneObservation> milestones;
    Observation final_train;
    Observation final_validation;
    std::size_t learning_events{};
    std::size_t forgetting_events{};
    double best_validation_accuracy{};
    std::size_t best_validation_epoch{};
    double final_30_epoch_min_validation_accuracy{};
    std::vector<double> final_parameters;
};

enum class SpatialExperimentClassification {
    StrongSupport,
    PartialSupport,
    Refutation,
    Inconclusive,
};

SpatialExperimentRun run_spatial_experiment(TrainingSpatialCoverage coverage);
bool exactly_reproduces(const SpatialExperimentRun& first,
                        const SpatialExperimentRun& second);
SpatialExperimentClassification classify_spatial_experiment(
    const SpatialExperimentRun& control,
    const SpatialExperimentRun& intervention);
std::string_view classification_name(SpatialExperimentClassification classification);

} // namespace tinyvision
