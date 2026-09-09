#include "tinyvision/spatial_experiment.hpp"

#include "tinyvision/model.hpp"

#include <algorithm>
#include <array>

namespace tinyvision {
namespace {

constexpr std::size_t kHidden = 24;
constexpr std::size_t kClasses = 4;
constexpr std::size_t kEpochs = 180;
constexpr double kLearningRate = 0.025;
constexpr std::array<std::size_t, 8> kMilestones{1, 10, 30, 60, 90, 120, 150, 180};

bool is_milestone(std::size_t epoch) {
    return std::find(kMilestones.begin(), kMilestones.end(), epoch) != kMilestones.end();
}

bool same_observation(const Observation& first, const Observation& second) {
    return first.average_loss == second.average_loss &&
           first.accuracy == second.accuracy &&
           first.minimum_margin == second.minimum_margin &&
           first.mean_true_probability == second.mean_true_probability &&
           first.confusion == second.confusion &&
           first.correct == second.correct;
}

bool same_milestone(const MilestoneObservation& first, const MilestoneObservation& second) {
    return first.epoch == second.epoch &&
           same_observation(first.train, second.train) &&
           same_observation(first.validation, second.validation);
}

} // namespace

SpatialExperimentRun run_spatial_experiment(TrainingSpatialCoverage coverage) {
    const auto train = make_spatial_coverage_training_dataset(coverage, 24, 11);
    const auto validation = make_generalization_dataset(SyntheticSplit::Validation, 24, 101);
    MLP model(train.width * train.height * train.channels, kHidden, kClasses, 7);

    SpatialExperimentRun result;
    result.coverage = coverage;
    result.milestones.reserve(kMilestones.size());
    auto previous_validation = observe(model, validation);
    result.best_validation_accuracy = previous_validation.accuracy;
    result.final_30_epoch_min_validation_accuracy = 1.0;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        for (const auto& sample : train.samples) {
            model.train_one(sample.rgb, sample.label, kLearningRate);
        }

        const auto train_observation = observe(model, train);
        const auto validation_observation = observe(model, validation);
        const auto transitions = compare_correctness(previous_validation, validation_observation);
        result.learning_events += transitions.learning_events;
        result.forgetting_events += transitions.forgetting_events;

        if (validation_observation.accuracy > result.best_validation_accuracy) {
            result.best_validation_accuracy = validation_observation.accuracy;
            result.best_validation_epoch = epoch;
        }
        if (epoch > kEpochs - 30) {
            result.final_30_epoch_min_validation_accuracy = std::min(
                result.final_30_epoch_min_validation_accuracy, validation_observation.accuracy);
        }
        if (is_milestone(epoch)) {
            result.milestones.push_back({epoch, train_observation, validation_observation});
        }
        previous_validation = validation_observation;
    }

    result.final_train = observe(model, train);
    result.final_validation = observe(model, validation);
    result.final_parameters = model.parameters();
    return result;
}

bool exactly_reproduces(const SpatialExperimentRun& first,
                        const SpatialExperimentRun& second) {
    return first.coverage == second.coverage &&
           first.milestones.size() == second.milestones.size() &&
           std::equal(first.milestones.begin(), first.milestones.end(),
                      second.milestones.begin(), same_milestone) &&
           same_observation(first.final_train, second.final_train) &&
           same_observation(first.final_validation, second.final_validation) &&
           first.learning_events == second.learning_events &&
           first.forgetting_events == second.forgetting_events &&
           first.best_validation_accuracy == second.best_validation_accuracy &&
           first.best_validation_epoch == second.best_validation_epoch &&
           first.final_30_epoch_min_validation_accuracy ==
               second.final_30_epoch_min_validation_accuracy &&
           first.final_parameters == second.final_parameters;
}

SpatialExperimentClassification classify_spatial_experiment(
    const SpatialExperimentRun& control,
    const SpatialExperimentRun& intervention) {
    const double improvement = intervention.final_validation.accuracy -
                               control.final_validation.accuracy;
    if (intervention.final_validation.accuracy >= 0.90 &&
        intervention.final_train.accuracy >= 0.98) {
        return SpatialExperimentClassification::StrongSupport;
    }
    if (improvement >= 0.20 && intervention.final_validation.accuracy < 0.90) {
        return SpatialExperimentClassification::PartialSupport;
    }
    if (improvement < 0.05) {
        return SpatialExperimentClassification::Refutation;
    }
    return SpatialExperimentClassification::Inconclusive;
}

std::string_view classification_name(SpatialExperimentClassification classification) {
    switch (classification) {
        case SpatialExperimentClassification::StrongSupport: return "STRONG SUPPORT";
        case SpatialExperimentClassification::PartialSupport: return "PARTIAL SUPPORT";
        case SpatialExperimentClassification::Refutation: return "REFUTATION";
        case SpatialExperimentClassification::Inconclusive: return "INCONCLUSIVE";
    }
    return "INCONCLUSIVE";
}

} // namespace tinyvision
