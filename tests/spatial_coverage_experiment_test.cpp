#include "tinyvision/dataset.hpp"
#include "tinyvision/spatial_experiment.hpp"

#include <array>
#include <iostream>

namespace {

bool same_dataset(const tinyvision::Dataset& first, const tinyvision::Dataset& second) {
    if (first.width != second.width || first.height != second.height ||
        first.channels != second.channels || first.samples.size() != second.samples.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.samples.size(); ++index) {
        if (first.samples[index].label != second.samples[index].label ||
            first.samples[index].rgb != second.samples[index].rgb) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const auto legacy = tinyvision::make_synthetic_dataset(24, 11);
    const auto control_dataset = tinyvision::make_spatial_coverage_training_dataset(
        tinyvision::TrainingSpatialCoverage::ControlA, 24, 11);
    const auto intervention_dataset = tinyvision::make_spatial_coverage_training_dataset(
        tinyvision::TrainingSpatialCoverage::InterventionB, 24, 11);

    if (!same_dataset(legacy, control_dataset)) {
        std::cerr << "Control A does not exactly preserve the TV-00/TV-01A TRAIN dataset\n";
        return 1;
    }
    if (control_dataset.samples.size() != 96 || intervention_dataset.samples.size() != 96) {
        std::cerr << "paired TRAIN sample count changed\n";
        return 1;
    }

    std::array<std::size_t, 4> control_class_counts{};
    std::array<std::size_t, 4> intervention_class_counts{};
    bool spatial_pixels_changed = false;
    for (std::size_t index = 0; index < control_dataset.samples.size(); ++index) {
        const auto& control_sample = control_dataset.samples[index];
        const auto& intervention_sample = intervention_dataset.samples[index];
        if (control_sample.label != intervention_sample.label) {
            std::cerr << "deterministic shuffle or class presentation order changed\n";
            return 1;
        }
        ++control_class_counts[control_sample.label];
        ++intervention_class_counts[intervention_sample.label];
        spatial_pixels_changed = spatial_pixels_changed ||
                                 control_sample.rgb != intervention_sample.rgb;
    }
    if (control_class_counts != std::array<std::size_t, 4>{24, 24, 24, 24} ||
        intervention_class_counts != control_class_counts || !spatial_pixels_changed) {
        std::cerr << "paired TRAIN coverage contract failed\n";
        return 1;
    }

    const auto control_first = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::ControlA);
    const auto control_second = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::ControlA);
    if (!tinyvision::exactly_reproduces(control_first, control_second)) {
        std::cerr << "Control A complete trajectory is not exactly reproducible\n";
        return 1;
    }
    if (control_first.milestones.size() != 8 ||
        control_first.final_train.accuracy != 1.0 ||
        control_first.final_validation.accuracy != 45.0 / 96.0 ||
        control_first.learning_events != 34 ||
        control_first.forgetting_events != 14 ||
        control_first.final_parameters.size() != 4732) {
        std::cerr << "Control A does not reproduce TV-01A\n";
        return 1;
    }

    const auto intervention_first = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::InterventionB);
    const auto intervention_second = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::InterventionB);
    if (!tinyvision::exactly_reproduces(intervention_first, intervention_second)) {
        std::cerr << "Intervention B complete trajectory is not exactly reproducible\n";
        return 1;
    }
    if (intervention_first.milestones.size() != 8 ||
        intervention_first.final_parameters.size() != control_first.final_parameters.size()) {
        std::cerr << "Intervention B experiment shape changed\n";
        return 1;
    }

    std::cout << "TV-01B paired deterministic experiment PASS control_validation="
              << control_first.final_validation.accuracy
              << " intervention_validation=" << intervention_first.final_validation.accuracy
              << " classification=" << tinyvision::classification_name(
                     tinyvision::classify_spatial_experiment(control_first, intervention_first))
              << " sealed_test_status=NOT_EVALUATED\n";
    return 0;
}
