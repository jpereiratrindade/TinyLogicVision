#include "tinyvision/dataset.hpp"
#include "tinyvision/spatial_experiment.hpp"

#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

constexpr double kControlBaselineAccuracy = 45.0 / 96.0;

std::uint64_t parameter_fingerprint(const std::vector<double>& parameters) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const double parameter : parameters) {
        std::uint64_t bits = std::bit_cast<std::uint64_t>(parameter);
        for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= bits & 0xffU;
            hash *= 1099511628211ULL;
            bits >>= 8U;
        }
    }
    return hash;
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

void print_run(std::string_view name, const tinyvision::SpatialExperimentRun& run) {
    std::cout << '\n' << name << " milestones\n";
    for (const auto& milestone : run.milestones) {
        std::cout << "epoch=" << std::setw(3) << milestone.epoch
                  << " train_loss=" << std::fixed << std::setprecision(6)
                  << milestone.train.average_loss
                  << " train_acc=" << std::setprecision(2) << 100.0 * milestone.train.accuracy << "%"
                  << " val_loss=" << std::setprecision(6) << milestone.validation.average_loss
                  << " val_acc=" << std::setprecision(2) << 100.0 * milestone.validation.accuracy << "%"
                  << " val_true_p=" << std::setprecision(6)
                  << milestone.validation.mean_true_probability
                  << " val_min_margin=" << milestone.validation.minimum_margin << '\n';
    }

    std::cout << name << " summary\n"
              << "final_train_loss=" << std::fixed << std::setprecision(6)
              << run.final_train.average_loss << '\n'
              << "final_train_accuracy=" << std::setprecision(2)
              << 100.0 * run.final_train.accuracy << "%\n"
              << "final_validation_loss=" << std::setprecision(6)
              << run.final_validation.average_loss << '\n'
              << "final_validation_accuracy=" << std::setprecision(2)
              << 100.0 * run.final_validation.accuracy << "%\n"
              << "best_validation_accuracy=" << 100.0 * run.best_validation_accuracy
              << "% best_epoch=" << run.best_validation_epoch << '\n'
              << "validation_learning_events=" << run.learning_events << '\n'
              << "validation_forgetting_events=" << run.forgetting_events << '\n'
              << "final_30_epoch_min_validation_accuracy="
              << 100.0 * run.final_30_epoch_min_validation_accuracy << "%\n"
              << "final_validation_true_probability=" << std::setprecision(6)
              << run.final_validation.mean_true_probability << '\n'
              << "final_validation_min_margin=" << run.final_validation.minimum_margin << '\n'
              << "parameter_count=" << run.final_parameters.size() << '\n'
              << "parameter_fingerprint=0x" << std::hex
              << parameter_fingerprint(run.final_parameters) << std::dec << '\n';
    print_confusion(run.final_validation.confusion);
}

void print_determinism(std::string_view name, bool exact) {
    std::cout << name << " repetition=" << (exact ? "EXACT" : "DIVERGED")
              << " milestones=" << (exact ? "EXACT" : "DIVERGED")
              << " final_metrics=" << (exact ? "EXACT" : "DIVERGED")
              << " confusion=" << (exact ? "EXACT" : "DIVERGED")
              << " events=" << (exact ? "EXACT" : "DIVERGED")
              << " final_30=" << (exact ? "EXACT" : "DIVERGED")
              << " parameter_vector=" << (exact ? "EXACT" : "DIVERGED") << '\n';
}

} // namespace

int main() {
    std::cout << "TinyLogicVision TV-01B REA Spatial-Coverage Intervention\n"
              << "authority_commit=a84d125529efca017d4b7d6e47615726786e9845\n"
              << "architecture=192->24(tanh)->4 learning_rate=0.025 epochs=180 seed=7\n"
              << "train_samples=96 samples_per_class=24 training_seed=11\n"
              << "validation_samples=96 samples_per_class=24 validation_seed=101\n"
              << "sealed_test_status=NOT_EVALUATED\n";

    const auto control_first = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::ControlA);
    const auto control_second = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::ControlA);
    const bool control_exact = tinyvision::exactly_reproduces(control_first, control_second);
    print_run("Control A", control_first);
    print_determinism("Control A", control_exact);

    if (!control_exact) {
        std::cerr << "Control A repeated trajectory diverged; Intervention B remains unexecuted\n";
        return 1;
    }
    if (control_first.final_validation.accuracy != kControlBaselineAccuracy ||
        control_first.final_train.accuracy != 1.0) {
        std::cerr << "Control A did not reproduce TV-01A; Intervention B remains unexecuted\n";
        return 1;
    }
    std::cout << "Control A baseline_gate=PASS\n";

    const auto intervention_first = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::InterventionB);
    const auto intervention_second = tinyvision::run_spatial_experiment(
        tinyvision::TrainingSpatialCoverage::InterventionB);
    const bool intervention_exact = tinyvision::exactly_reproduces(
        intervention_first, intervention_second);
    print_run("Intervention B", intervention_first);
    print_determinism("Intervention B", intervention_exact);
    if (!intervention_exact) {
        std::cerr << "Intervention B repeated trajectory diverged; result is unclassified\n";
        return 1;
    }

    const double improvement = intervention_first.final_validation.accuracy -
                               control_first.final_validation.accuracy;
    const auto classification = tinyvision::classify_spatial_experiment(
        control_first, intervention_first);
    std::cout << "\nprimary_validation_improvement_pp=" << std::fixed << std::setprecision(2)
              << 100.0 * improvement << '\n'
              << "preregistered_classification="
              << tinyvision::classification_name(classification) << '\n'
              << "sealed_test_status=NOT_EVALUATED\n";
    return 0;
}
