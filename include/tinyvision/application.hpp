#pragma once

#include "tinyvision/model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tinyvision {

struct ApplicationSample {
    std::vector<double> input;
    std::size_t label{};
    std::filesystem::path source;
};

struct ApplicationSplit {
    std::vector<std::string> class_names;
    std::vector<ApplicationSample> samples;
};

struct ApplicationMetrics {
    double average_loss{};
    double accuracy{};
    double mean_true_probability{};
    std::vector<std::vector<std::size_t>> confusion;
};

struct ApplicationModel {
    std::vector<std::string> class_names;
    std::size_t image_width{};
    std::size_t image_height{};
    MLP network;

    ApplicationModel(std::vector<std::string> names,
                     std::size_t width,
                     std::size_t height,
                     MLP model);
};

struct ApplicationTrainingConfig {
    std::size_t hidden_size{24};
    std::uint32_t model_seed{7};
    std::uint32_t order_seed{11};
    double learning_rate{0.025};
    std::size_t epochs{180};
};

struct ApplicationEpoch {
    std::size_t epoch{};
    ApplicationMetrics train;
    ApplicationMetrics development;
};

struct ApplicationTrainingResult {
    ApplicationModel model;
    ApplicationMetrics final_train;
    ApplicationMetrics final_development;
    std::vector<ApplicationEpoch> milestones;

    ApplicationTrainingResult(ApplicationModel trained_model,
                              ApplicationMetrics train_metrics,
                              ApplicationMetrics development_metrics,
                              std::vector<ApplicationEpoch> epoch_metrics);
};

ApplicationSplit load_application_split(const std::filesystem::path& split_directory);
ApplicationSplit load_application_split(const std::filesystem::path& split_directory,
                                        const std::vector<std::string>& class_names);
ApplicationMetrics evaluate_application(const MLP& model, const ApplicationSplit& split);
ApplicationTrainingResult train_application(const ApplicationSplit& train,
                                             const ApplicationSplit& development,
                                             const ApplicationTrainingConfig& config = {});
void save_application_model(const ApplicationModel& model, const std::filesystem::path& path);
ApplicationModel load_application_model(const std::filesystem::path& path);

} // namespace tinyvision
