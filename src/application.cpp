#include "tinyvision/application.hpp"

#include "tinyvision/image.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace tinyvision {
namespace {

constexpr std::size_t kImageWidth = 8;
constexpr std::size_t kImageHeight = 8;
constexpr std::array<std::size_t, 8> kMilestones{1, 10, 30, 60, 90, 120, 150, 180};
constexpr const char* kModelMagic = "TINYLOGICVISION_APP_MODEL";
constexpr std::size_t kModelVersion = 1;

bool supported_image(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

std::vector<std::string> discover_classes(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("split directory does not exist: " + directory.string());
    }
    std::vector<std::string> classes;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_directory()) classes.push_back(entry.path().filename().string());
    }
    std::sort(classes.begin(), classes.end());
    if (classes.size() < 2) {
        throw std::invalid_argument("application split requires at least two class directories");
    }
    return classes;
}

std::vector<std::filesystem::path> class_images(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && supported_image(entry.path())) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::invalid_argument("class contains no PNG/JPEG images: " + directory.string());
    }
    return paths;
}

bool is_milestone(std::size_t epoch, std::size_t final_epoch) {
    return epoch == final_epoch ||
           std::find(kMilestones.begin(), kMilestones.end(), epoch) != kMilestones.end();
}

} // namespace

ApplicationModel::ApplicationModel(std::vector<std::string> names,
                                   std::size_t width,
                                   std::size_t height,
                                   MLP model)
    : class_names(std::move(names)),
      image_width(width),
      image_height(height),
      network(std::move(model)) {}

ApplicationTrainingResult::ApplicationTrainingResult(
    ApplicationModel trained_model,
    ApplicationMetrics train_metrics,
    ApplicationMetrics development_metrics,
    std::vector<ApplicationEpoch> epoch_metrics)
    : model(std::move(trained_model)),
      final_train(std::move(train_metrics)),
      final_development(std::move(development_metrics)),
      milestones(std::move(epoch_metrics)) {}

ApplicationSplit load_application_split(const std::filesystem::path& split_directory) {
    return load_application_split(split_directory, discover_classes(split_directory));
}

ApplicationSplit load_application_split(const std::filesystem::path& split_directory,
                                        const std::vector<std::string>& class_names) {
    if (class_names.size() < 2) throw std::invalid_argument("at least two classes are required");
    if (discover_classes(split_directory) != class_names) {
        throw std::invalid_argument("split class directories do not match the model classes");
    }

    ApplicationSplit split;
    split.class_names = class_names;
    for (std::size_t label = 0; label < class_names.size(); ++label) {
        const auto directory = split_directory / class_names[label];
        for (const auto& path : class_images(directory)) {
            split.samples.push_back({
                resize_rgb_bilinear(load_rgb_image(path), kImageWidth, kImageHeight),
                label,
                path,
            });
        }
    }
    return split;
}

ApplicationMetrics evaluate_application(const MLP& model, const ApplicationSplit& split) {
    if (split.samples.empty()) throw std::invalid_argument("cannot evaluate an empty split");
    if (model.output_size() != split.class_names.size()) {
        throw std::invalid_argument("model outputs do not match split classes");
    }

    ApplicationMetrics metrics;
    metrics.confusion.assign(split.class_names.size(),
                             std::vector<std::size_t>(split.class_names.size()));
    std::size_t correct = 0;
    for (const auto& sample : split.samples) {
        if (sample.label >= split.class_names.size()) {
            throw std::invalid_argument("sample label is outside class range");
        }
        const auto probabilities = model.predict(sample.input);
        const auto predicted = static_cast<std::size_t>(std::distance(
            probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
        correct += predicted == sample.label ? 1 : 0;
        ++metrics.confusion[sample.label][predicted];
        metrics.average_loss += model.loss(sample.input, sample.label);
        metrics.mean_true_probability += probabilities[sample.label];
    }

    const auto count = static_cast<double>(split.samples.size());
    metrics.average_loss /= count;
    metrics.accuracy = static_cast<double>(correct) / count;
    metrics.mean_true_probability /= count;
    if (!std::isfinite(metrics.average_loss) || !std::isfinite(metrics.accuracy) ||
        !std::isfinite(metrics.mean_true_probability)) {
        throw std::runtime_error("non-finite application metrics");
    }
    return metrics;
}

ApplicationTrainingResult train_application(const ApplicationSplit& train,
                                             const ApplicationSplit& development,
                                             const ApplicationTrainingConfig& config) {
    if (train.samples.empty() || development.samples.empty()) {
        throw std::invalid_argument("TRAIN and DEV must both contain images");
    }
    if (train.class_names != development.class_names) {
        throw std::invalid_argument("TRAIN and DEV classes differ");
    }
    if (config.hidden_size == 0 || config.epochs == 0 ||
        !(config.learning_rate > 0.0) || !std::isfinite(config.learning_rate)) {
        throw std::invalid_argument("invalid application training configuration");
    }

    MLP model(kImageWidth * kImageHeight * 3,
              config.hidden_size,
              train.class_names.size(),
              config.model_seed);
    std::vector<std::size_t> order(train.samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 order_rng(config.order_seed);
    std::shuffle(order.begin(), order.end(), order_rng);

    std::vector<ApplicationEpoch> milestones;
    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        for (const auto index : order) {
            const auto& sample = train.samples[index];
            model.train_one(sample.input, sample.label, config.learning_rate);
        }
        if (is_milestone(epoch, config.epochs)) {
            milestones.push_back({epoch,
                                  evaluate_application(model, train),
                                  evaluate_application(model, development)});
        }
    }

    auto final_train = evaluate_application(model, train);
    auto final_development = evaluate_application(model, development);
    ApplicationModel artifact(train.class_names, kImageWidth, kImageHeight, std::move(model));
    return {std::move(artifact),
            std::move(final_train),
            std::move(final_development),
            std::move(milestones)};
}

void save_application_model(const ApplicationModel& model, const std::filesystem::path& path) {
    if (model.class_names.size() != model.network.output_size() ||
        model.network.input_size() != model.image_width * model.image_height * 3) {
        throw std::invalid_argument("inconsistent application model");
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create model file: " + path.string());

    output << kModelMagic << ' ' << kModelVersion << '\n'
           << model.image_width << ' ' << model.image_height << ' '
           << model.network.hidden_size() << ' ' << model.network.output_size() << '\n'
           << model.class_names.size() << '\n';
    for (const auto& name : model.class_names) output << std::quoted(name) << '\n';

    const auto parameters = model.network.parameters();
    output << parameters.size() << '\n'
           << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const double parameter : parameters) output << parameter << '\n';
    output.flush();
    if (!output) throw std::runtime_error("failed to write model file: " + path.string());
}

ApplicationModel load_application_model(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open model file: " + path.string());

    std::string magic;
    std::size_t version = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t hidden = 0;
    std::size_t output_size = 0;
    std::size_t class_count = 0;
    input >> magic >> version >> width >> height >> hidden >> output_size >> class_count;
    if (!input || magic != kModelMagic || version != kModelVersion ||
        width != kImageWidth || height != kImageHeight || hidden == 0 || hidden > 4096 ||
        output_size < 2 || output_size > 1024 || class_count != output_size) {
        throw std::runtime_error("invalid or unsupported model header: " + path.string());
    }

    std::vector<std::string> class_names(class_count);
    for (auto& name : class_names) input >> std::quoted(name);
    std::size_t parameter_count = 0;
    input >> parameter_count;
    const std::size_t expected_parameters =
        hidden * width * height * 3 + hidden + output_size * hidden + output_size;
    if (!input || parameter_count != expected_parameters ||
        std::any_of(class_names.begin(), class_names.end(),
                    [](const std::string& name) { return name.empty(); }) ||
        !std::is_sorted(class_names.begin(), class_names.end()) ||
        std::adjacent_find(class_names.begin(), class_names.end()) != class_names.end()) {
        throw std::runtime_error("inconsistent model metadata: " + path.string());
    }
    std::vector<double> parameters(parameter_count);
    for (auto& parameter : parameters) input >> parameter;
    if (!input) throw std::runtime_error("truncated model file: " + path.string());
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("unexpected data after model parameters");

    MLP model(width * height * 3, hidden, output_size, 0);
    model.set_parameters(parameters);
    return {std::move(class_names), width, height, std::move(model)};
}

} // namespace tinyvision
