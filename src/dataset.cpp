#include "tinyvision/dataset.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace tinyvision {
namespace {

constexpr std::size_t kWidth = 8;
constexpr std::size_t kHeight = 8;
constexpr std::size_t kClasses = 4;

// Frozen TV-00 spatial rule. Do not alter without explicitly ending baseline compatibility.
bool training_foreground(std::size_t label,
                         std::size_t x,
                         std::size_t y,
                         std::size_t phase) {
    switch (label) {
        case 0: return ((y + phase) / 2) % 2 == 0;
        case 1: return ((x + phase) / 2) % 2 == 0;
        case 2: return ((x + phase) / 2 + (y + phase) / 2) % 2 == 0;
        case 3: return (x + y + phase) % 4 < 2;
        default: throw std::invalid_argument("unknown synthetic class");
    }
}

bool shifted_foreground(std::size_t label,
                        std::size_t x,
                        std::size_t y,
                        std::size_t shift_x,
                        std::size_t shift_y) {
    const auto sx = x + shift_x;
    const auto sy = y + shift_y;
    switch (label) {
        case 0: return (sy / 2) % 2 == 0;
        case 1: return (sx / 2) % 2 == 0;
        case 2: return ((sx / 2) + (sy / 2)) % 2 == 0;
        case 3: return (sx + sy) % 4 < 2;
        default: throw std::invalid_argument("unknown synthetic class");
    }
}

} // namespace

Dataset make_synthetic_dataset(std::size_t samples_per_class, std::uint32_t seed) {
    if (samples_per_class == 0) throw std::invalid_argument("samples_per_class must be positive");
    Dataset dataset;
    dataset.width = kWidth;
    dataset.height = kHeight;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-0.04, 0.04);
    std::uniform_real_distribution<double> color_jitter(-0.08, 0.08);

    for (std::size_t label = 0; label < kClasses; ++label) {
        for (std::size_t sample_index = 0; sample_index < samples_per_class; ++sample_index) {
            Sample sample;
            sample.label = label;
            sample.rgb.resize(kWidth * kHeight * 3);
            const std::size_t phase = sample_index % 2;
            const std::array<double, 3> tint = {
                std::clamp(0.72 + color_jitter(rng), 0.0, 1.0),
                std::clamp(0.66 + color_jitter(rng), 0.0, 1.0),
                std::clamp(0.60 + color_jitter(rng), 0.0, 1.0),
            };

            for (std::size_t y = 0; y < kHeight; ++y) {
                for (std::size_t x = 0; x < kWidth; ++x) {
                    const bool fg = training_foreground(label, x, y, phase);
                    const double level = fg ? 0.85 : 0.15;
                    const std::size_t base = (y * kWidth + x) * 3;
                    for (std::size_t c = 0; c < 3; ++c) {
                        sample.rgb[base + c] = std::clamp(level * tint[c] + noise(rng), 0.0, 1.0);
                    }
                }
            }
            dataset.samples.push_back(std::move(sample));
        }
    }

    std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);
    return dataset;
}

Dataset make_generalization_dataset(SyntheticSplit split,
                                    std::size_t samples_per_class,
                                    std::uint32_t seed) {
    if (split == SyntheticSplit::Training) return make_synthetic_dataset(samples_per_class, seed);
    if (samples_per_class == 0) throw std::invalid_argument("samples_per_class must be positive");

    Dataset dataset;
    dataset.width = kWidth;
    dataset.height = kHeight;

    std::mt19937 rng(seed);
    const double noise_radius = split == SyntheticSplit::Validation ? 0.07 : 0.10;
    const double jitter_radius = split == SyntheticSplit::Validation ? 0.12 : 0.16;
    const double foreground_level = split == SyntheticSplit::Validation ? 0.80 : 0.75;
    const double background_level = split == SyntheticSplit::Validation ? 0.20 : 0.25;
    std::uniform_real_distribution<double> noise(-noise_radius, noise_radius);
    std::uniform_real_distribution<double> color_jitter(-jitter_radius, jitter_radius);
    std::uniform_real_distribution<double> brightness(-0.04, 0.04);

    for (std::size_t label = 0; label < kClasses; ++label) {
        for (std::size_t sample_index = 0; sample_index < samples_per_class; ++sample_index) {
            Sample sample;
            sample.label = label;
            sample.rgb.resize(kWidth * kHeight * 3);

            std::size_t shift_x = (sample_index + 1) % 4;
            std::size_t shift_y = ((sample_index / 2) + 1) % 4;
            if (split == SyntheticSplit::Test) {
                shift_x = (sample_index * 3 + 2) % 4;
                shift_y = (sample_index * 2 + 3) % 4;
            }

            const std::array<double, 3> tint = {
                std::clamp(0.72 + color_jitter(rng), 0.0, 1.0),
                std::clamp(0.66 + color_jitter(rng), 0.0, 1.0),
                std::clamp(0.60 + color_jitter(rng), 0.0, 1.0),
            };
            const double offset = brightness(rng);

            for (std::size_t y = 0; y < kHeight; ++y) {
                for (std::size_t x = 0; x < kWidth; ++x) {
                    const bool fg = shifted_foreground(label, x, y, shift_x, shift_y);
                    const double level = fg ? foreground_level : background_level;
                    const std::size_t base = (y * kWidth + x) * 3;
                    for (std::size_t c = 0; c < 3; ++c) {
                        sample.rgb[base + c] = std::clamp(level * tint[c] + offset + noise(rng), 0.0, 1.0);
                    }
                }
            }
            dataset.samples.push_back(std::move(sample));
        }
    }

    std::shuffle(dataset.samples.begin(), dataset.samples.end(), rng);
    return dataset;
}

std::string_view class_name(std::size_t label) {
    static constexpr std::array<std::string_view, 4> names = {
        "horizontal", "vertical", "checkerboard", "diagonal"
    };
    return label < names.size() ? names[label] : "unknown";
}

std::string_view split_name(SyntheticSplit split) {
    switch (split) {
        case SyntheticSplit::Training: return "train";
        case SyntheticSplit::Validation: return "validation";
        case SyntheticSplit::Test: return "test";
    }
    return "unknown";
}

} // namespace tinyvision
