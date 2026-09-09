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

bool foreground(std::size_t label, std::size_t x, std::size_t y, std::size_t phase) {
    switch (label) {
        case 0: return ((y + phase) / 2) % 2 == 0;                  // horizontal bands
        case 1: return ((x + phase) / 2) % 2 == 0;                  // vertical bands
        case 2: return ((x + phase) / 2 + (y + phase) / 2) % 2 == 0; // checkerboard
        case 3: return (x + y + phase) % 4 < 2;                     // diagonals
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
                    const bool fg = foreground(label, x, y, phase);
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

std::string_view class_name(std::size_t label) {
    static constexpr std::array<std::string_view, 4> names = {
        "horizontal", "vertical", "checkerboard", "diagonal"
    };
    return label < names.size() ? names[label] : "unknown";
}

} // namespace tinyvision
