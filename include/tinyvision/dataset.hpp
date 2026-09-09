#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace tinyvision {

struct Sample {
    std::vector<double> rgb;
    std::size_t label{};
};

struct Dataset {
    std::size_t width{};
    std::size_t height{};
    std::size_t channels{3};
    std::vector<Sample> samples;
};

Dataset make_synthetic_dataset(std::size_t samples_per_class = 24, std::uint32_t seed = 11);
std::string_view class_name(std::size_t label);

} // namespace tinyvision
