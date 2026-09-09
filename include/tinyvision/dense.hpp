#pragma once

#include "tinyvision/application.hpp"
#include "tinyvision/image.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace tinyvision {

struct DenseDecision {
    std::size_t origin_x{};
    std::size_t origin_y{};
    double center_x{};
    double center_y{};
    std::size_t display_x{};
    std::size_t display_y{};
    std::size_t grid_x{};
    std::size_t grid_y{};
    std::size_t predicted_index{};
    std::string predicted_class;
    double probability{};
    std::size_t second_index{};
    std::string second_class;
    double second_probability{};
    double margin{};
    std::string status; // "CLASSIFIED" or "UNCERTAIN"
    bool is_uncertain{};
};

struct DenseMapConfig {
    std::size_t stride{1}; // 1, 2, 4, 8
    double confidence_threshold{0.0};
    double margin_threshold{0.0};
    bool sentinel_native_10m{false};
};

struct DenseMapResult {
    std::size_t source_width{};
    std::size_t source_height{};
    std::size_t grid_width{};
    std::size_t grid_height{};
    std::size_t total_decisions{};
    std::size_t classified_count{};
    std::size_t uncertain_count{};
    std::vector<DenseDecision> decisions;
    DenseMapConfig config;
};

DenseMapResult classify_dense(const ApplicationModel& model,
                              const RgbImage& image,
                              const DenseMapConfig& config);

void export_dense_map(const DenseMapResult& result,
                      const ApplicationModel& model,
                      const RgbImage& source_image,
                      const std::filesystem::path& model_path,
                      const std::filesystem::path& source_path,
                      const std::filesystem::path& output_dir);

} // namespace tinyvision
