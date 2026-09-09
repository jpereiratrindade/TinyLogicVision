#pragma once

#include "tinyvision/application.hpp"
#include "tinyvision/image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tinyvision {

// Formal Spatial Decision Semantics:
// SUPPORT: 8x8 pixels [origin_x, origin_x + 7] x [origin_y, origin_y + 7]
// DECISION POINT: exact continuous geometric center (center_x = origin_x + 3.5, center_y = origin_y + 3.5)
// GRID CELL: discrete coordinates in decision raster (grid_x = origin_x / stride, grid_y = origin_y / stride)
// DISPLAY ANCHOR: discrete integer visualization anchor (display_x = origin_x + 4, display_y = origin_y + 4)
// INVARIANT: SUPPORT != DECISION POINT != DISPLAY CELL
struct DenseDecision {
    std::size_t grid_x{};
    std::size_t grid_y{};
    std::size_t origin_x{};
    std::size_t origin_y{};
    double center_x{};
    double center_y{};
    std::size_t display_x{};
    std::size_t display_y{};
    std::size_t predicted_index{};
    std::string predicted_class;
    double probability{}; // Top-1 uncalibrated softmax probability
    std::size_t second_index{};
    std::string second_class;
    double second_probability{}; // Top-2 softmax probability
    double margin{};             // top1_prob - top2_prob
    std::string status;          // "CLASSIFIED" or "UNCERTAIN" (by configured threshold)
    bool is_uncertain{};
};

struct ClassColor {
    std::string name;
    std::size_t index{};
    std::string hex_color;
    std::array<std::uint8_t, 3> rgb{};
};

struct PaletteConfig {
    std::vector<ClassColor> classes;
    ClassColor uncertain{"UNCERTAIN", 9999, "#808080", {128, 128, 128}};
};

PaletteConfig get_canonical_palette(const std::vector<std::string>& class_names);

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
    PaletteConfig palette;
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
