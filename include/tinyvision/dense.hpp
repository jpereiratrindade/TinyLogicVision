#pragma once

#include "tinyvision/application.hpp"
#include "tinyvision/geo.hpp"
#include "tinyvision/image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
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
    double map_x{};
    double map_y{};
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

// Compact, cache-friendly decision record without strings for streaming & tiled engine
struct CompactDecision {
    std::uint32_t grid_x{};
    std::uint32_t grid_y{};
    std::uint32_t origin_x{};
    std::uint32_t origin_y{};
    std::uint16_t predicted_index{};
    std::uint16_t second_index{};
    float probability{};
    float second_probability{};
    float margin{};
    bool is_uncertain{};
};

#pragma pack(push, 1)
struct CompactDecisionRecord {
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t origin_x;
    std::uint32_t origin_y;
    std::uint16_t predicted_index;
    std::uint16_t second_index;
    float probability;
    float second_probability;
    float margin;
    std::uint8_t is_uncertain;
    std::uint8_t reserved[3];
};
#pragma pack(pop)

static_assert(sizeof(CompactDecisionRecord) == 36, "CompactDecisionRecord must be exactly 36 bytes");

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
    bool sentinel_nominal_10m{false}; // User-declared nominal 10m/px scale (unverified metadata)
    std::size_t threads{0};            // In-memory reference engine only; TV-STREAM-01 is sequential.
    std::size_t tile_width{256};
    std::size_t tile_height{256};
    std::size_t max_decisions{5'000'000}; // In-memory reference-engine cap; ignored by bounded streaming.
    bool write_decision_csv{false}; // Potentially huge for stride 1; primary outputs are rasters.
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
    std::vector<CompactDecision> compact_decisions;
    std::vector<std::size_t> class_counts;
    DenseMapConfig config;
    PaletteConfig palette;
    GeoMetadata metadata;
    std::string implementation_mode{"PARALLEL_IN_MEMORY"};
    std::size_t thread_count{1};
    std::size_t peak_tile_input_elements{0};
    std::size_t peak_tile_decisions{0};
    std::size_t peak_working_set_bytes{0};
};

// Extracts exactly 192 normalized RGB values in scanline order without interpolation
std::vector<double> extract_rgb_input_vector(const RgbImage& image,
                                             std::size_t origin_x,
                                             std::size_t origin_y);

// Zero-allocation version writing directly into span
void extract_rgb_input_vector_into(const RgbImage& image,
                                   std::size_t origin_x,
                                   std::size_t origin_y,
                                   std::span<double> output_192);

DenseMapResult classify_dense(const ApplicationModel& model,
                              const RgbImage& image,
                              const DenseMapConfig& config,
                              const GeoMetadata& meta = {});

DenseMapResult classify_dense_source(const ApplicationModel& model,
                                     const InputSource& source,
                                     const DenseMapConfig& config);

void export_dense_map(const DenseMapResult& result,
                      const ApplicationModel& model,
                      const RgbImage& source_image,
                      const std::filesystem::path& model_path,
                      const std::filesystem::path& source_path,
                      const std::filesystem::path& output_dir);

// TV-DENSE-STREAM-01: whole-scene inference whose working memory is bounded by
// tile size (plus the 7-pixel support halo), independent of total scene size.
// Artifacts are written incrementally into output_dir; returned vectors are empty.
DenseMapResult classify_and_export_dense_streaming(
    const ApplicationModel& model,
    const InputSource& source,
    const RgbImage& preview_image,
    const DenseMapConfig& config,
    const std::filesystem::path& model_path,
    const std::filesystem::path& source_path,
    const std::filesystem::path& output_dir);

} // namespace tinyvision
