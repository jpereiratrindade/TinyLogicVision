#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/geo.hpp"
#include "tinyvision/image.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

#ifdef TINYVISION_WITH_GDAL
#include <gdal_priv.h>
#endif

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

class ProceduralSentinelSource final : public tinyvision::InputSource {
public:
    ProceduralSentinelSource(std::size_t width, std::size_t height)
        : width_(width), height_(height), schema_(tinyvision::InputSchema::create_sentinel2_10m()) {
        metadata_.has_geo = true;
        metadata_.crs = "EPSG:32722";
        metadata_.geotransform = {500000.0, 10.0, 0.25, 7500000.0, -0.1, -10.0};
        metadata_.pixel_size_x = 10.0;
        metadata_.pixel_size_y = 10.0;
        metadata_.raster_width = width;
        metadata_.raster_height = height;
    }

    std::size_t width() const override { return width_; }
    std::size_t height() const override { return height_; }
    std::size_t channels() const override { return 4; }
    const tinyvision::InputSchema& schema() const override { return schema_; }
    const tinyvision::GeoMetadata& spatial_metadata() const override { return metadata_; }

    void read_window_into(std::size_t x, std::size_t y, std::span<double> output) const override {
        read_region_into(x, y, 8, 8, output);
    }

    void read_region_into(std::size_t x0, std::size_t y0, std::size_t width,
                          std::size_t height, std::span<double> output) const override {
        if (x0 + width > width_ || y0 + height > height_ || output.size() != width * height * 4) {
            throw std::out_of_range("procedural test region invalid");
        }
        max_region_elements_ = std::max(max_region_elements_, output.size());
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                for (std::size_t channel = 0; channel < 4; ++channel) {
                    output[(y * width + x) * 4 + channel] =
                        static_cast<double>(((x0 + x) * 17 + (y0 + y) * 11 + channel * 23) % 1000) / 1000.0;
                }
            }
        }
    }

    std::size_t max_region_elements() const { return max_region_elements_; }

private:
    std::size_t width_;
    std::size_t height_;
    tinyvision::InputSchema schema_;
    tinyvision::GeoMetadata metadata_;
    mutable std::size_t max_region_elements_{0};
};

tinyvision::RgbImage make_preview(std::size_t width, std::size_t height) {
    tinyvision::RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(width * height * 3, 100);
    return image;
}

} // namespace

int main() {
    std::cout << "Running dense_streaming_test...\n";
#ifndef TINYVISION_WITH_GDAL
    std::cout << "PASS: dense_streaming_test (GDAL unavailable; streaming dependency explicit)\n";
    return 0;
#else
    const auto root = std::filesystem::temp_directory_path() / "tinyvision_dense_streaming_test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    std::filesystem::create_directories(root);

    tinyvision::MLP network(256, 12, 3, 77);
    tinyvision::ApplicationModel model({"forest", "field", "soil"},
        tinyvision::InputSchema::create_sentinel2_10m(), std::move(network));
    tinyvision::DenseMapConfig config;
    config.stride = 1;
    config.tile_width = 7;
    config.tile_height = 5;
    config.max_decisions = 1; // Must not constrain the streaming path.

    ProceduralSentinelSource small_source(40, 36);
    auto legacy_config = config;
    legacy_config.max_decisions = 0;
    legacy_config.threads = 1;
    const auto legacy = tinyvision::classify_dense_source(model, small_source, legacy_config);
    const auto streamed = tinyvision::classify_and_export_dense_streaming(
        model, small_source, make_preview(40, 36), config,
        root / "model.tlv", root / "sentinel_preview.png", root / "small");

    if (streamed.implementation_mode != "BOUNDED_TILE_STREAMING" ||
        !streamed.decisions.empty() || !streamed.compact_decisions.empty() ||
        streamed.total_decisions != legacy.total_decisions ||
        streamed.classified_count != legacy.classified_count ||
        streamed.uncertain_count != legacy.uncertain_count) {
        fail("streaming summary or bounded result-storage contract mismatch");
    }
    if (std::filesystem::exists(root / "small" / "classification.csv")) {
        fail("large per-decision CSV must remain opt-in on the streaming path");
    }

    std::ifstream binary(root / "small" / "decisions.bin", std::ios::binary);
    binary.seekg(64);
    for (std::size_t index = 0; index < legacy.compact_decisions.size(); ++index) {
        tinyvision::CompactDecisionRecord record{};
        binary.read(reinterpret_cast<char*>(&record), sizeof(record));
        const auto& expected = legacy.compact_decisions[index];
        if (!binary || record.grid_x != expected.grid_x || record.grid_y != expected.grid_y ||
            record.origin_x != expected.origin_x || record.origin_y != expected.origin_y ||
            record.predicted_index != expected.predicted_index || record.second_index != expected.second_index ||
            record.is_uncertain != (expected.is_uncertain ? 1 : 0) ||
            std::abs(record.probability - expected.probability) > 1e-7f ||
            std::abs(record.margin - expected.margin) > 1e-7f) {
            fail("streaming binary decisions differ from legacy reference");
        }
    }

    GDALAllRegister();
    auto* class_map = static_cast<GDALDataset*>(GDALOpen((root / "small" / "class_map.tif").string().c_str(), GA_ReadOnly));
    if (class_map == nullptr || class_map->GetRasterXSize() != static_cast<int>(legacy.grid_width) ||
        class_map->GetRasterYSize() != static_cast<int>(legacy.grid_height)) {
        if (class_map != nullptr) GDALClose(class_map);
        fail("streaming primary class raster dimensions mismatch");
    }
    std::vector<std::uint8_t> classes(legacy.total_decisions);
    if (class_map->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, static_cast<int>(legacy.grid_width),
            static_cast<int>(legacy.grid_height), classes.data(), static_cast<int>(legacy.grid_width),
            static_cast<int>(legacy.grid_height), GDT_Byte, 0, 0, nullptr) != CE_None) {
        GDALClose(class_map);
        fail("cannot read streaming class raster");
    }
    GDALClose(class_map);
    for (std::size_t index = 0; index < classes.size(); ++index) {
        const auto expected = legacy.compact_decisions[index].is_uncertain ? 254 :
            static_cast<std::uint8_t>(legacy.compact_decisions[index].predicted_index);
        if (classes[index] != expected) fail("streaming class raster differs from legacy reference");
    }

    ProceduralSentinelSource tall_source(40, 360);
    const auto tall = tinyvision::classify_and_export_dense_streaming(
        model, tall_source, make_preview(40, 360), config,
        root / "model.tlv", root / "sentinel_preview.png", root / "tall");
    if (tall.total_decisions <= streamed.total_decisions * 5 ||
        tall.peak_working_set_bytes != streamed.peak_working_set_bytes ||
        tall_source.max_region_elements() != small_source.max_region_elements()) {
        fail("working memory changed with total Sentinel scene height");
    }

    std::filesystem::remove_all(root, ignored);
    std::cout << "PASS: dense_streaming_test (Sentinel tile+halo equivalence and bounded memory)\n";
    return 0;
#endif
}
