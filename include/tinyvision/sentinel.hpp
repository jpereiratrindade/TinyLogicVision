#pragma once

#include "tinyvision/geo.hpp"
#include "tinyvision/image.hpp"
#include "tinyvision/schema.hpp"
#include "tinyvision/tensor.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace tinyvision {

struct Sentinel2Product {
    std::filesystem::path b2_path;
    std::filesystem::path b3_path;
    std::filesystem::path b4_path;
    std::filesystem::path b8_path;
    InputSchema schema{InputSchema::create_sentinel2_10m(8, 8)};
    GeoMetadata metadata;

    static Sentinel2Product discover_from_safe(const std::filesystem::path& safe_directory);
    static Sentinel2Product create_from_files(const std::filesystem::path& b2,
                                             const std::filesystem::path& b3,
                                             const std::filesystem::path& b4,
                                             const std::filesystem::path& b8,
                                             GeoMetadata meta = {});

    // Creates an 8-bit True Color RGB preview (B4=Red, B3=Green, B2=Blue) for visualization
    static RgbImage generate_rgb_preview(const MultichannelTensor& tensor, double display_gain = 2.5);
};

} // namespace tinyvision
