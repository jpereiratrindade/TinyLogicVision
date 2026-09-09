#include "tinyvision/sentinel.hpp"

#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace tinyvision {

Sentinel2Product Sentinel2Product::create_from_files(const std::filesystem::path& b2,
                                                     const std::filesystem::path& b3,
                                                     const std::filesystem::path& b4,
                                                     const std::filesystem::path& b8,
                                                     GeoMetadata meta) {
    if (!std::filesystem::exists(b2)) throw std::invalid_argument("Sentinel-2 B2 file not found: " + b2.string());
    if (!std::filesystem::exists(b3)) throw std::invalid_argument("Sentinel-2 B3 file not found: " + b3.string());
    if (!std::filesystem::exists(b4)) throw std::invalid_argument("Sentinel-2 B4 file not found: " + b4.string());
    if (!std::filesystem::exists(b8)) throw std::invalid_argument("Sentinel-2 B8 file not found: " + b8.string());

    Sentinel2Product prod;
    prod.b2_path = b2;
    prod.b3_path = b3;
    prod.b4_path = b4;
    prod.b8_path = b8;
    prod.schema = InputSchema::create_sentinel2_10m(8, 8);
    prod.metadata = std::move(meta);
    return prod;
}

Sentinel2Product Sentinel2Product::discover_from_safe(const std::filesystem::path& safe_directory) {
    if (!std::filesystem::is_directory(safe_directory)) {
        throw std::invalid_argument("SAFE path is not a directory: " + safe_directory.string());
    }

    std::vector<std::filesystem::path> b2_matches;
    std::vector<std::filesystem::path> b3_matches;
    std::vector<std::filesystem::path> b4_matches;
    std::vector<std::filesystem::path> b8_matches;

    // Search patterns for 10m bands (e.g. *B02_10m*, *B03_10m*, *B04_10m*, *B08_10m* or *B02*, *B03*, *B04*, *B08*)
    std::regex b2_re(R"((_B02_10m|_B02|_B2)\.(jp2|tif|tiff)$)", std::regex::icase);
    std::regex b3_re(R"((_B03_10m|_B03|_B3)\.(jp2|tif|tiff)$)", std::regex::icase);
    std::regex b4_re(R"((_B04_10m|_B04|_B4)\.(jp2|tif|tiff)$)", std::regex::icase);
    std::regex b8_re(R"((_B08_10m|_B08|_B8)\.(jp2|tif|tiff)$)", std::regex::icase);

    for (const auto& entry : std::filesystem::recursive_directory_iterator(safe_directory)) {
        if (entry.is_regular_file()) {
            const std::string fn = entry.path().filename().string();
            if (std::regex_search(fn, b2_re)) b2_matches.push_back(entry.path());
            else if (std::regex_search(fn, b3_re)) b3_matches.push_back(entry.path());
            else if (std::regex_search(fn, b4_re)) b4_matches.push_back(entry.path());
            else if (std::regex_search(fn, b8_re)) b8_matches.push_back(entry.path());
        }
    }

    if (b2_matches.empty() || b3_matches.empty() || b4_matches.empty() || b8_matches.empty()) {
        throw std::invalid_argument("missing required 10m Sentinel-2 bands in SAFE product");
    }
    if (b2_matches.size() > 1 || b3_matches.size() > 1 || b4_matches.size() > 1 || b8_matches.size() > 1) {
        throw std::invalid_argument("ambiguous multiple matches for 10m Sentinel-2 bands in SAFE product");
    }

    return create_from_files(b2_matches[0], b3_matches[0], b4_matches[0], b8_matches[0]);
}

RgbImage Sentinel2Product::generate_rgb_preview(const MultichannelTensor& tensor, double display_gain) {
    if (tensor.channels < 4) {
        throw std::invalid_argument("Sentinel-2 preview requires at least 4 channels (B2, B3, B4, B8)");
    }

    RgbImage preview;
    preview.width = tensor.width;
    preview.height = tensor.height;
    preview.pixels.resize(tensor.width * tensor.height * 3);

    // Channel mapping for True Color preview:
    // Red: Channel 2 (B4)
    // Green: Channel 1 (B3)
    // Blue: Channel 0 (B2)
    for (std::size_t y = 0; y < tensor.height; ++y) {
        for (std::size_t x = 0; x < tensor.width; ++x) {
            const double val_b2 = tensor.get_raw_value(x, y, 0); // Blue
            const double val_b3 = tensor.get_raw_value(x, y, 1); // Green
            const double val_b4 = tensor.get_raw_value(x, y, 2); // Red

            // Scale reflectance (0..10000 DN = 0..1.0 reflectance) with display gain to 0..255 byte
            const double norm_r = (val_b4 * 0.0001) * display_gain * 255.0;
            const double norm_g = (val_b3 * 0.0001) * display_gain * 255.0;
            const double norm_b = (val_b2 * 0.0001) * display_gain * 255.0;

            const std::size_t idx = (y * tensor.width + x) * 3;
            preview.pixels[idx + 0] = static_cast<std::uint8_t>(std::clamp(norm_r, 0.0, 255.0));
            preview.pixels[idx + 1] = static_cast<std::uint8_t>(std::clamp(norm_g, 0.0, 255.0));
            preview.pixels[idx + 2] = static_cast<std::uint8_t>(std::clamp(norm_b, 0.0, 255.0));
        }
    }

    return preview;
}

} // namespace tinyvision
