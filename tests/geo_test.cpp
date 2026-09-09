#include "tinyvision/geo.hpp"
#include "tinyvision/schema.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running geo_test...\n";

    // 1. Test pixel to map coordinate transformation
    tinyvision::GeoMetadata meta;
    meta.has_geo = true;
    meta.crs = "EPSG:32722";
    meta.raster_width = 1000;
    meta.raster_height = 1000;
    meta.geotransform = {500000.0, 10.0, 0.0, 7500000.0, 0.0, -10.0};
    meta.pixel_size_x = 10.0;
    meta.pixel_size_y = 10.0;

    const auto [map_x, map_y] = meta.pixel_to_map(10.5, 20.5);
    const double expected_mx = 500000.0 + 10.5 * 10.0; // 500105.0
    const double expected_my = 7500000.0 - 20.5 * 10.0; // 7499795.0

    if (std::abs(map_x - expected_mx) > 1e-9 || std::abs(map_y - expected_my) > 1e-9) {
        fail("pixel_to_map coordinate calculation mismatch");
    }

    // 2. Test InputSource polymorphism
    tinyvision::RgbImage img;
    img.width = 16;
    img.height = 16;
    img.pixels.resize(16 * 16 * 3, 128);
    tinyvision::RgbImageSource rgb_source(img, meta);

    if (rgb_source.width() != 16 || rgb_source.channels() != 3) {
        fail("rgb_source dimension mismatch");
    }
    std::vector<double> buf(192);
    rgb_source.read_window_into(0, 0, buf);
    if (std::abs(buf[0] - (128.0 / 255.0)) > 1e-9) {
        fail("rgb_source read_window_into normalization mismatch");
    }

    // 3. Test multiband alignment checks
    auto meta_b2 = meta;
    auto meta_b3 = meta;
    auto meta_b4 = meta;
    auto meta_b8 = meta;

    // Matching 4 bands
    std::vector<tinyvision::GeoMetadata> bands_ok = {meta_b2, meta_b3, meta_b4, meta_b8};
    bool threw = false;
    try {
        tinyvision::validate_multiband_alignment(bands_ok);
    } catch (...) {
        threw = true;
    }
    if (threw) {
        fail("Matching bands failed alignment validation");
    }

    // Dimension mismatch
    auto meta_misaligned_dim = meta;
    meta_misaligned_dim.raster_width = 500;
    std::vector<tinyvision::GeoMetadata> bands_dim_err = {meta_b2, meta_misaligned_dim};
    threw = false;
    try {
        tinyvision::validate_multiband_alignment(bands_dim_err);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        fail("Dimension mismatch was not rejected");
    }

    // CRS mismatch
    auto meta_misaligned_crs = meta;
    meta_misaligned_crs.crs = "EPSG:4326";
    std::vector<tinyvision::GeoMetadata> bands_crs_err = {meta_b2, meta_misaligned_crs};
    threw = false;
    try {
        tinyvision::validate_multiband_alignment(bands_crs_err);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        fail("CRS mismatch was not rejected");
    }

    // Geotransform mismatch
    auto meta_misaligned_gt = meta;
    meta_misaligned_gt.geotransform[1] = 20.0; // 20m resolution instead of 10m
    std::vector<tinyvision::GeoMetadata> bands_gt_err = {meta_b2, meta_misaligned_gt};
    threw = false;
    try {
        tinyvision::validate_multiband_alignment(bands_gt_err);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        fail("Geotransform resolution mismatch was not rejected");
    }

    std::cout << "PASS: geo_test (CRS/geotransform mapping + multiband strict alignment gate)\n";
    return 0;
}
