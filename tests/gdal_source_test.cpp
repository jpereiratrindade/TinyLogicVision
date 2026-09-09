#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/gdal_source.hpp"
#include "tinyvision/schema.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef TINYVISION_WITH_GDAL
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#endif

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

#ifdef TINYVISION_WITH_GDAL
void create_band(const std::filesystem::path& path,
                 std::uint16_t value,
                 double origin_x = 500000.0) {
    auto* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (driver == nullptr) fail("GTiff driver unavailable");
    auto* dataset = driver->Create(path.string().c_str(), 16, 16, 1, GDT_UInt16, nullptr);
    if (dataset == nullptr) fail("failed to create GDAL fixture");

    double transform[6]{origin_x, 10.0, 0.0, 7500000.0, 0.0, -10.0};
    if (dataset->SetGeoTransform(transform) != CE_None) fail("failed to set fixture geotransform");
    OGRSpatialReference reference;
    if (reference.importFromEPSG(32722) != OGRERR_NONE) fail("failed to create fixture CRS");
    if (dataset->SetSpatialRef(&reference) != CE_None) fail("failed to set fixture CRS");

    std::vector<std::uint16_t> pixels(16 * 16, value);
    if (dataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 16, 16,
                                            pixels.data(), 16, 16, GDT_UInt16,
                                            0, 0, nullptr) != CE_None) {
        fail("failed to write GDAL fixture");
    }
    GDALClose(dataset);
}
#endif

} // namespace

int main() {
    std::cout << "Running gdal_source_test...\n";
#ifndef TINYVISION_WITH_GDAL
    if (tinyvision::GdalMultibandSource::available()) fail("GDAL source incorrectly reports availability");
    std::cout << "PASS: gdal_source_test (GDAL unavailable; optional path skipped)\n";
    return 0;
#else
    GDALAllRegister();
    const auto root = std::filesystem::temp_directory_path() / "tinyvision_gdal_source_test";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    std::filesystem::create_directories(root);

    const auto b2 = root / "fixture_B02_10m.tif";
    const auto b3 = root / "fixture_B03_10m.tif";
    const auto b4 = root / "fixture_B04_10m.tif";
    const auto b8 = root / "fixture_B08_10m.tif";
    create_band(b2, 1000);
    create_band(b3, 2000);
    create_band(b4, 3000);
    create_band(b8, 4000);

    tinyvision::GdalMultibandSource source(b2, b3, b4, b8);
    if (!source.available() || source.width() != 16 || source.height() != 16 || source.channels() != 4) {
        fail("GDAL multiband source dimensions or availability mismatch");
    }
    if (!source.spatial_metadata().has_geo || source.schema().input_size() != 256) {
        fail("GDAL multiband source schema or georeferencing missing");
    }

    std::vector<double> patch(256);
    source.read_window_into(0, 0, patch);
    const double expected[]{0.1, 0.2, 0.3, 0.4};
    for (std::size_t channel = 0; channel < 4; ++channel) {
        if (std::abs(patch[channel] - expected[channel]) > 1e-12) {
            fail("native Sentinel normalization or B2/B3/B4/B8 order mismatch");
        }
    }

    tinyvision::MLP sentinel_network(256, 4, 2, 42);
    tinyvision::ApplicationModel sentinel_model(
        {"class_a", "class_b"}, tinyvision::InputSchema::create_sentinel2_10m(),
        std::move(sentinel_network));
    tinyvision::DenseMapConfig config;
    config.stride = 8;
    config.threads = 2;
    const auto result = tinyvision::classify_dense_source(sentinel_model, source, config);
    if (result.total_decisions != 4 || result.metadata.crs.empty()) {
        fail("native Sentinel dense classification result mismatch");
    }

    bool schema_rejected = false;
    try {
        tinyvision::MLP rgb_network(192, 4, 2, 42);
        tinyvision::ApplicationModel rgb_model({"class_a", "class_b"}, 8, 8, std::move(rgb_network));
        (void)tinyvision::classify_dense_source(rgb_model, source, config);
    } catch (const std::invalid_argument&) {
        schema_rejected = true;
    }
    if (!schema_rejected) fail("RGB model was accepted for a Sentinel multiband source");

    const auto misaligned_b8 = root / "misaligned_B08_10m.tif";
    create_band(misaligned_b8, 4000, 500010.0);
    bool alignment_rejected = false;
    try {
        tinyvision::GdalMultibandSource invalid_source(b2, b3, b4, misaligned_b8);
    } catch (const std::invalid_argument&) {
        alignment_rejected = true;
    }
    if (!alignment_rejected) fail("misaligned Sentinel band geotransform was accepted");

    std::filesystem::remove_all(root, cleanup_error);
    std::cout << "PASS: gdal_source_test (native B2/B3/B4/B8 source, schema gate, alignment gate)\n";
    return 0;
#endif
}
