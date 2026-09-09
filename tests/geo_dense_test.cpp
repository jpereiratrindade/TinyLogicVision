#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/geo.hpp"
#include "tinyvision/image.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef TINYVISION_WITH_GDAL
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#endif

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running geo_dense_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_geo_dense_test";
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
    std::filesystem::create_directories(tmp_dir);

    // 1. Create model & synthetic image
    std::vector<std::string> classes{"floresta", "campo", "solo"};
    tinyvision::MLP mlp(192, 24, 3, 42);
    tinyvision::ApplicationModel model(classes, 8, 8, mlp);

    tinyvision::RgbImage img;
    img.width = 24;
    img.height = 20;
    img.pixels.resize(24 * 20 * 3, 100);

    // 2. Setup GeoMetadata
    tinyvision::GeoMetadata meta;
    meta.has_geo = true;
    meta.crs = "EPSG:32722";
    meta.geotransform = {500000.0, 10.0, 1.25, 7500000.0, -0.75, -10.0};
    meta.pixel_size_x = 10.0;
    meta.pixel_size_y = 10.0;
    meta.raster_width = 24;
    meta.raster_height = 20;

    tinyvision::DenseMapConfig cfg;
    cfg.stride = 2;

    const auto res = tinyvision::classify_dense(model, img, cfg, meta);
    if (!res.metadata.has_geo) {
        fail("Result does not carry geo metadata");
    }

    const auto out_dir = tmp_dir / "dense_geo_out";
#ifndef TINYVISION_WITH_GDAL
    bool unavailable_rejected = false;
    try {
        tinyvision::export_dense_map(res, model, img, "model.tlv", "image.png", out_dir);
    } catch (const std::runtime_error&) {
        unavailable_rejected = true;
    }
    if (!unavailable_rejected) fail("georeferenced output was silently written without GDAL");
    std::filesystem::remove_all(tmp_dir, ec);
    std::cout << "PASS: geo_dense_test (GeoTIFF export correctly requires GDAL)\n";
    return 0;
#else
    tinyvision::export_dense_map(res, model, img, "model.tlv", "image.png", out_dir);

    // 3. Verify GeoTIFF files were generated
    const auto class_tif = out_dir / "class_map.tif";
    const auto conf_tif = out_dir / "confidence.tif";
    const auto margin_tif = out_dir / "margin.tif";

    if (!std::filesystem::exists(class_tif)) fail("class_map.tif missing");
    if (!std::filesystem::exists(conf_tif)) fail("confidence.tif missing");
    if (!std::filesystem::exists(margin_tif)) fail("margin.tif missing");

    // Reopen every product through GDAL and verify CRS, full affine transform,
    // datatype, dimensions and nodata. This catches plain TIFFs mislabeled as GeoTIFF.
    GDALAllRegister();
    for (const auto& [path, expected_type, expected_nodata] :
         std::vector<std::tuple<std::filesystem::path, GDALDataType, double>>{
             {class_tif, GDT_Byte, 255.0},
             {conf_tif, GDT_Float32, -9999.0},
             {margin_tif, GDT_Float32, -9999.0}}) {
        auto* dataset = static_cast<GDALDataset*>(GDALOpen(path.string().c_str(), GA_ReadOnly));
        if (dataset == nullptr) fail("GDAL cannot reopen exported GeoTIFF");
        if (dataset->GetRasterXSize() != static_cast<int>(res.grid_width) ||
            dataset->GetRasterYSize() != static_cast<int>(res.grid_height) ||
            dataset->GetRasterBand(1)->GetRasterDataType() != expected_type) {
            GDALClose(dataset);
            fail("GeoTIFF dimensions or datatype mismatch");
        }

        const auto* reference = dataset->GetSpatialRef();
        const char* authority = reference == nullptr ? nullptr : reference->GetAuthorityCode(nullptr);
        if (authority == nullptr || std::string(authority) != "32722") {
            GDALClose(dataset);
            fail("GeoTIFF did not preserve EPSG:32722 CRS");
        }

        double actual_transform[6]{};
        if (dataset->GetGeoTransform(actual_transform) != CE_None) {
            GDALClose(dataset);
            fail("GeoTIFF geotransform missing");
        }
        const double center_offset = 3.5 - static_cast<double>(cfg.stride) / 2.0;
        const double expected_transform[6]{
            meta.geotransform[0] + center_offset * meta.geotransform[1] + center_offset * meta.geotransform[2],
            meta.geotransform[1] * static_cast<double>(cfg.stride),
            meta.geotransform[2] * static_cast<double>(cfg.stride),
            meta.geotransform[3] + center_offset * meta.geotransform[4] + center_offset * meta.geotransform[5],
            meta.geotransform[4] * static_cast<double>(cfg.stride),
            meta.geotransform[5] * static_cast<double>(cfg.stride),
        };
        for (std::size_t i = 0; i < 6; ++i) {
            if (std::abs(actual_transform[i] - expected_transform[i]) > 1e-9) {
                GDALClose(dataset);
                fail("GeoTIFF full affine transform mismatch");
            }
        }
        const auto source_center = meta.pixel_to_map(3.5, 3.5);
        const double output_center_x = actual_transform[0] + 0.5 * actual_transform[1] + 0.5 * actual_transform[2];
        const double output_center_y = actual_transform[3] + 0.5 * actual_transform[4] + 0.5 * actual_transform[5];
        if (std::abs(output_center_x - source_center.first) > 1e-9 ||
            std::abs(output_center_y - source_center.second) > 1e-9) {
            GDALClose(dataset);
            fail("first GeoTIFF pixel center does not match first decision point");
        }

        int has_nodata = 0;
        const double nodata = dataset->GetRasterBand(1)->GetNoDataValue(&has_nodata);
        if (!has_nodata || nodata != expected_nodata) {
            GDALClose(dataset);
            fail("GeoTIFF nodata value mismatch");
        }
        GDALClose(dataset);
    }

    // 4. Verify classification.csv contains map_x and map_y
    {
        const auto csv_path = out_dir / "classification.csv";
        std::ifstream f(csv_path);
        std::string header;
        std::getline(f, header);
        if (header.find("map_x,map_y") == std::string::npos) {
            fail("classification.csv missing map_x,map_y columns in georeferenced output");
        }
    }

    // 5. Verify run.json geospatial block
    {
        const auto json_path = out_dir / "run.json";
        std::ifstream f(json_path);
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (content.find("\"available\": true") == std::string::npos ||
            content.find("EPSG:32722") == std::string::npos ||
            content.find("\"254\": \"UNCERTAIN\"") == std::string::npos) {
            fail("run.json missing required geospatial attributes or class codes");
        }
    }

    // Cleanup
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: geo_dense_test (GDAL GeoTIFF CRS, affine transform, nodata, coordinates)\n";
    return 0;
#endif
}
