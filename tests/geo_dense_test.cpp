#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/geo.hpp"
#include "tinyvision/image.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running geo_dense_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_geo_dense_test";
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
    meta.geotransform = {500000.0, 10.0, 0.0, 7500000.0, 0.0, -10.0};
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
    tinyvision::export_dense_map(res, model, img, "model.tlv", "image.png", out_dir);

    // 3. Verify GeoTIFF files were generated
    const auto class_tif = out_dir / "class_map.tif";
    const auto conf_tif = out_dir / "confidence.tif";
    const auto margin_tif = out_dir / "margin.tif";

    if (!std::filesystem::exists(class_tif)) fail("class_map.tif missing");
    if (!std::filesystem::exists(conf_tif)) fail("confidence.tif missing");
    if (!std::filesystem::exists(margin_tif)) fail("margin.tif missing");

    // Verify TIFF magic on class_map.tif
    {
        std::ifstream f(class_tif, std::ios::binary);
        char hdr[4];
        f.read(hdr, 4);
        if (hdr[0] != 'I' || hdr[1] != 'I' || hdr[2] != 42 || hdr[3] != 0) {
            fail("class_map.tif is not a valid Little-Endian TIFF file");
        }
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
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: geo_dense_test (GeoTIFF generation, coordinates in CSV, run.json geo metadata)\n";
    return 0;
}
