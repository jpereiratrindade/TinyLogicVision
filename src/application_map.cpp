#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/gdal_source.hpp"
#include "tinyvision/image.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tinyvision_map MODEL.tlv IMAGE.png|IMAGE.jpg OUTPUT_DIR [options]\n"
                  << "options:\n"
                  << "  --stride 1|2|4|8      Decision spacing (default: 1)\n"
                  << "  --confidence FLOAT    Minimum top-1 probability threshold [0.0..1.0] (default: 0.0)\n"
                  << "  --margin FLOAT        Minimum margin (top1 - top2) threshold [0.0..1.0] (default: 0.0)\n"
                  << "  --threads N           Number of inference threads (default: auto)\n"
                  << "  --max-decisions N     In-memory safety limit (default: 5000000; 0 = explicit unlimited)\n"
                  << "  --band-b2 PATH        Native Sentinel-2 B2 10m raster\n"
                  << "  --band-b3 PATH        Native Sentinel-2 B3 10m raster\n"
                  << "  --band-b4 PATH        Native Sentinel-2 B4 10m raster\n"
                  << "  --band-b8 PATH        Native Sentinel-2 B8 10m raster\n"
                  << "  --sentinel-10m        Declare nominal 10m/pixel resolution (operator-declared scale; context=80m, spacing=stride*10m)\n";
        return 2;
    }

    try {
        const std::filesystem::path model_path(argv[1]);
        const std::filesystem::path image_path(argv[2]);
        const std::filesystem::path output_dir(argv[3]);

        tinyvision::DenseMapConfig config;
        std::array<std::filesystem::path, 4> sentinel_bands;
        std::array<bool, 4> has_sentinel_band{};

        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--stride" && i + 1 < argc) {
                config.stride = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (arg == "--confidence" && i + 1 < argc) {
                config.confidence_threshold = std::stod(argv[++i]);
            } else if (arg == "--margin" && i + 1 < argc) {
                config.margin_threshold = std::stod(argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                config.threads = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (arg == "--max-decisions" && i + 1 < argc) {
                config.max_decisions = static_cast<std::size_t>(std::stoull(argv[++i]));
            } else if (arg == "--band-b2" && i + 1 < argc) {
                sentinel_bands[0] = argv[++i];
                has_sentinel_band[0] = true;
            } else if (arg == "--band-b3" && i + 1 < argc) {
                sentinel_bands[1] = argv[++i];
                has_sentinel_band[1] = true;
            } else if (arg == "--band-b4" && i + 1 < argc) {
                sentinel_bands[2] = argv[++i];
                has_sentinel_band[2] = true;
            } else if (arg == "--band-b8" && i + 1 < argc) {
                sentinel_bands[3] = argv[++i];
                has_sentinel_band[3] = true;
            } else if (arg == "--sentinel-10m") {
                config.sentinel_nominal_10m = true;
            } else {
                std::cerr << "tinyvision_map: unknown option " << arg << '\n';
                return 2;
            }
        }

        const auto model = tinyvision::load_application_model(model_path);
        const auto preview_image = tinyvision::load_rgb_image(image_path);
        const bool any_sentinel_band = std::any_of(has_sentinel_band.begin(), has_sentinel_band.end(), [](bool value) {
            return value;
        });
        const bool all_sentinel_bands = std::all_of(has_sentinel_band.begin(), has_sentinel_band.end(), [](bool value) {
            return value;
        });
        if (any_sentinel_band && !all_sentinel_bands) {
            throw std::invalid_argument("all four Sentinel-2 bands (--band-b2, --band-b3, --band-b4, --band-b8) are required");
        }

        tinyvision::DenseMapResult result;
        std::string source_mode;
        if (all_sentinel_bands) {
            tinyvision::GdalMultibandSource source(
                sentinel_bands[0], sentinel_bands[1], sentinel_bands[2], sentinel_bands[3]);
            result = tinyvision::classify_dense_source(model, source, config);
            source_mode = "SENTINEL2_MULTIBAND_NATIVE";
        } else {
            result = tinyvision::classify_dense(model, preview_image, config);
            source_mode = "RGB_IMAGE";
        }
        tinyvision::export_dense_map(result, model, preview_image, model_path, image_path, output_dir);

        std::cout << "TinyLogicVision dense spatial classification\n"
                  << "model=" << model_path.filename().string() << '\n'
                  << "classes=" << model.class_names.size() << '\n'
                  << "source_mode=" << source_mode << '\n'
                  << "image=" << image_path.filename().string()
                  << " preview_dimensions=" << preview_image.width << 'x' << preview_image.height
                  << " source_dimensions=" << result.source_width << 'x' << result.source_height << '\n'
                  << "stride=" << config.stride
                  << " grid=" << result.grid_width << 'x' << result.grid_height
                  << " decisions=" << result.total_decisions << '\n'
                  << "classified=" << result.classified_count
                  << " uncertain=" << result.uncertain_count << '\n'
                  << "confidence_threshold=" << std::fixed << std::setprecision(4) << config.confidence_threshold
                  << " margin_threshold=" << config.margin_threshold << '\n';

        if (config.sentinel_nominal_10m) {
            std::cout << "spatial_mode=SENTINEL_NOMINAL_10M_DECLARED\n"
                      << "nominal_context_m=80\n"
                      << "nominal_decision_spacing_m=" << (config.stride * 10) << '\n';
        } else {
            std::cout << "spatial_mode=DISPLAY_IMAGE_PIXEL_SPACE\n";
        }

        std::cout << "artifacts_saved=" << output_dir.string() << '\n'
                  << "status=READY\n";

    } catch (const std::exception& error) {
        std::cerr << "tinyvision_map: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
