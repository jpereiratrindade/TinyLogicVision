#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/image.hpp"

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
                  << "  --sentinel-10m        Mark source as native Sentinel-2 10m pixels (80m context, stride*10m spacing)\n";
        return 2;
    }

    try {
        const std::filesystem::path model_path(argv[1]);
        const std::filesystem::path image_path(argv[2]);
        const std::filesystem::path output_dir(argv[3]);

        tinyvision::DenseMapConfig config;

        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--stride" && i + 1 < argc) {
                config.stride = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (arg == "--confidence" && i + 1 < argc) {
                config.confidence_threshold = std::stod(argv[++i]);
            } else if (arg == "--margin" && i + 1 < argc) {
                config.margin_threshold = std::stod(argv[++i]);
            } else if (arg == "--sentinel-10m") {
                config.sentinel_native_10m = true;
            } else {
                std::cerr << "tinyvision_map: unknown option " << arg << '\n';
                return 2;
            }
        }

        const auto model = tinyvision::load_application_model(model_path);
        const auto image = tinyvision::load_rgb_image(image_path);

        const auto result = tinyvision::classify_dense(model, image, config);
        tinyvision::export_dense_map(result, model, image, model_path, image_path, output_dir);

        std::cout << "TinyLogicVision dense spatial classification\n"
                  << "model=" << model_path.filename().string() << '\n'
                  << "classes=" << model.class_names.size() << '\n'
                  << "image=" << image_path.filename().string()
                  << " dimensions=" << image.width << 'x' << image.height << '\n'
                  << "stride=" << config.stride
                  << " grid=" << result.grid_width << 'x' << result.grid_height
                  << " decisions=" << result.total_decisions << '\n'
                  << "classified=" << result.classified_count
                  << " uncertain=" << result.uncertain_count << '\n'
                  << "confidence_threshold=" << std::fixed << std::setprecision(4) << config.confidence_threshold
                  << " margin_threshold=" << config.margin_threshold << '\n';

        if (config.sentinel_native_10m) {
            std::cout << "spatial_mode=SENTINEL_NATIVE_10M\n"
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
