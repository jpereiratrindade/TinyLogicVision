#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/image.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

tinyvision::RgbImage make_test_image(std::size_t width, std::size_t height) {
    tinyvision::RgbImage img;
    img.width = width;

    img.height = height;
    img.pixels.resize(width * height * 3);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t idx = (y * width + x) * 3;
            img.pixels[idx + 0] = static_cast<std::uint8_t>((x * 17 + y * 7) % 256);
            img.pixels[idx + 1] = static_cast<std::uint8_t>((x * 29 + y * 13) % 256);
            img.pixels[idx + 2] = static_cast<std::uint8_t>((x * 43 + y * 23) % 256);
        }
    }
    return img;
}

} // namespace

int main() {
    try {
        // 1. Setup model with 3 classes
        std::vector<std::string> class_names{"floresta", "campo", "solo"};
        tinyvision::MLP mlp(192, 24, 3, 7);
        tinyvision::ApplicationModel model(class_names, 8, 8, mlp);

        // 2. Test grid dimensions for stride 1, 2, 4, 8 on 24x20 image
        const auto image = make_test_image(24, 20);

        for (const auto stride : {1, 2, 4, 8}) {
            tinyvision::DenseMapConfig config;
            config.stride = stride;
            const auto result = tinyvision::classify_dense(model, image, config);

            const std::size_t expected_nx = (24 - 8) / stride + 1;
            const std::size_t expected_ny = (20 - 8) / stride + 1;
            const std::size_t expected_total = expected_nx * expected_ny;

            if (result.grid_width != expected_nx ||
                result.grid_height != expected_ny ||
                result.total_decisions != expected_total ||
                result.decisions.size() != expected_total) {
                std::cerr << "dense grid dimension check failed for stride " << stride << '\n';
                return 1;
            }

            // Check center coordinate formula and display anchor
            for (const auto& d : result.decisions) {
                if (std::abs(d.center_x - (static_cast<double>(d.origin_x) + 3.5)) > 1e-9 ||
                    std::abs(d.center_y - (static_cast<double>(d.origin_y) + 3.5)) > 1e-9 ||
                    d.display_x != d.origin_x + 4 ||
                    d.display_y != d.origin_y + 4) {
                    std::cerr << "geometric center or display coordinate check failed\n";
                    return 1;
                }
            }
        }

        // 3. Test exact byte-to-byte extraction and repeatability
        tinyvision::DenseMapConfig config_s1;
        config_s1.stride = 1;
        const auto run1 = tinyvision::classify_dense(model, image, config_s1);
        const auto run2 = tinyvision::classify_dense(model, image, config_s1);

        if (run1.total_decisions != run2.total_decisions) {
            std::cerr << "dense classification is not deterministic across runs\n";
            return 1;
        }

        for (std::size_t i = 0; i < run1.total_decisions; ++i) {
            if (run1.decisions[i].predicted_class != run2.decisions[i].predicted_class ||
                std::abs(run1.decisions[i].probability - run2.decisions[i].probability) > 1e-9 ||
                std::abs(run1.decisions[i].margin - run2.decisions[i].margin) > 1e-9) {
                std::cerr << "dense decision mismatch at index " << i << '\n';
                return 1;
            }
        }

        // 4. Test uncertainty thresholds
        tinyvision::DenseMapConfig config_thresh;
        config_thresh.stride = 1;
        config_thresh.confidence_threshold = 0.99999; // very high threshold to force UNCERTAIN
        config_thresh.margin_threshold = 0.99999;
        const auto run_uncertain = tinyvision::classify_dense(model, image, config_thresh);

        if (run_uncertain.uncertain_count != run_uncertain.total_decisions ||
            run_uncertain.classified_count != 0) {
            std::cerr << "uncertainty threshold logic failed\n";
            return 1;
        }

        // 5. Test export to disk
        const auto tmp_out = std::filesystem::temp_directory_path() / ("tinyvision_dense_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto dummy_model_path = tmp_out / "model.tlv";
        const auto dummy_source_path = tmp_out / "source.png";

        std::filesystem::create_directories(tmp_out);
        tinyvision::save_application_model(model, dummy_model_path);
        tinyvision::save_png_image(dummy_source_path, image);

        tinyvision::DenseMapConfig config_export;
        config_export.stride = 2;
        config_export.sentinel_native_10m = true;
        const auto result_export = tinyvision::classify_dense(model, image, config_export);
        tinyvision::export_dense_map(result_export, model, image, dummy_model_path, dummy_source_path, tmp_out);

        // Verify all 5 files exist
        if (!std::filesystem::is_regular_file(tmp_out / "classification.csv") ||
            !std::filesystem::is_regular_file(tmp_out / "run.json") ||
            !std::filesystem::is_regular_file(tmp_out / "class_map.png") ||
            !std::filesystem::is_regular_file(tmp_out / "confidence.png") ||
            !std::filesystem::is_regular_file(tmp_out / "overlay.png")) {
            std::cerr << "exported dense artifacts missing on disk\n";
            return 1;
        }

        // Verify CSV header
        std::ifstream csv(tmp_out / "classification.csv");
        std::string header;
        std::getline(csv, header);
        if (header != "origin_x,origin_y,center_x,center_y,display_x,display_y,predicted_class,probability,second_class,second_probability,margin,status") {
            std::cerr << "CSV header mismatch: " << header << '\n';
            return 1;
        }

        // Cleanup
        std::error_code ec;
        std::filesystem::remove_all(tmp_out, ec);

        std::cout << "TinyLogicVision dense spatial classification test PASS\n"
                  << "decisions_stride1=221 decisions_stride2=63\n"
                  << "center_geometry=EXACT repeatability=EXACT\n";

    } catch (const std::exception& error) {
        std::cerr << "dense_test error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
