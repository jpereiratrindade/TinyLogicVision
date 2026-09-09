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
        // 1. Direct 8x8 Input Vector Extraction Witness (EXACT_RGB_INPUT_VECTOR):
        // Verify all 192 normalized double values match the exact RGB scanline order without interpolation
        {
            const std::size_t ref_w = 16;
            const std::size_t ref_h = 16;
            tinyvision::RgbImage ref_img;
            ref_img.width = ref_w;
            ref_img.height = ref_h;
            ref_img.pixels.resize(ref_w * ref_h * 3);

            for (std::size_t y = 0; y < ref_h; ++y) {
                for (std::size_t x = 0; x < ref_w; ++x) {
                    const std::size_t idx = (y * ref_w + x) * 3;
                    ref_img.pixels[idx + 0] = static_cast<std::uint8_t>((x * 11 + y * 3) % 256);
                    ref_img.pixels[idx + 1] = static_cast<std::uint8_t>((x * 13 + y * 5) % 256);
                    ref_img.pixels[idx + 2] = static_cast<std::uint8_t>((x * 17 + y * 7) % 256);
                }
            }

            // Test extraction at multiple offsets
            for (const auto [ox, oy] : std::vector<std::pair<std::size_t, std::size_t>>{{0, 0}, {2, 3}, {8, 8}}) {
                const auto window = tinyvision::extract_rgb_input_vector(ref_img, ox, oy);
                if (window.size() != 192) {
                    std::cerr << "extracted window size mismatch: got " << window.size() << " expected 192\n";
                    return 1;
                }

                // Verify every single one of the 192 values
                for (std::size_t wy = 0; wy < 8; ++wy) {
                    for (std::size_t wx = 0; wx < 8; ++wx) {
                        const std::size_t src_idx = ((oy + wy) * ref_w + (ox + wx)) * 3;
                        const std::size_t dst_idx = (wy * 8 + wx) * 3;
                        const double expected_r = static_cast<double>(ref_img.pixels[src_idx + 0]) / 255.0;
                        const double expected_g = static_cast<double>(ref_img.pixels[src_idx + 1]) / 255.0;
                        const double expected_b = static_cast<double>(ref_img.pixels[src_idx + 2]) / 255.0;

                        if (std::abs(window[dst_idx + 0] - expected_r) > 1e-15 ||
                            std::abs(window[dst_idx + 1] - expected_g) > 1e-15 ||
                            std::abs(window[dst_idx + 2] - expected_b) > 1e-15) {
                            std::cerr << "exact RGB input vector mismatch at (" << wx << ", " << wy << ")\n";
                            return 1;
                        }
                    }
                }
            }

            // Verify out-of-range bounds check
            bool threw = false;
            try {
                tinyvision::extract_rgb_input_vector(ref_img, 9, 0); // 9 + 8 = 17 > 16
            } catch (const std::out_of_range&) {
                threw = true;
            }
            if (!threw) {
                std::cerr << "out-of-bounds patch extraction did not throw out_of_range\n";
                return 1;
            }
        }

        // 2. Canonical Palette Authority Verification
        {
            std::vector<std::string> classes{"floresta", "campo", "solo"};
            const auto pal = tinyvision::get_canonical_palette(classes);
            if (pal.classes.size() != 3) {
                std::cerr << "canonical palette class count mismatch\n";
                return 1;
            }
            if (pal.classes[0].name != "floresta" || pal.classes[0].hex_color != "#22c55e" ||
                pal.classes[0].rgb[0] != 34 || pal.classes[0].rgb[1] != 197 || pal.classes[0].rgb[2] != 94) {
                std::cerr << "canonical palette class 0 color mismatch\n";
                return 1;
            }
            if (pal.uncertain.name != "UNCERTAIN" || pal.uncertain.hex_color != "#808080" ||
                pal.uncertain.rgb[0] != 128 || pal.uncertain.rgb[1] != 128 || pal.uncertain.rgb[2] != 128) {
                std::cerr << "canonical palette uncertain color mismatch\n";
                return 1;
            }
        }

        // 3. Setup model with 3 classes
        std::vector<std::string> class_names{"floresta", "campo", "solo"};
        tinyvision::MLP mlp(192, 24, 3, 7);
        tinyvision::ApplicationModel model(class_names, 8, 8, mlp);

        // 4. Test grid dimensions & spatial semantics for stride 1, 2, 4, 8 on 24x20 image
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

            // Check center coordinate formula, grid coordinates, and display anchor
            for (const auto& d : result.decisions) {
                if (d.grid_x != d.origin_x / stride ||
                    d.grid_y != d.origin_y / stride ||
                    std::abs(d.center_x - (static_cast<double>(d.origin_x) + 3.5)) > 1e-9 ||
                    std::abs(d.center_y - (static_cast<double>(d.origin_y) + 3.5)) > 1e-9 ||
                    d.display_x != d.origin_x + 4 ||
                    d.display_y != d.origin_y + 4) {
                    std::cerr << "geometric center, grid or display coordinate check failed\n";
                    return 1;
                }
            }
        }

        // 5. Test repeatability
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

        // 6. Test uncertainty thresholds
        tinyvision::DenseMapConfig config_thresh;
        config_thresh.stride = 1;
        config_thresh.confidence_threshold = 0.99999; // force UNCERTAIN
        config_thresh.margin_threshold = 0.99999;
        const auto run_uncertain = tinyvision::classify_dense(model, image, config_thresh);

        if (run_uncertain.uncertain_count != run_uncertain.total_decisions ||
            run_uncertain.classified_count != 0) {
            std::cerr << "uncertainty threshold logic failed\n";
            return 1;
        }

        // 7. Test export to disk & artifact contracts & overlay cell centering
        const auto tmp_out = std::filesystem::temp_directory_path() / ("tinyvision_dense_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto dummy_model_path = tmp_out / "model.tlv";
        const auto dummy_source_path = tmp_out / "source.png";

        std::filesystem::create_directories(tmp_out);
        tinyvision::save_application_model(model, dummy_model_path);
        tinyvision::save_png_image(dummy_source_path, image);

        tinyvision::DenseMapConfig config_export;
        config_export.stride = 2;
        config_export.sentinel_nominal_10m = true;
        const auto result_export = tinyvision::classify_dense(model, image, config_export);
        tinyvision::export_dense_map(result_export, model, image, dummy_model_path, dummy_source_path, tmp_out);

        // Verify all 6 files exist
        if (!std::filesystem::is_regular_file(tmp_out / "classification.csv") ||
            !std::filesystem::is_regular_file(tmp_out / "run.json") ||
            !std::filesystem::is_regular_file(tmp_out / "class_map.png") ||
            !std::filesystem::is_regular_file(tmp_out / "confidence.png") ||
            !std::filesystem::is_regular_file(tmp_out / "margin.png") ||
            !std::filesystem::is_regular_file(tmp_out / "overlay.png")) {
            std::cerr << "exported dense artifacts missing on disk\n";
            return 1;
        }

        // Verify CSV header with grid_x and grid_y
        std::ifstream csv(tmp_out / "classification.csv");
        std::string header;
        std::getline(csv, header);
        if (header != "grid_x,grid_y,origin_x,origin_y,center_x,center_y,display_x,display_y,predicted_class,probability,second_class,second_probability,margin,status") {
            std::cerr << "CSV header mismatch: " << header << '\n';
            return 1;
        }

        // Verify raster dimensions
        const auto class_map_img = tinyvision::load_rgb_image(tmp_out / "class_map.png");
        const auto conf_map_img = tinyvision::load_rgb_image(tmp_out / "confidence.png");
        const auto margin_map_img = tinyvision::load_rgb_image(tmp_out / "margin.png");
        const auto overlay_img = tinyvision::load_rgb_image(tmp_out / "overlay.png");

        if (class_map_img.width != result_export.grid_width || class_map_img.height != result_export.grid_height ||
            conf_map_img.width != result_export.grid_width || conf_map_img.height != result_export.grid_height ||
            margin_map_img.width != result_export.grid_width || margin_map_img.height != result_export.grid_height) {
            std::cerr << "grid space raster dimension mismatch\n";
            return 1;
        }

        if (overlay_img.width != image.width || overlay_img.height != image.height) {
            std::cerr << "overlay raster dimension mismatch with source image\n";
            return 1;
        }

        // Verify that class_map pixels strictly match the canonical palette colors
        for (const auto& d : result_export.decisions) {
            const std::size_t idx = (d.grid_y * class_map_img.width + d.grid_x) * 3;
            const auto expected_rgb = result_export.palette.classes[d.predicted_index].rgb;
            if (class_map_img.pixels[idx + 0] != expected_rgb[0] ||
                class_map_img.pixels[idx + 1] != expected_rgb[1] ||
                class_map_img.pixels[idx + 2] != expected_rgb[2]) {
                std::cerr << "class_map pixel color mismatch with canonical palette\n";
                return 1;
            }
        }

        // Verify overlay cell centering on first decision:
        // stride = 2: display_x = 0 + 4 = 4, display_y = 0 + 4 = 4.
        // cell_x0 = 4 - 1 = 3, cell_y0 = 4 - 1 = 3.
        // Pixels (3, 3), (4, 3), (3, 4), (4, 4) must be blended with class color.
        {
            const auto& first_dec = result_export.decisions[0];
            const auto expected_rgb = result_export.palette.classes[first_dec.predicted_index].rgb;
            for (std::size_t py = 3; py <= 4; ++py) {
                for (std::size_t px = 3; px <= 4; ++px) {
                    const std::size_t idx = (py * overlay_img.width + px) * 3;
                    const auto expected_r = static_cast<std::uint8_t>(0.5 * image.pixels[idx + 0] + 0.5 * expected_rgb[0]);
                    const auto expected_g = static_cast<std::uint8_t>(0.5 * image.pixels[idx + 1] + 0.5 * expected_rgb[1]);
                    const auto expected_b = static_cast<std::uint8_t>(0.5 * image.pixels[idx + 2] + 0.5 * expected_rgb[2]);

                    if (overlay_img.pixels[idx + 0] != expected_r ||
                        overlay_img.pixels[idx + 1] != expected_g ||
                        overlay_img.pixels[idx + 2] != expected_b) {
                        std::cerr << "overlay centered cell pixel blending mismatch at (" << px << ", " << py << ")\n";
                        return 1;
                    }
                }
            }
        }

        // Cleanup
        std::error_code ec;
        std::filesystem::remove_all(tmp_out, ec);

        std::cout << "TinyLogicVision dense spatial classification test PASS\n"
                  << "extraction_witness=EXACT_RGB_INPUT_VECTOR\n"
                  << "decisions_stride1=221 decisions_stride2=63\n"
                  << "center_geometry=EXACT repeatability=EXACT\n"
                  << "overlay_cell_centering=EXACT\n"
                  << "unified_palette=PASS margin_map=PASS\n";

    } catch (const std::exception& error) {
        std::cerr << "dense_test error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
