#include "tinyvision/application.hpp"
#include "tinyvision/image.hpp"

#include <png.h>

extern "C" {
#include <jpeglib.h>
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool same_metrics(const tinyvision::ApplicationMetrics& first,
                  const tinyvision::ApplicationMetrics& second) {
    return first.average_loss == second.average_loss &&
           first.accuracy == second.accuracy &&
           first.mean_true_probability == second.mean_true_probability &&
           first.confusion == second.confusion;
}

bool exactly_reproduces(const tinyvision::ApplicationTrainingResult& first,
                        const tinyvision::ApplicationTrainingResult& second) {
    if (first.model.class_names != second.model.class_names ||
        first.model.network.parameters() != second.model.network.parameters() ||
        !same_metrics(first.final_train, second.final_train) ||
        !same_metrics(first.final_development, second.final_development) ||
        first.milestones.size() != second.milestones.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.milestones.size(); ++index) {
        if (first.milestones[index].epoch != second.milestones[index].epoch ||
            !same_metrics(first.milestones[index].train, second.milestones[index].train) ||
            !same_metrics(first.milestones[index].development,
                          second.milestones[index].development)) {
            return false;
        }
    }
    return true;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("tinyvision-application-test-" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::vector<std::uint8_t> color_patch(std::uint8_t red,
                                      std::uint8_t green,
                                      std::uint8_t blue,
                                      int variation) {
    constexpr std::size_t width = 16;
    constexpr std::size_t height = 16;
    std::vector<std::uint8_t> pixels(width * height * 3);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const int delta = ((x + y) % 2 == 0) ? variation : -variation;
            const auto assign = [&](std::size_t channel, std::uint8_t base) {
                pixels[(y * width + x) * 3 + channel] = static_cast<std::uint8_t>(
                    std::clamp(static_cast<int>(base) + delta, 0, 255));
            };
            assign(0, red);
            assign(1, green);
            assign(2, blue);
        }
    }
    return pixels;
}

void write_png(const std::filesystem::path& path, const std::vector<std::uint8_t>& pixels) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = 16;
    image.height = 16;
    image.format = PNG_FORMAT_RGB;
    const auto filename = path.string();
    if (png_image_write_to_file(&image, filename.c_str(), 0, pixels.data(), 0, nullptr) == 0) {
        throw std::runtime_error("failed to create PNG fixture");
    }
}

void write_jpeg(const std::filesystem::path& path,
                const std::vector<std::uint8_t>& pixels) {
    const auto filename = path.string();
    std::FILE* file = std::fopen(filename.c_str(), "wb");
    if (file == nullptr) throw std::runtime_error("failed to create JPEG fixture");

    jpeg_compress_struct compressor{};
    jpeg_error_mgr error{};
    compressor.err = jpeg_std_error(&error);
    jpeg_create_compress(&compressor);
    jpeg_stdio_dest(&compressor, file);
    compressor.image_width = 16;
    compressor.image_height = 16;
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, 95, TRUE);
    jpeg_start_compress(&compressor, TRUE);
    while (compressor.next_scanline < compressor.image_height) {
        auto* row = const_cast<JSAMPLE*>(
            pixels.data() + static_cast<std::size_t>(compressor.next_scanline) * 16 * 3);
        JSAMPROW rows[] = {row};
        jpeg_write_scanlines(&compressor, rows, 1);
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);
    std::fclose(file);
}

void make_class(const std::filesystem::path& root,
                const std::string& name,
                std::uint8_t red,
                std::uint8_t green,
                std::uint8_t blue) {
    for (const auto& split : {"train", "dev", "probe"}) {
        std::filesystem::create_directories(root / split / name);
    }
    write_png(root / "train" / name / "01.png", color_patch(red, green, blue, 5));
    write_jpeg(root / "train" / name / "02.jpg", color_patch(red, green, blue, 9));
    write_png(root / "dev" / name / "01.png", color_patch(red, green, blue, 3));
    write_jpeg(root / "probe" / name / "01.jpg", color_patch(red, green, blue, 1));
}

} // namespace

int main() {
    try {
        TemporaryDirectory temporary;
        make_class(temporary.path(), "soil", 170, 100, 45);
        make_class(temporary.path(), "vegetation", 30, 180, 45);
        make_class(temporary.path(), "water", 25, 90, 200);

        const auto train = tinyvision::load_application_split(temporary.path() / "train");
        const auto development = tinyvision::load_application_split(
            temporary.path() / "dev", train.class_names);
        if (train.class_names != std::vector<std::string>{"soil", "vegetation", "water"} ||
            train.samples.size() != 6 || development.samples.size() != 3) {
            std::cerr << "application directory dataset contract failed\n";
            return 1;
        }

        const auto png = tinyvision::load_rgb_image(
            temporary.path() / "train" / "vegetation" / "01.png");
        const auto jpeg = tinyvision::load_rgb_image(
            temporary.path() / "train" / "water" / "02.jpg");
        const auto resized_png = tinyvision::resize_rgb_bilinear(png);
        if (png.width != 16 || png.height != 16 || jpeg.width != 16 || jpeg.height != 16 ||
            resized_png.size() != 192 || resized_png != tinyvision::resize_rgb_bilinear(png)) {
            std::cerr << "PNG/JPEG load or deterministic resize contract failed\n";
            return 1;
        }

        tinyvision::ApplicationTrainingConfig config;
        config.epochs = 60;
        auto result = tinyvision::train_application(train, development, config);
        const auto repeated = tinyvision::train_application(train, development, config);
        if (result.model.network.parameters().size() != 4707 ||
            result.final_train.accuracy != 1.0 || result.final_development.accuracy != 1.0) {
            std::cerr << "application fixture was not learned\n";
            return 1;
        }
        if (!exactly_reproduces(result, repeated)) {
            std::cerr << "application training trajectory is not exactly reproducible\n";
            return 1;
        }

        const auto model_path = temporary.path() / "fixture-model.tlv";
        const auto expected_parameters = result.model.network.parameters();
        tinyvision::save_application_model(result.model, model_path);
        auto loaded = tinyvision::load_application_model(model_path);
        if (loaded.class_names != result.model.class_names ||
            loaded.network.parameters() != expected_parameters) {
            std::cerr << "application model did not round-trip exactly\n";
            return 1;
        }

        const auto probe = tinyvision::load_application_split(
            temporary.path() / "probe", loaded.class_names);
        const auto probe_metrics = tinyvision::evaluate_application(loaded.network, probe);
        if (probe.samples.size() != 3 || probe_metrics.accuracy != 1.0) {
            std::cerr << "held-out application fixture evaluation failed\n";
            return 1;
        }

        std::cout << "TV-APP-00 engineering pipeline PASS"
                  << " classes=3 parameters=4707"
                  << " train=1 dev=1 probe_fixture=1 repetition=EXACT"
                  << " synthetic_test_status=NOT_EVALUATED\n";
    } catch (const std::exception& error) {
        std::cerr << "application pipeline test: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
