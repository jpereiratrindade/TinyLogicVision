#include "tinyvision/image.hpp"

#include <png.h>

extern "C" {
#include <jpeglib.h>
}

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <setjmp.h>
#include <stdexcept>
#include <string>

namespace tinyvision {
namespace {

std::string lowercase_extension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension;
}

RgbImage load_png(const std::filesystem::path& path) {
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    const auto filename = path.string();
    if (png_image_begin_read_from_file(&png, filename.c_str()) == 0) {
        throw std::runtime_error("cannot read PNG " + filename + ": " + png.message);
    }
    png.format = PNG_FORMAT_RGB;

    RgbImage image;
    image.width = png.width;
    image.height = png.height;
    image.pixels.resize(PNG_IMAGE_SIZE(png));
    if (png_image_finish_read(&png, nullptr, image.pixels.data(), 0, nullptr) == 0) {
        const std::string message = png.message;
        png_image_free(&png);
        throw std::runtime_error("cannot decode PNG " + filename + ": " + message);
    }
    png_image_free(&png);
    return image;
}

struct JpegErrorContext {
    jpeg_error_mgr manager{};
    jmp_buf jump{};
    char message[JMSG_LENGTH_MAX]{};
    std::uint8_t* decoded{};
    std::FILE* file{};
    bool decoder_created{};
};

extern "C" void handle_jpeg_error(j_common_ptr common) {
    auto* context = reinterpret_cast<JpegErrorContext*>(common->err);
    context->manager.format_message(common, context->message);
    longjmp(context->jump, 1);
}

RgbImage load_jpeg(const std::filesystem::path& path) {
    const auto filename = path.string();
    JpegErrorContext error;
    error.file = std::fopen(filename.c_str(), "rb");
    if (error.file == nullptr) throw std::runtime_error("cannot open JPEG " + filename);

    jpeg_decompress_struct decoder{};
    decoder.err = jpeg_std_error(&error.manager);
    error.manager.error_exit = handle_jpeg_error;

    if (setjmp(error.jump) != 0) {
        if (error.decoded != nullptr) std::free(error.decoded);
        if (error.decoder_created) jpeg_destroy_decompress(&decoder);
        std::fclose(error.file);
        throw std::runtime_error("cannot decode JPEG " + filename + ": " + error.message);
    }

    jpeg_create_decompress(&decoder);
    error.decoder_created = true;
    jpeg_stdio_src(&decoder, error.file);
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);

    const auto row_bytes = static_cast<std::size_t>(decoder.output_width) * 3;
    const auto byte_count = row_bytes * static_cast<std::size_t>(decoder.output_height);
    error.decoded = static_cast<std::uint8_t*>(std::malloc(byte_count));
    if (error.decoded == nullptr) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(error.file);
        throw std::bad_alloc();
    }

    while (decoder.output_scanline < decoder.output_height) {
        auto* row = error.decoded + static_cast<std::size_t>(decoder.output_scanline) * row_bytes;
        JSAMPROW rows[] = {row};
        jpeg_read_scanlines(&decoder, rows, 1);
    }

    const auto width = static_cast<std::size_t>(decoder.output_width);
    const auto height = static_cast<std::size_t>(decoder.output_height);
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    error.decoder_created = false;
    std::fclose(error.file);
    error.file = nullptr;

    RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(error.decoded, error.decoded + byte_count);
    std::free(error.decoded);
    error.decoded = nullptr;
    return image;
}

} // namespace

RgbImage load_rgb_image(const std::filesystem::path& path) {
    const auto extension = lowercase_extension(path);
    if (extension == ".png") return load_png(path);
    if (extension == ".jpg" || extension == ".jpeg") return load_jpeg(path);
    throw std::invalid_argument("unsupported image extension: " + path.string());
}

void save_png_image(const std::filesystem::path& path, const RgbImage& image) {
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() != image.width * image.height * 3) {
        throw std::invalid_argument("invalid RGB image to save: " + path.string());
    }
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    png.width = static_cast<png_uint_32>(image.width);
    png.height = static_cast<png_uint_32>(image.height);
    png.format = PNG_FORMAT_RGB;
    const auto filename = path.string();
    if (png_image_write_to_file(&png, filename.c_str(), 0, image.pixels.data(), 0, nullptr) == 0) {
        const std::string message = png.message;
        throw std::runtime_error("failed to save PNG " + filename + ": " + message);
    }
}


std::vector<double> resize_rgb_bilinear(const RgbImage& image,
                                        std::size_t output_width,
                                        std::size_t output_height) {
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() != image.width * image.height * 3) {
        throw std::invalid_argument("invalid RGB image");
    }
    if (output_width == 0 || output_height == 0) {
        throw std::invalid_argument("output dimensions must be positive");
    }

    std::vector<double> output(output_width * output_height * 3);
    for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
        const double source_y = (static_cast<double>(output_y) + 0.5) *
                                    static_cast<double>(image.height) /
                                    static_cast<double>(output_height) -
                                0.5;
        const double bounded_y = std::clamp(source_y, 0.0, static_cast<double>(image.height - 1));
        const auto y0 = static_cast<std::size_t>(bounded_y);
        const auto y1 = std::min(y0 + 1, image.height - 1);
        const double y_weight = bounded_y - static_cast<double>(y0);

        for (std::size_t output_x = 0; output_x < output_width; ++output_x) {
            const double source_x = (static_cast<double>(output_x) + 0.5) *
                                        static_cast<double>(image.width) /
                                        static_cast<double>(output_width) -
                                    0.5;
            const double bounded_x = std::clamp(
                source_x, 0.0, static_cast<double>(image.width - 1));
            const auto x0 = static_cast<std::size_t>(bounded_x);
            const auto x1 = std::min(x0 + 1, image.width - 1);
            const double x_weight = bounded_x - static_cast<double>(x0);

            for (std::size_t channel = 0; channel < 3; ++channel) {
                const auto pixel = [&](std::size_t x, std::size_t y) {
                    return static_cast<double>(image.pixels[(y * image.width + x) * 3 + channel]);
                };
                const double top = pixel(x0, y0) * (1.0 - x_weight) +
                                   pixel(x1, y0) * x_weight;
                const double bottom = pixel(x0, y1) * (1.0 - x_weight) +
                                      pixel(x1, y1) * x_weight;
                output[(output_y * output_width + output_x) * 3 + channel] =
                    (top * (1.0 - y_weight) + bottom * y_weight) / 255.0;
            }
        }
    }
    return output;
}

} // namespace tinyvision
