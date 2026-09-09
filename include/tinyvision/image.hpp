#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace tinyvision {

struct RgbImage {
    std::size_t width{};
    std::size_t height{};
    std::vector<std::uint8_t> pixels;
};

RgbImage load_rgb_image(const std::filesystem::path& path);
void save_png_image(const std::filesystem::path& path, const RgbImage& image);
std::vector<double> resize_rgb_bilinear(const RgbImage& image,
                                        std::size_t output_width = 8,
                                        std::size_t output_height = 8);

} // namespace tinyvision
