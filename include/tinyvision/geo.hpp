#pragma once

#include "tinyvision/dense.hpp"
#include "tinyvision/image.hpp"
#include "tinyvision/schema.hpp"
#include "tinyvision/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tinyvision {

struct GeoMetadata {
    bool has_geo{false};
    std::string crs;
    std::array<double, 6> geotransform{0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    std::size_t raster_width{0};
    std::size_t raster_height{0};
    double pixel_size_x{1.0};
    double pixel_size_y{1.0};
    double nodata_value{-9999.0};
    bool has_nodata{false};
    std::string source_sha256;

    std::pair<double, double> pixel_to_map(double px, double py) const {
        double mx = geotransform[0] + px * geotransform[1] + py * geotransform[2];
        double my = geotransform[3] + px * geotransform[4] + py * geotransform[5];
        return {mx, my};
    }
};

class InputSource {
public:
    virtual ~InputSource() = default;
    virtual std::size_t width() const = 0;
    virtual std::size_t height() const = 0;
    virtual std::size_t channels() const = 0;
    virtual const InputSchema& schema() const = 0;
    virtual const GeoMetadata& spatial_metadata() const = 0;
    virtual void read_window_into(std::size_t origin_x, std::size_t origin_y, std::span<double> out_buf) const = 0;
};

class RgbImageSource : public InputSource {
public:
    explicit RgbImageSource(RgbImage image, GeoMetadata meta = {})
        : image_(std::move(image)),
          meta_(std::move(meta)),
          schema_(InputSchema::create_canonical_rgb(8, 8)) {
        meta_.raster_width = image_.width;
        meta_.raster_height = image_.height;
    }

    std::size_t width() const override { return image_.width; }
    std::size_t height() const override { return image_.height; }
    std::size_t channels() const override { return 3; }
    const InputSchema& schema() const override { return schema_; }
    const GeoMetadata& spatial_metadata() const override { return meta_; }

    void read_window_into(std::size_t origin_x, std::size_t origin_y, std::span<double> out_buf) const override {
        extract_rgb_input_vector_into(image_, origin_x, origin_y, out_buf);
    }

    const RgbImage& image() const noexcept { return image_; }

private:
    RgbImage image_;
    GeoMetadata meta_;
    InputSchema schema_;
};

class TensorSource : public InputSource {
public:
    explicit TensorSource(MultichannelTensor tensor, GeoMetadata meta = {})
        : tensor_(std::move(tensor)), meta_(std::move(meta)) {
        meta_.raster_width = tensor_.width;
        meta_.raster_height = tensor_.height;
    }

    std::size_t width() const override { return tensor_.width; }
    std::size_t height() const override { return tensor_.height; }
    std::size_t channels() const override { return tensor_.channels; }
    const InputSchema& schema() const override { return tensor_.schema; }
    const GeoMetadata& spatial_metadata() const override { return meta_; }

    void read_window_into(std::size_t origin_x, std::size_t origin_y, std::span<double> out_buf) const override {
        tensor_.extract_normalized_patch_into(origin_x, origin_y, out_buf);
    }

    const MultichannelTensor& tensor() const noexcept { return tensor_; }

private:
    MultichannelTensor tensor_;
    GeoMetadata meta_;
};

void validate_multiband_alignment(const std::vector<GeoMetadata>& band_metas);

} // namespace tinyvision
