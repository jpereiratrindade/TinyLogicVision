#include "tinyvision/geo.hpp"
#include "tinyvision/dense.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace tinyvision {

RgbImageSource::RgbImageSource(RgbImage image, GeoMetadata meta)
    : image_(std::move(image)),
      meta_(std::move(meta)),
      schema_(InputSchema::create_canonical_rgb(8, 8)) {
    meta_.raster_width = image_.width;
    meta_.raster_height = image_.height;
}

void RgbImageSource::read_window_into(std::size_t origin_x, std::size_t origin_y, std::span<double> out_buf) const {
    extract_rgb_input_vector_into(image_, origin_x, origin_y, out_buf);
}

void RgbImageSource::read_region_into(std::size_t origin_x, std::size_t origin_y,
                                      std::size_t region_width, std::size_t region_height,
                                      std::span<double> out_buf) const {
    if (origin_x + region_width > image_.width || origin_y + region_height > image_.height ||
        out_buf.size() != region_width * region_height * 3) {
        throw std::out_of_range("RGB region exceeds image bounds or output size is invalid");
    }
    for (std::size_t y = 0; y < region_height; ++y) {
        for (std::size_t x = 0; x < region_width; ++x) {
            const std::size_t source_index = ((origin_y + y) * image_.width + origin_x + x) * 3;
            const std::size_t output_index = (y * region_width + x) * 3;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                out_buf[output_index + channel] = static_cast<double>(image_.pixels[source_index + channel]) / 255.0;
            }
        }
    }
}

void TensorSource::read_region_into(std::size_t origin_x, std::size_t origin_y,
                                    std::size_t region_width, std::size_t region_height,
                                    std::span<double> out_buf) const {
    if (origin_x + region_width > tensor_.width || origin_y + region_height > tensor_.height ||
        out_buf.size() != region_width * region_height * tensor_.channels) {
        throw std::out_of_range("tensor region exceeds bounds or output size is invalid");
    }
    for (std::size_t y = 0; y < region_height; ++y) {
        for (std::size_t x = 0; x < region_width; ++x) {
            for (std::size_t channel = 0; channel < tensor_.channels; ++channel) {
                const double raw = tensor_.get_raw_value(origin_x + x, origin_y + y, channel);
                const auto& normalization = tensor_.schema.channel_specs[channel].normalization;
                double value = raw;
                if (normalization.type == NormalizationType::UINT8_DIV_255) value = raw / 255.0;
                else if (normalization.type == NormalizationType::LINEAR) {
                    value = raw * normalization.scale + normalization.offset;
                }
                out_buf[(y * region_width + x) * tensor_.channels + channel] = value;
            }
        }
    }
}

void validate_multiband_alignment(const std::vector<GeoMetadata>& band_metas) {
    if (band_metas.empty()) {
        throw std::invalid_argument("no band metadata provided for alignment check");
    }

    const auto& base = band_metas[0];
    for (std::size_t i = 1; i < band_metas.size(); ++i) {
        const auto& other = band_metas[i];

        if (base.raster_width != other.raster_width || base.raster_height != other.raster_height) {
            std::ostringstream ss;
            ss << "multiband dimension mismatch between band 0 (" << base.raster_width << 'x' << base.raster_height
               << ") and band " << i << " (" << other.raster_width << 'x' << other.raster_height << ")";
            throw std::invalid_argument(ss.str());
        }

        if (base.has_geo != other.has_geo) {
            throw std::invalid_argument("multiband georeferencing availability mismatch between band 0 and band " +
                                        std::to_string(i));
        }

        if (base.has_geo && other.has_geo) {
            if (base.crs != other.crs) {
                std::ostringstream ss;
                ss << "multiband CRS mismatch between band 0 (" << base.crs << ") and band " << i << " (" << other.crs << ")";
                throw std::invalid_argument(ss.str());
            }

            for (std::size_t gt_idx = 0; gt_idx < 6; ++gt_idx) {
                if (std::abs(base.geotransform[gt_idx] - other.geotransform[gt_idx]) > 1e-6) {
                    std::ostringstream ss;
                    ss << "multiband geotransform mismatch at index " << gt_idx << " between band 0 and band " << i;
                    throw std::invalid_argument(ss.str());
                }
            }
        }
    }
}

} // namespace tinyvision
