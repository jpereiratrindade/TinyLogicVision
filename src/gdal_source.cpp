#include "tinyvision/gdal_source.hpp"

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TINYVISION_WITH_GDAL
#include <gdal_priv.h>
#endif

namespace tinyvision {

struct GdalMultibandSource::Impl {
    InputSchema schema{InputSchema::create_sentinel2_10m(8, 8)};
    GeoMetadata metadata;
    std::size_t width{};
    std::size_t height{};
    mutable std::mutex read_mutex;
#ifdef TINYVISION_WITH_GDAL
    std::array<GDALDataset*, 4> datasets{};

    ~Impl() {
        for (auto* dataset : datasets) {
            if (dataset != nullptr) GDALClose(dataset);
        }
    }
#endif
};

#ifdef TINYVISION_WITH_GDAL
namespace {

GeoMetadata read_metadata(GDALDataset& dataset, const std::filesystem::path& path) {
    GeoMetadata metadata;
    metadata.raster_width = static_cast<std::size_t>(dataset.GetRasterXSize());
    metadata.raster_height = static_cast<std::size_t>(dataset.GetRasterYSize());

    double transform[6]{};
    const char* projection = dataset.GetProjectionRef();
    const bool has_projection = projection != nullptr && projection[0] != '\0';
    const bool has_transform = dataset.GetGeoTransform(transform) == CE_None;
    metadata.has_geo = has_projection && has_transform;
    if (has_projection) metadata.crs = projection;
    if (has_transform) {
        std::copy(std::begin(transform), std::end(transform), metadata.geotransform.begin());
        metadata.pixel_size_x = std::hypot(transform[1], transform[4]);
        metadata.pixel_size_y = std::hypot(transform[2], transform[5]);
    }

    if (dataset.GetRasterCount() < 1) {
        throw std::invalid_argument("raster has no bands: " + path.string());
    }
    int has_nodata = 0;
    metadata.nodata_value = dataset.GetRasterBand(1)->GetNoDataValue(&has_nodata);
    metadata.has_nodata = has_nodata != 0;
    return metadata;
}

} // namespace
#endif

GdalMultibandSource::GdalMultibandSource(const std::filesystem::path& b2,
                                         const std::filesystem::path& b3,
                                         const std::filesystem::path& b4,
                                         const std::filesystem::path& b8)
    : impl_(std::make_unique<Impl>()) {
#ifdef TINYVISION_WITH_GDAL
    GDALAllRegister();
    const std::array<std::filesystem::path, 4> paths{b2, b3, b4, b8};
    std::vector<GeoMetadata> metadata;
    metadata.reserve(paths.size());

    try {
        for (std::size_t index = 0; index < paths.size(); ++index) {
            auto* dataset = static_cast<GDALDataset*>(GDALOpenEx(
                paths[index].string().c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY,
                nullptr, nullptr, nullptr));
            if (dataset == nullptr) {
                throw std::invalid_argument("cannot open Sentinel-2 band with GDAL: " + paths[index].string());
            }
            impl_->datasets[index] = dataset;
            metadata.push_back(read_metadata(*dataset, paths[index]));
        }
        validate_multiband_alignment(metadata);
    } catch (...) {
        impl_.reset();
        throw;
    }

    impl_->metadata = metadata.front();
    impl_->width = metadata.front().raster_width;
    impl_->height = metadata.front().raster_height;
#else
    (void)b2;
    (void)b3;
    (void)b4;
    (void)b8;
    throw std::runtime_error("TinyLogicVision was built without GDAL support");
#endif
}

GdalMultibandSource::~GdalMultibandSource() = default;

bool GdalMultibandSource::available() noexcept {
#ifdef TINYVISION_WITH_GDAL
    return true;
#else
    return false;
#endif
}

std::size_t GdalMultibandSource::width() const { return impl_->width; }
std::size_t GdalMultibandSource::height() const { return impl_->height; }
std::size_t GdalMultibandSource::channels() const { return impl_->schema.channels; }
const InputSchema& GdalMultibandSource::schema() const { return impl_->schema; }
const GeoMetadata& GdalMultibandSource::spatial_metadata() const { return impl_->metadata; }

void GdalMultibandSource::read_window_into(std::size_t origin_x,
                                           std::size_t origin_y,
                                           std::span<double> out_buf) const {
#ifdef TINYVISION_WITH_GDAL
    constexpr std::size_t patch_width = 8;
    constexpr std::size_t patch_height = 8;
    if (out_buf.size() != patch_width * patch_height * impl_->schema.channels) {
        throw std::invalid_argument("output span size mismatch for Sentinel-2 patch");
    }
    if (origin_x + patch_width > impl_->width || origin_y + patch_height > impl_->height) {
        throw std::out_of_range("Sentinel-2 patch exceeds raster boundaries");
    }

    std::lock_guard lock(impl_->read_mutex);
    std::array<double, patch_width * patch_height> band_values{};
    for (std::size_t channel = 0; channel < impl_->datasets.size(); ++channel) {
        auto* band = impl_->datasets[channel]->GetRasterBand(1);
        const auto status = band->RasterIO(
            GF_Read,
            static_cast<int>(origin_x), static_cast<int>(origin_y),
            static_cast<int>(patch_width), static_cast<int>(patch_height),
            band_values.data(),
            static_cast<int>(patch_width), static_cast<int>(patch_height),
            GDT_Float64, 0, 0, nullptr);
        if (status != CE_None) {
            throw std::runtime_error("GDAL failed to read Sentinel-2 band window");
        }

        const auto& normalization = impl_->schema.channel_specs[channel].normalization;
        for (std::size_t pixel = 0; pixel < band_values.size(); ++pixel) {
            out_buf[pixel * impl_->schema.channels + channel] =
                band_values[pixel] * normalization.scale + normalization.offset;
        }
    }
#else
    (void)origin_x;
    (void)origin_y;
    (void)out_buf;
    throw std::runtime_error("TinyLogicVision was built without GDAL support");
#endif
}

void GdalMultibandSource::read_region_into(std::size_t origin_x, std::size_t origin_y,
                                           std::size_t region_width, std::size_t region_height,
                                           std::span<double> out_buf) const {
#ifdef TINYVISION_WITH_GDAL
    const std::size_t channels = impl_->schema.channels;
    if (origin_x + region_width > impl_->width || origin_y + region_height > impl_->height ||
        out_buf.size() != region_width * region_height * channels) {
        throw std::out_of_range("Sentinel-2 region exceeds raster bounds or output size is invalid");
    }
    std::lock_guard lock(impl_->read_mutex);
    std::vector<double> band_values(region_width * region_height);
    for (std::size_t channel = 0; channel < impl_->datasets.size(); ++channel) {
        auto* band = impl_->datasets[channel]->GetRasterBand(1);
        if (band->RasterIO(GF_Read, static_cast<int>(origin_x), static_cast<int>(origin_y),
                           static_cast<int>(region_width), static_cast<int>(region_height),
                           band_values.data(), static_cast<int>(region_width),
                           static_cast<int>(region_height), GDT_Float64, 0, 0, nullptr) != CE_None) {
            throw std::runtime_error("GDAL failed to read Sentinel-2 tile with halo");
        }
        const auto& normalization = impl_->schema.channel_specs[channel].normalization;
        for (std::size_t pixel = 0; pixel < band_values.size(); ++pixel) {
            out_buf[pixel * channels + channel] =
                band_values[pixel] * normalization.scale + normalization.offset;
        }
    }
#else
    (void)origin_x; (void)origin_y; (void)region_width; (void)region_height; (void)out_buf;
    throw std::runtime_error("TinyLogicVision was built without GDAL support");
#endif
}

} // namespace tinyvision
