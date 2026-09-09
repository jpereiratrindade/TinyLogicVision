#include "tinyvision/dense.hpp"
#include "tinyvision/h3_index.hpp"
#include "tinyvision/image.hpp"
#include "tinyvision/provenance.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef TINYVISION_WITH_GDAL
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#endif

namespace tinyvision {
namespace {

struct Sha256Context {
    std::uint32_t state[8]{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                           0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::uint64_t count{0};
    std::uint8_t buffer[64]{0};
};

std::uint32_t right_rotate(std::uint32_t value, std::uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

void sha256_transform(Sha256Context& ctx, const std::uint8_t data[64]) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    std::uint32_t w[64];
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = (static_cast<std::uint32_t>(data[i * 4]) << 24) |
               (static_cast<std::uint32_t>(data[i * 4 + 1]) << 16) |
               (static_cast<std::uint32_t>(data[i * 4 + 2]) << 8) |
               (static_cast<std::uint32_t>(data[i * 4 + 3]));
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 = right_rotate(w[i - 15], 7) ^ right_rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = right_rotate(w[i - 2], 17) ^ right_rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
    std::uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t s1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        const std::uint32_t s0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx.state[0] += a; ctx.state[1] += b; ctx.state[2] += c; ctx.state[3] += d;
    ctx.state[4] += e; ctx.state[5] += f; ctx.state[6] += g; ctx.state[7] += h;
}

void sha256_update(Sha256Context& ctx, const std::uint8_t* data, std::size_t length) {
    std::size_t buffer_index = ctx.count % 64;
    ctx.count += length;

    for (std::size_t i = 0; i < length; ++i) {
        ctx.buffer[buffer_index++] = data[i];
        if (buffer_index == 64) {
            sha256_transform(ctx, ctx.buffer);
            buffer_index = 0;
        }
    }
}

std::string sha256_final(Sha256Context& ctx) {
    const std::uint64_t total_bits = ctx.count * 8;
    std::size_t buffer_index = ctx.count % 64;

    ctx.buffer[buffer_index++] = 0x80;
    if (buffer_index > 56) {
        while (buffer_index < 64) ctx.buffer[buffer_index++] = 0;
        sha256_transform(ctx, ctx.buffer);
        buffer_index = 0;
    }
    while (buffer_index < 56) ctx.buffer[buffer_index++] = 0;

    for (int i = 7; i >= 0; --i) {
        ctx.buffer[56 + (7 - i)] = static_cast<std::uint8_t>((total_bits >> (i * 8)) & 0xff);
    }
    sha256_transform(ctx, ctx.buffer);

    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < 8; ++i) {
        ss << std::setw(8) << ctx.state[i];
    }
    return ss.str();
}

std::string compute_file_sha256(const std::filesystem::path& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) return "";
    Sha256Context ctx;
    char buffer[65536];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        sha256_update(ctx, reinterpret_cast<const std::uint8_t*>(buffer), static_cast<std::size_t>(file.gcount()));
    }
    return sha256_final(ctx);
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (ch < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
            } else {
                escaped << static_cast<char>(ch);
            }
        }
    }
    return escaped.str();
}

std::array<std::uint8_t, 3> get_decision_color(const PaletteConfig& palette,
                                               std::size_t class_index,
                                               bool is_uncertain) {
    if (is_uncertain) {
        return palette.uncertain.rgb;
    }
    if (class_index < palette.classes.size()) {
        return palette.classes[class_index].rgb;
    }
    return {128, 128, 128};
}

enum class GeoRasterType { byte, float32 };

void write_geotiff_raster(const std::filesystem::path& path,
                          const void* pixel_data,
                          std::size_t byte_count,
                          std::size_t width,
                          std::size_t height,
                          GeoRasterType raster_type,
                          const GeoMetadata& meta,
                          std::size_t stride) {
#ifdef TINYVISION_WITH_GDAL
    const std::size_t bytes_per_pixel = raster_type == GeoRasterType::byte ? 1 : sizeof(float);
    if (width == 0 || height == 0 || byte_count != width * height * bytes_per_pixel) {
        throw std::invalid_argument("invalid GeoTIFF raster buffer dimensions");
    }

    GDALAllRegister();
    auto* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (driver == nullptr) {
        throw std::runtime_error("GDAL GTiff driver is unavailable");
    }
    const GDALDataType data_type = raster_type == GeoRasterType::byte ? GDT_Byte : GDT_Float32;
    auto* dataset = driver->Create(path.string().c_str(), static_cast<int>(width),
                                   static_cast<int>(height), 1, data_type, nullptr);
    if (dataset == nullptr) {
        throw std::runtime_error("cannot create GeoTIFF: " + path.string());
    }

    try {
        // Each output pixel represents one decision point. Its center must coincide
        // with the center of the source 8x8 support, while its basis follows stride.
        const double center_offset = 3.5 - static_cast<double>(stride) / 2.0;
        double output_transform[6]{
            meta.geotransform[0] + center_offset * meta.geotransform[1] + center_offset * meta.geotransform[2],
            meta.geotransform[1] * static_cast<double>(stride),
            meta.geotransform[2] * static_cast<double>(stride),
            meta.geotransform[3] + center_offset * meta.geotransform[4] + center_offset * meta.geotransform[5],
            meta.geotransform[4] * static_cast<double>(stride),
            meta.geotransform[5] * static_cast<double>(stride),
        };
        if (dataset->SetGeoTransform(output_transform) != CE_None) {
            throw std::runtime_error("cannot set GeoTIFF geotransform: " + path.string());
        }

        OGRSpatialReference reference;
        reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        if (reference.SetFromUserInput(meta.crs.c_str()) != OGRERR_NONE ||
            dataset->SetSpatialRef(&reference) != CE_None) {
            throw std::runtime_error("cannot set GeoTIFF CRS '" + meta.crs + "': " + path.string());
        }

        auto* band = dataset->GetRasterBand(1);
        band->SetNoDataValue(raster_type == GeoRasterType::byte ? 255.0 : -9999.0);
        if (band->RasterIO(GF_Write, 0, 0, static_cast<int>(width), static_cast<int>(height),
                           const_cast<void*>(pixel_data), static_cast<int>(width),
                           static_cast<int>(height), data_type, 0, 0, nullptr) != CE_None) {
            throw std::runtime_error("cannot write GeoTIFF pixels: " + path.string());
        }
        dataset->FlushCache();
        GDALClose(dataset);
    } catch (...) {
        GDALClose(dataset);
        throw;
    }
#else
    (void)path;
    (void)pixel_data;
    (void)byte_count;
    (void)width;
    (void)height;
    (void)raster_type;
    (void)meta;
    (void)stride;
    throw std::runtime_error("georeferenced GeoTIFF export requires a build with GDAL support");
#endif
}

} // namespace

PaletteConfig get_canonical_palette(const std::vector<std::string>& class_names) {
    static const std::vector<std::pair<std::string, std::array<std::uint8_t, 3>>> palette_colors = {
        {"#22c55e", {34, 197, 94}},   // emerald
        {"#eab308", {234, 179, 8}},   // amber
        {"#f97316", {249, 115, 22}},  // orange
        {"#3b82f6", {59, 130, 246}},  // blue
        {"#a855f7", {168, 85, 247}},  // purple
        {"#ec4899", {236, 72, 153}},  // pink
        {"#14b8a6", {20, 184, 166}},  // teal
        {"#ef4444", {239, 68, 68}},   // red
    };

    PaletteConfig config;
    for (std::size_t i = 0; i < class_names.size(); ++i) {
        const auto& [hex, rgb] = palette_colors[i % palette_colors.size()];
        config.classes.push_back({class_names[i], i, hex, rgb});
    }
    config.uncertain = {"UNCERTAIN", 9999, "#808080", {128, 128, 128}};
    return config;
}

void extract_rgb_input_vector_into(const RgbImage& image,
                                   std::size_t origin_x,
                                   std::size_t origin_y,
                                   std::span<double> output_192) {
    if (origin_x + 8 > image.width || origin_y + 8 > image.height) {
        throw std::out_of_range("8x8 patch exceeds image boundaries");
    }
    if (image.pixels.size() != image.width * image.height * 3) {
        throw std::invalid_argument("invalid RGB image byte count");
    }
    if (output_192.size() != 192) {
        throw std::invalid_argument("output span must have size 192");
    }

    for (std::size_t wy = 0; wy < 8; ++wy) {
        const std::size_t src_row = ((origin_y + wy) * image.width + origin_x) * 3;
        const std::size_t dst_row = (wy * 8) * 3;
        for (std::size_t wx = 0; wx < 8; ++wx) {
            output_192[dst_row + wx * 3 + 0] = static_cast<double>(image.pixels[src_row + wx * 3 + 0]) / 255.0;
            output_192[dst_row + wx * 3 + 1] = static_cast<double>(image.pixels[src_row + wx * 3 + 1]) / 255.0;
            output_192[dst_row + wx * 3 + 2] = static_cast<double>(image.pixels[src_row + wx * 3 + 2]) / 255.0;
        }
    }
}

std::vector<double> extract_rgb_input_vector(const RgbImage& image,
                                             std::size_t origin_x,
                                             std::size_t origin_y) {
    std::vector<double> window(192);
    extract_rgb_input_vector_into(image, origin_x, origin_y, window);
    return window;
}

DenseMapResult classify_dense_source(const ApplicationModel& model,
                                     const InputSource& source,
                                     const DenseMapConfig& config) {
    if (config.stride != 1 && config.stride != 2 && config.stride != 4 && config.stride != 8) {
        throw std::invalid_argument("stride must be 1, 2, 4, or 8");
    }
    if (source.width() < 8 || source.height() < 8) {
        throw std::invalid_argument("raster source must be at least 8x8 pixels");
    }
    if (model.schema != source.schema()) {
        throw std::invalid_argument("model input schema does not match raster source schema");
    }
    if (model.network.input_size() != source.schema().input_size()) {
        throw std::invalid_argument("model input count does not match raster source schema");
    }

    const std::size_t grid_width = (source.width() - 8) / config.stride + 1;
    const std::size_t grid_height = (source.height() - 8) / config.stride + 1;
    if (grid_height != 0 && grid_width > std::numeric_limits<std::size_t>::max() / grid_height) {
        throw std::overflow_error("dense decision grid size overflow");
    }
    const std::size_t total_decisions = grid_width * grid_height;
    if (config.max_decisions != 0 && total_decisions > config.max_decisions) {
        throw std::length_error(
            "dense grid has " + std::to_string(total_decisions) +
            " decisions, exceeding the in-memory safety limit of " +
            std::to_string(config.max_decisions) +
            "; increase stride or explicitly set --max-decisions 0/N");
    }

    DenseMapResult result;
    result.source_width = source.width();
    result.source_height = source.height();
    result.grid_width = grid_width;
    result.grid_height = grid_height;
    result.total_decisions = total_decisions;
    result.config = config;
    result.palette = get_canonical_palette(model.class_names);
    result.metadata = source.spatial_metadata();
    result.decisions.resize(total_decisions);
    result.compact_decisions.resize(total_decisions);

    std::size_t num_threads = config.threads;
    if (num_threads == 0) {
        num_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    num_threads = std::min<std::size_t>(num_threads, grid_height);
    result.thread_count = num_threads;
    result.implementation_mode = "PARALLEL_IN_MEMORY";

    const std::size_t num_classes = model.class_names.size();
    const std::size_t input_elements = model.network.input_size();

    auto process_rows = [&](std::size_t gy_start, std::size_t gy_end) {
        MLPWorkspace ws;
        std::vector<double> patch_buf(input_elements);

        for (std::size_t gy = gy_start; gy < gy_end; ++gy) {
            const std::size_t y = gy * config.stride;
            const std::size_t row_offset = gy * grid_width;

            for (std::size_t gx = 0; gx < grid_width; ++gx) {
                const std::size_t x = gx * config.stride;
                const std::size_t dec_idx = row_offset + gx;

                source.read_window_into(x, y, patch_buf);
                model.network.forward_into(patch_buf, ws);
                const auto& probabilities = ws.probabilities;

                std::size_t top1_idx = 0;
                std::size_t top2_idx = 0;
                double top1_prob = -1.0;
                double top2_prob = -1.0;

                for (std::size_t c = 0; c < num_classes; ++c) {
                    if (probabilities[c] > top1_prob) {
                        top2_idx = top1_idx;
                        top2_prob = top1_prob;
                        top1_idx = c;
                        top1_prob = probabilities[c];
                    } else if (probabilities[c] > top2_prob) {
                        top2_idx = c;
                        top2_prob = probabilities[c];
                    }
                }

                const double margin = top1_prob - (num_classes > 1 ? top2_prob : 0.0);
                const bool is_uncertain = (top1_prob < config.confidence_threshold) ||
                                          (margin < config.margin_threshold);

                const auto [map_x, map_y] = result.metadata.pixel_to_map(static_cast<double>(x) + 3.5, static_cast<double>(y) + 3.5);

                auto& cd = result.compact_decisions[dec_idx];
                cd.grid_x = static_cast<std::uint32_t>(gx);
                cd.grid_y = static_cast<std::uint32_t>(gy);
                cd.origin_x = static_cast<std::uint32_t>(x);
                cd.origin_y = static_cast<std::uint32_t>(y);
                cd.predicted_index = static_cast<std::uint16_t>(top1_idx);
                cd.second_index = static_cast<std::uint16_t>(top2_idx);
                cd.probability = static_cast<float>(top1_prob);
                cd.second_probability = static_cast<float>(num_classes > 1 ? top2_prob : 0.0);
                cd.margin = static_cast<float>(margin);
                cd.is_uncertain = is_uncertain;

                auto& d = result.decisions[dec_idx];
                d.grid_x = gx;
                d.grid_y = gy;
                d.origin_x = x;
                d.origin_y = y;
                d.center_x = static_cast<double>(x) + 3.5;
                d.center_y = static_cast<double>(y) + 3.5;
                d.display_x = x + 4;
                d.display_y = y + 4;
                d.map_x = map_x;
                d.map_y = map_y;
                d.predicted_index = top1_idx;
                d.predicted_class = model.class_names[top1_idx];
                d.probability = top1_prob;
                d.second_index = top2_idx;
                d.second_class = num_classes > 1 ? model.class_names[top2_idx] : "";
                d.second_probability = num_classes > 1 ? top2_prob : 0.0;
                d.margin = margin;
                d.is_uncertain = is_uncertain;
                d.status = is_uncertain ? "UNCERTAIN" : "CLASSIFIED";
            }
        }
    };

    if (num_threads <= 1) {
        process_rows(0, grid_height);
    } else {
        std::vector<std::future<void>> futures;
        const std::size_t rows_per_thread = (grid_height + num_threads - 1) / num_threads;
        for (std::size_t t = 0; t < num_threads; ++t) {
            const std::size_t start_row = t * rows_per_thread;
            const std::size_t end_row = std::min(grid_height, start_row + rows_per_thread);
            if (start_row < end_row) {
                futures.push_back(std::async(std::launch::async, process_rows, start_row, end_row));
            }
        }
        for (auto& f : futures) {
            f.get();
        }
    }

    std::size_t classified = 0;
    std::size_t uncertain = 0;
    for (const auto& cd : result.compact_decisions) {
        if (cd.is_uncertain) {
            ++uncertain;
        } else {
            ++classified;
        }
    }
    result.classified_count = classified;
    result.uncertain_count = uncertain;

    return result;
}

DenseMapResult classify_dense(const ApplicationModel& model,
                              const RgbImage& image,
                              const DenseMapConfig& config,
                              const GeoMetadata& meta) {
    RgbImageSource source(image, meta);
    return classify_dense_source(model, source, config);
}

void export_dense_map(const DenseMapResult& result,
                      const ApplicationModel& model,
                      const RgbImage& source_image,
                      const std::filesystem::path& model_path,
                      const std::filesystem::path& source_path,
                      const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);

    // 1. Export classification.csv (with map_x, map_y when geo is available)
    const auto csv_path = output_dir / "classification.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        throw std::runtime_error("failed to open " + csv_path.string() + " for writing");
    }

    if (result.metadata.has_geo) {
        csv << "grid_x,grid_y,origin_x,origin_y,center_x,center_y,display_x,display_y,map_x,map_y,predicted_class,probability,second_class,second_probability,margin,status\n";
        for (const auto& d : result.decisions) {
            csv << d.grid_x << ',' << d.grid_y << ','
                << d.origin_x << ',' << d.origin_y << ','
                << std::fixed << std::setprecision(1) << d.center_x << ',' << d.center_y << ','
                << d.display_x << ',' << d.display_y << ','
                << std::setprecision(3) << d.map_x << ',' << d.map_y << ','
                << d.predicted_class << ','
                << std::setprecision(6) << d.probability << ','
                << d.second_class << ','
                << d.second_probability << ','
                << d.margin << ','
                << d.status << '\n';
        }
    } else {
        csv << "grid_x,grid_y,origin_x,origin_y,center_x,center_y,display_x,display_y,predicted_class,probability,second_class,second_probability,margin,status\n";
        for (const auto& d : result.decisions) {
            csv << d.grid_x << ',' << d.grid_y << ','
                << d.origin_x << ',' << d.origin_y << ','
                << std::fixed << std::setprecision(1) << d.center_x << ',' << d.center_y << ','
                << d.display_x << ',' << d.display_y << ','
                << d.predicted_class << ','
                << std::setprecision(6) << d.probability << ','
                << d.second_class << ','
                << d.second_probability << ','
                << d.margin << ','
                << d.status << '\n';
        }
    }

    // 2. Export decisions.bin for O(1) indexed lookup
    const auto bin_path = output_dir / "decisions.bin";
    std::ofstream binf(bin_path, std::ios::binary | std::ios::trunc);
    if (binf) {
        char header[64]{0};
        std::memcpy(header, "TLV_DEC\0", 8);
        const std::uint32_t version = 1;
        const std::uint32_t gw = static_cast<std::uint32_t>(result.grid_width);
        const std::uint32_t gh = static_cast<std::uint32_t>(result.grid_height);
        const std::uint32_t stride = static_cast<std::uint32_t>(result.config.stride);
        const std::uint32_t class_count = static_cast<std::uint32_t>(model.class_names.size());
        const std::uint32_t record_size = sizeof(CompactDecisionRecord);

        std::memcpy(header + 8, &version, 4);
        std::memcpy(header + 12, &gw, 4);
        std::memcpy(header + 16, &gh, 4);
        std::memcpy(header + 20, &stride, 4);
        std::memcpy(header + 24, &class_count, 4);
        std::memcpy(header + 28, &record_size, 4);
        binf.write(header, 64);

        for (const auto& cd : result.compact_decisions) {
            CompactDecisionRecord rec{};
            rec.grid_x = cd.grid_x;
            rec.grid_y = cd.grid_y;
            rec.origin_x = cd.origin_x;
            rec.origin_y = cd.origin_y;
            rec.predicted_index = cd.predicted_index;
            rec.second_index = cd.second_index;
            rec.probability = cd.probability;
            rec.second_probability = cd.second_probability;
            rec.margin = cd.margin;
            rec.is_uncertain = cd.is_uncertain ? 1 : 0;
            binf.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
        }
    }

    // 3. Export run.json
    const auto json_path = output_dir / "run.json";
    std::ofstream js(json_path);
    if (!js) {
        throw std::runtime_error("failed to open " + json_path.string() + " for writing");
    }

    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&time_t_now, &tm_buf);
    char timestamp_buf[64];
    std::strftime(timestamp_buf, sizeof(timestamp_buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    std::map<std::string, std::size_t> class_counts;
    for (const auto& name : model.class_names) class_counts[name] = 0;
    for (const auto& d : result.decisions) {
        if (!d.is_uncertain) {
            class_counts[d.predicted_class]++;
        }
    }

    const std::string model_sha = compute_file_sha256(model_path);
    const std::string source_sha = compute_file_sha256(source_path);

    js << "{\n"
       << "  \"run_id\": \"" << json_escape(output_dir.filename().string()) << "\",\n"
       << "  \"timestamp\": \"" << timestamp_buf << "\",\n"
       << "  \"model_path\": \"" << json_escape(model_path.string()) << "\",\n"
       << "  \"model_sha256\": \"" << model_sha << "\",\n"
       << "  \"source_image\": \"" << json_escape(source_path.filename().string()) << "\",\n"
       << "  \"source_image_path\": \"" << json_escape(source_path.string()) << "\",\n"
       << "  \"source_sha256\": \"" << source_sha << "\",\n"
       << "  \"source_width\": " << result.source_width << ",\n"
       << "  \"source_height\": " << result.source_height << ",\n"
       << "  \"preview_width\": " << source_image.width << ",\n"
       << "  \"preview_height\": " << source_image.height << ",\n"
       << "  \"input_modality\": \"" << json_escape(modality_to_string(model.schema.modality)) << "\",\n"
       << "  \"input_channels\": " << model.schema.channels << ",\n"
       << "  \"patch_width\": 8,\n"
       << "  \"patch_height\": 8,\n"
       << "  \"stride\": " << result.config.stride << ",\n"
       << "  \"grid_width\": " << result.grid_width << ",\n"
       << "  \"grid_height\": " << result.grid_height << ",\n"
       << "  \"decision_count\": " << result.total_decisions << ",\n"
       << "  \"classes\": [";
    for (std::size_t i = 0; i < model.class_names.size(); ++i) {
        js << "\"" << json_escape(model.class_names[i]) << "\"" << (i + 1 < model.class_names.size() ? ", " : "");
    }
    js << "],\n"
       << "  \"spatial_semantics\": {\n"
       << "    \"support\": \"8x8 pixels [origin_x, origin_x+7] x [origin_y, origin_y+7]\",\n"
       << "    \"center_geometry\": \"Exact geometric center (origin_x + 3.5, origin_y + 3.5)\",\n"
       << "    \"display_anchor\": \"Integer visualization anchor (origin_x + 4, origin_y + 4)\",\n"
       << "    \"grid_coordinates\": \"Decision index in dense raster [0..grid_width-1] x [0..grid_height-1]\",\n"
       << "    \"concept_distinction\": \"SUPPORT != DECISION POINT != DISPLAY CELL\"\n"
       << "  },\n"
       << "  \"uncertainty_semantics\": {\n"
       << "    \"top1_probability\": \"Top-1 uncalibrated softmax probability\",\n"
       << "    \"margin\": \"Top-1 minus Top-2 softmax probability\",\n"
       << "    \"status_definition\": \"UNCERTAIN when top1 < confidence_threshold OR margin < margin_threshold (UNCERTAIN_BY_CONFIGURED_THRESHOLD)\"\n"
       << "  },\n"
       << "  \"engineering_stats\": {\n"
       << "    \"implementation_mode\": \"" << json_escape(result.implementation_mode) << "\",\n"
       << "    \"thread_count\": " << result.thread_count << ",\n"
       << "    \"tile_dimensions\": [" << result.config.tile_width << ", " << result.config.tile_height << "],\n"
       << "    \"max_decisions\": " << result.config.max_decisions << ",\n"
       << "    \"decision_record_version\": 1,\n"
       << "    \"indexed_binary_available\": true\n"
       << "  },\n"
       << "  \"confidence_threshold\": " << std::fixed << std::setprecision(4) << result.config.confidence_threshold << ",\n"
       << "  \"margin_threshold\": " << result.config.margin_threshold << ",\n"
       << "  \"sentinel_nominal_10m\": " << (result.config.sentinel_nominal_10m ? "true" : "false") << ",\n"
       << "  \"operator_declared_nominal_10m\": " << (result.config.sentinel_nominal_10m ? "true" : "false") << ",\n"
       << "  \"resolution_status\": \"" << (result.config.sentinel_nominal_10m ? "Operator declared nominal 10 m/pixel (unverified metadata)" : (result.metadata.has_geo ? "Georeferenced CRS metadata" : "Display image space (no metric scale declared)")) << "\",\n";
    if (result.config.sentinel_nominal_10m) {
        js << "  \"nominal_context_m\": 80.0,\n"
           << "  \"nominal_decision_spacing_m\": " << static_cast<double>(result.config.stride * 10) << ",\n";
    } else {
        js << "  \"nominal_context_m\": null,\n"
           << "  \"nominal_decision_spacing_m\": null,\n";
    }
    js << "  \"palette\": {\n"
       << "    \"classes\": [\n";
    for (std::size_t i = 0; i < result.palette.classes.size(); ++i) {
        const auto& c = result.palette.classes[i];
        js << "      {\"name\": \"" << json_escape(c.name) << "\", \"index\": " << c.index
           << ", \"color\": \"" << json_escape(c.hex_color) << "\", \"rgb\": ["
           << static_cast<int>(c.rgb[0]) << ", " << static_cast<int>(c.rgb[1]) << ", " << static_cast<int>(c.rgb[2]) << "]}"
           << (i + 1 < result.palette.classes.size() ? ",\n" : "\n");
    }
    js << "    ],\n"
       << "    \"uncertain\": {\"name\": \"" << json_escape(result.palette.uncertain.name) << "\", \"color\": \"" << json_escape(result.palette.uncertain.hex_color)
       << "\", \"rgb\": [" << static_cast<int>(result.palette.uncertain.rgb[0]) << ", "
       << static_cast<int>(result.palette.uncertain.rgb[1]) << ", "
       << static_cast<int>(result.palette.uncertain.rgb[2]) << "]}\n"
       << "  },\n"
       << "  \"geospatial\": {\n"
       << "    \"available\": " << (result.metadata.has_geo ? "true" : "false") << ",\n"
       << "    \"crs\": " << (result.metadata.has_geo ? ("\"" + json_escape(result.metadata.crs) + "\"") : "null") << ",\n"
       << "    \"geotransform\": ["
       << result.metadata.geotransform[0] << ", " << result.metadata.geotransform[1] << ", "
       << result.metadata.geotransform[2] << ", " << result.metadata.geotransform[3] << ", "
       << result.metadata.geotransform[4] << ", " << result.metadata.geotransform[5] << "],\n"
       << "    \"class_codes\": {\n";
    for (std::size_t i = 0; i < model.class_names.size(); ++i) {
        js << "      \"" << i << "\": \"" << json_escape(model.class_names[i]) << "\",\n";
    }
    js << "      \"254\": \"UNCERTAIN\",\n"
       << "      \"255\": \"NODATA\"\n"
       << "    }\n"
       << "  },\n"
       << "  \"h3\": {\n"
       << "    \"available\": " << (result.metadata.has_geo && h3_aggregation_available() ? "true" : "false") << ",\n"
       << "    \"implementation\": \"" << (h3_available() ? "OFFICIAL_H3" : "UNAVAILABLE") << "\",\n"
       << "    \"resolution\": " << (result.metadata.has_geo && h3_aggregation_available() ? "9" : "null") << "\n"
       << "  },\n"
       << "  \"summary\": {\n"
       << "    \"total_decisions\": " << result.total_decisions << ",\n"
       << "    \"classified_count\": " << result.classified_count << ",\n"
       << "    \"uncertain_count\": " << result.uncertain_count << ",\n"
       << "    \"class_counts\": {\n";
    std::size_t c_idx = 0;
    for (const auto& [name, count] : class_counts) {
        js << "      \"" << json_escape(name) << "\": " << count << (++c_idx < class_counts.size() ? ",\n" : "\n");
    }
    js << "    }\n"
       << "  }\n"
       << "}\n";
    js.close();
    if (!js) {
        throw std::runtime_error("failed to finalize " + json_path.string());
    }

    // 4. Export class_map.png
    RgbImage class_map;
    class_map.width = result.grid_width;
    class_map.height = result.grid_height;
    class_map.pixels.resize(result.grid_width * result.grid_height * 3);

    for (const auto& cd : result.compact_decisions) {
        const auto color = get_decision_color(result.palette, cd.predicted_index, cd.is_uncertain);
        const std::size_t idx = (cd.grid_y * result.grid_width + cd.grid_x) * 3;
        class_map.pixels[idx + 0] = color[0];
        class_map.pixels[idx + 1] = color[1];
        class_map.pixels[idx + 2] = color[2];
    }
    save_png_image(output_dir / "class_map.png", class_map);

    // 5. Export confidence.png
    RgbImage conf_map;
    conf_map.width = result.grid_width;
    conf_map.height = result.grid_height;
    conf_map.pixels.resize(result.grid_width * result.grid_height * 3);

    for (const auto& cd : result.compact_decisions) {
        const auto val = static_cast<std::uint8_t>(std::clamp(cd.probability * 255.0f, 0.0f, 255.0f));
        const std::size_t idx = (cd.grid_y * result.grid_width + cd.grid_x) * 3;
        conf_map.pixels[idx + 0] = val;
        conf_map.pixels[idx + 1] = val;
        conf_map.pixels[idx + 2] = val;
    }
    save_png_image(output_dir / "confidence.png", conf_map);

    // 6. Export margin.png
    RgbImage margin_map;
    margin_map.width = result.grid_width;
    margin_map.height = result.grid_height;
    margin_map.pixels.resize(result.grid_width * result.grid_height * 3);

    for (const auto& cd : result.compact_decisions) {
        const auto val = static_cast<std::uint8_t>(std::clamp(cd.margin * 255.0f, 0.0f, 255.0f));
        const std::size_t idx = (cd.grid_y * result.grid_width + cd.grid_x) * 3;
        margin_map.pixels[idx + 0] = val;
        margin_map.pixels[idx + 1] = val;
        margin_map.pixels[idx + 2] = val;
    }
    save_png_image(output_dir / "margin.png", margin_map);

    // 7. Export overlay.png
    RgbImage overlay = source_image;
    const double preview_scale_x = static_cast<double>(source_image.width) /
                                   static_cast<double>(result.source_width);
    const double preview_scale_y = static_cast<double>(source_image.height) /
                                   static_cast<double>(result.source_height);
    const std::size_t display_stride_x = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(static_cast<double>(result.config.stride) * preview_scale_x)));
    const std::size_t display_stride_y = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(static_cast<double>(result.config.stride) * preview_scale_y)));
    const std::int64_t half_stride_x = static_cast<std::int64_t>(display_stride_x / 2);
    const std::int64_t half_stride_y = static_cast<std::int64_t>(display_stride_y / 2);

    for (const auto& cd : result.compact_decisions) {
        const auto color = get_decision_color(result.palette, cd.predicted_index, cd.is_uncertain);
        const std::int64_t display_x = static_cast<std::int64_t>(std::floor(
            (static_cast<double>(cd.origin_x) + 3.5) * preview_scale_x));
        const std::int64_t display_y = static_cast<std::int64_t>(std::floor(
            (static_cast<double>(cd.origin_y) + 3.5) * preview_scale_y));
        const std::int64_t cell_x0 = display_x - half_stride_x;
        const std::int64_t cell_y0 = display_y - half_stride_y;

        for (std::size_t dy = 0; dy < display_stride_y; ++dy) {
            const std::int64_t py = cell_y0 + static_cast<std::int64_t>(dy);
            if (py < 0 || py >= static_cast<std::int64_t>(source_image.height)) continue;
            for (std::size_t dx = 0; dx < display_stride_x; ++dx) {
                const std::int64_t px = cell_x0 + static_cast<std::int64_t>(dx);
                if (px < 0 || px >= static_cast<std::int64_t>(source_image.width)) continue;

                const std::size_t idx = (static_cast<std::size_t>(py) * source_image.width + static_cast<std::size_t>(px)) * 3;
                overlay.pixels[idx + 0] = static_cast<std::uint8_t>(0.5 * source_image.pixels[idx + 0] + 0.5 * color[0]);
                overlay.pixels[idx + 1] = static_cast<std::uint8_t>(0.5 * source_image.pixels[idx + 1] + 0.5 * color[1]);
                overlay.pixels[idx + 2] = static_cast<std::uint8_t>(0.5 * source_image.pixels[idx + 2] + 0.5 * color[2]);
            }
        }
    }
    save_png_image(output_dir / "overlay.png", overlay);

    // 8. When true GeoMetadata is present, export GeoTIFFs: class_map.tif, confidence.tif, margin.tif
    if (result.metadata.has_geo) {
        // class_map.tif (uint8 class codes: 0..N-1, 254=UNCERTAIN, 255=NODATA)
        std::vector<std::uint8_t> geo_classes(result.total_decisions);
        std::vector<float> geo_conf(result.total_decisions);
        std::vector<float> geo_margin(result.total_decisions);

        for (std::size_t i = 0; i < result.total_decisions; ++i) {
            const auto& cd = result.compact_decisions[i];
            geo_classes[i] = cd.is_uncertain ? 254 : static_cast<std::uint8_t>(cd.predicted_index);
            geo_conf[i] = cd.probability;
            geo_margin[i] = cd.margin;
        }

        write_geotiff_raster(output_dir / "class_map.tif", geo_classes.data(), geo_classes.size(),
                             result.grid_width, result.grid_height, GeoRasterType::byte,
                             result.metadata, result.config.stride);
        write_geotiff_raster(output_dir / "confidence.tif", geo_conf.data(), geo_conf.size() * sizeof(float),
                             result.grid_width, result.grid_height, GeoRasterType::float32,
                             result.metadata, result.config.stride);
        write_geotiff_raster(output_dir / "margin.tif", geo_margin.data(), geo_margin.size() * sizeof(float),
                             result.grid_width, result.grid_height, GeoRasterType::float32,
                             result.metadata, result.config.stride);

        if (h3_aggregation_available()) {
            const auto aggregates = aggregate_dense_run_h3(
                result, 9, output_dir.filename().string(), model_sha);
            export_h3_aggregation_csv(aggregates, model.class_names,
                                      output_dir / "classification_h3.csv");
        }
    }

    // 9. Export provenance from the actual model, source, run and artifacts.
    const std::string run_id = "run:" + output_dir.filename().string();
    const std::string model_id = "model:" + (model_sha.empty() ? model_path.filename().string() : model_sha);
    const std::string source_id = "source:" + (source_sha.empty() ? source_path.filename().string() : source_sha);
    ProvenanceGraph provenance;
    provenance.add_node({source_id, NodeType::SOURCE, source_path.filename().string(), source_sha,
                         {{"path", source_path.string()}, {"crs", result.metadata.has_geo ? result.metadata.crs : "NOT_AVAILABLE"}}});
    provenance.add_node({model_id, NodeType::MODEL, model_path.filename().string(), model_sha,
                         {{"input_modality", modality_to_string(model.schema.modality)},
                          {"input_channels", std::to_string(model.schema.channels)}}});
    provenance.add_node({run_id, NodeType::CLASSIFICATION_RUN, output_dir.filename().string(), "",
                         {{"stride", std::to_string(result.config.stride)},
                          {"decision_count", std::to_string(result.total_decisions)}}});
    provenance.add_edge({run_id, model_id, EdgeType::EVALUATED_ON, timestamp_buf, {}});
    provenance.add_edge({run_id, source_id, EdgeType::DERIVED_FROM, timestamp_buf, {}});

    const std::vector<std::string> artifact_names{
        "classification.csv", "decisions.bin", "run.json", "class_map.png",
        "confidence.png", "margin.png", "overlay.png", "class_map.tif",
        "confidence.tif", "margin.tif", "classification_h3.csv"};
    for (const auto& artifact_name : artifact_names) {
        const auto artifact_path = output_dir / artifact_name;
        if (!std::filesystem::is_regular_file(artifact_path)) continue;
        const std::string artifact_id = "artifact:" + output_dir.filename().string() + ":" + artifact_name;
        provenance.add_node({artifact_id, NodeType::ARTIFACT, artifact_name,
                             compute_file_sha256(artifact_path), {{"path", artifact_path.string()}}});
        provenance.add_edge({artifact_id, run_id, EdgeType::PRODUCED_BY, timestamp_buf, {}});
    }
    provenance.export_json(output_dir / "provenance.json");
    provenance.export_jsonl(output_dir / "evidence.jsonl");
}

} // namespace tinyvision
