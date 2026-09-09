#include "tinyvision/dense.hpp"
#include "tinyvision/image.hpp"

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
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tinyvision {
namespace {

// Self-contained SHA-256 for provenance hashing without extra library dependencies
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

DenseMapResult classify_dense(const ApplicationModel& model,
                              const RgbImage& image,
                              const DenseMapConfig& config) {
    if (config.stride != 1 && config.stride != 2 && config.stride != 4 && config.stride != 8) {
        throw std::invalid_argument("stride must be 1, 2, 4, or 8");
    }
    if (image.width < 8 || image.height < 8) {
        throw std::invalid_argument("image must be at least 8x8 pixels");
    }
    if (image.pixels.size() != image.width * image.height * 3) {
        throw std::invalid_argument("invalid RGB image byte count");
    }

    const std::size_t grid_width = (image.width - 8) / config.stride + 1;
    const std::size_t grid_height = (image.height - 8) / config.stride + 1;
    const std::size_t total_decisions = grid_width * grid_height;

    DenseMapResult result;
    result.source_width = image.width;
    result.source_height = image.height;
    result.grid_width = grid_width;
    result.grid_height = grid_height;
    result.total_decisions = total_decisions;
    result.config = config;
    result.palette = get_canonical_palette(model.class_names);
    result.decisions.resize(total_decisions);
    result.compact_decisions.resize(total_decisions);

    std::size_t num_threads = config.threads;
    if (num_threads == 0) {
        num_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    num_threads = std::min<std::size_t>(num_threads, grid_height);
    result.thread_count = num_threads;
    result.implementation_mode = "TILED_STREAMING";

    const std::size_t num_classes = model.class_names.size();

    auto process_rows = [&](std::size_t gy_start, std::size_t gy_end) {
        MLPWorkspace ws;
        std::array<double, 192> patch_buf{};

        for (std::size_t gy = gy_start; gy < gy_end; ++gy) {
            const std::size_t y = gy * config.stride;
            const std::size_t row_offset = gy * grid_width;

            for (std::size_t gx = 0; gx < grid_width; ++gx) {
                const std::size_t x = gx * config.stride;
                const std::size_t dec_idx = row_offset + gx;

                // Extract without heap allocation
                extract_rgb_input_vector_into(image, x, y, patch_buf);

                // Run forward into reusable workspace
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

                // 1. Fill compact decision
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

                // 2. Fill standard DenseDecision for backwards compatibility
                auto& d = result.decisions[dec_idx];
                d.grid_x = gx;
                d.grid_y = gy;
                d.origin_x = x;
                d.origin_y = y;
                d.center_x = static_cast<double>(x) + 3.5;
                d.center_y = static_cast<double>(y) + 3.5;
                d.display_x = x + 4;
                d.display_y = y + 4;
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

    // Count totals deterministically
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

void export_dense_map(const DenseMapResult& result,
                      const ApplicationModel& model,
                      const RgbImage& source_image,
                      const std::filesystem::path& model_path,
                      const std::filesystem::path& source_path,
                      const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);

    // 1. Export classification.csv
    const auto csv_path = output_dir / "classification.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        throw std::runtime_error("failed to open " + csv_path.string() + " for writing");
    }
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

    // 2. Export decisions.bin for O(1) indexed lookup
    const auto bin_path = output_dir / "decisions.bin";
    std::ofstream binf(bin_path, std::ios::binary | std::ios::trunc);
    if (binf) {
        // 64-byte Header
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

        // Fixed records
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
       << "  \"run_id\": \"" << output_dir.filename().string() << "\",\n"
       << "  \"timestamp\": \"" << timestamp_buf << "\",\n"
       << "  \"model_path\": \"" << model_path.string() << "\",\n"
       << "  \"model_sha256\": \"" << model_sha << "\",\n"
       << "  \"source_image\": \"" << source_path.filename().string() << "\",\n"
       << "  \"source_sha256\": \"" << source_sha << "\",\n"
       << "  \"source_width\": " << result.source_width << ",\n"
       << "  \"source_height\": " << result.source_height << ",\n"
       << "  \"patch_width\": 8,\n"
       << "  \"patch_height\": 8,\n"
       << "  \"stride\": " << result.config.stride << ",\n"
       << "  \"grid_width\": " << result.grid_width << ",\n"
       << "  \"grid_height\": " << result.grid_height << ",\n"
       << "  \"decision_count\": " << result.total_decisions << ",\n"
       << "  \"classes\": [";
    for (std::size_t i = 0; i < model.class_names.size(); ++i) {
        js << "\"" << model.class_names[i] << "\"" << (i + 1 < model.class_names.size() ? ", " : "");
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
       << "    \"implementation_mode\": \"" << result.implementation_mode << "\",\n"
       << "    \"thread_count\": " << result.thread_count << ",\n"
       << "    \"tile_dimensions\": [" << result.config.tile_width << ", " << result.config.tile_height << "],\n"
       << "    \"decision_record_version\": 1,\n"
       << "    \"indexed_binary_available\": true\n"
       << "  },\n"
       << "  \"confidence_threshold\": " << std::fixed << std::setprecision(4) << result.config.confidence_threshold << ",\n"
       << "  \"margin_threshold\": " << result.config.margin_threshold << ",\n"
       << "  \"sentinel_nominal_10m\": " << (result.config.sentinel_nominal_10m ? "true" : "false") << ",\n"
       << "  \"operator_declared_nominal_10m\": " << (result.config.sentinel_nominal_10m ? "true" : "false") << ",\n"
       << "  \"resolution_status\": \"" << (result.config.sentinel_nominal_10m ? "Operator declared nominal 10 m/pixel (unverified metadata)" : "Display image space (no metric scale declared)") << "\",\n";
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
        js << "      {\"name\": \"" << c.name << "\", \"index\": " << c.index
           << ", \"color\": \"" << c.hex_color << "\", \"rgb\": ["
           << static_cast<int>(c.rgb[0]) << ", " << static_cast<int>(c.rgb[1]) << ", " << static_cast<int>(c.rgb[2]) << "]}"
           << (i + 1 < result.palette.classes.size() ? ",\n" : "\n");
    }
    js << "    ],\n"
       << "    \"uncertain\": {\"name\": \"" << result.palette.uncertain.name << "\", \"color\": \"" << result.palette.uncertain.hex_color
       << "\", \"rgb\": [" << static_cast<int>(result.palette.uncertain.rgb[0]) << ", "
       << static_cast<int>(result.palette.uncertain.rgb[1]) << ", "
       << static_cast<int>(result.palette.uncertain.rgb[2]) << "]}\n"
       << "  },\n"
       << "  \"geospatial\": {\n"
       << "    \"available\": false,\n"
       << "    \"crs\": null,\n"
       << "    \"transform\": null,\n"
       << "    \"h3\": null\n"
       << "  },\n"
       << "  \"summary\": {\n"
       << "    \"total_decisions\": " << result.total_decisions << ",\n"
       << "    \"classified_count\": " << result.classified_count << ",\n"
       << "    \"uncertain_count\": " << result.uncertain_count << ",\n"
       << "    \"class_counts\": {\n";
    std::size_t c_idx = 0;
    for (const auto& [name, count] : class_counts) {
        js << "      \"" << name << "\": " << count << (++c_idx < class_counts.size() ? ",\n" : "\n");
    }
    js << "    }\n"
       << "  }\n"
       << "}\n";

    // 4. Export class_map.png (GRID SPACE: grid_width x grid_height)
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

    // 5. Export confidence.png (GRID SPACE: Top-1 Probability map)
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

    // 6. Export margin.png (GRID SPACE: Top-1 minus Top-2 Margin map)
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

    // 7. Export overlay.png (SOURCE IMAGE SPACE: width x height, single authoritative C++ projection centered on decisions)
    RgbImage overlay = source_image;
    const std::size_t stride = result.config.stride;
    const std::int64_t half_stride = static_cast<std::int64_t>(stride / 2);

    for (const auto& cd : result.compact_decisions) {
        const auto color = get_decision_color(result.palette, cd.predicted_index, cd.is_uncertain);
        const std::int64_t display_x = static_cast<std::int64_t>(cd.origin_x + 4);
        const std::int64_t display_y = static_cast<std::int64_t>(cd.origin_y + 4);
        const std::int64_t cell_x0 = display_x - half_stride;
        const std::int64_t cell_y0 = display_y - half_stride;

        for (std::size_t dy = 0; dy < stride; ++dy) {
            const std::int64_t py = cell_y0 + static_cast<std::int64_t>(dy);
            if (py < 0 || py >= static_cast<std::int64_t>(source_image.height)) continue;
            for (std::size_t dx = 0; dx < stride; ++dx) {
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
}

} // namespace tinyvision
