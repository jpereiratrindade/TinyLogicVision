#include "tinyvision/tensor.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tinyvision {

double MultichannelTensor::get_raw_value(std::size_t x, std::size_t y, std::size_t c) const {
    if (x >= width || y >= height || c >= channels) {
        throw std::out_of_range("tensor coordinate out of bounds");
    }

    std::size_t elem_idx = 0;
    if (layout == DataLayout::INTERLEAVED) {
        elem_idx = (y * width + x) * channels + c;
    } else {
        elem_idx = c * (width * height) + (y * width + x);
    }

    const std::size_t byte_offset = elem_idx * element_size();
    if (byte_offset + element_size() > data.size()) {
        throw std::runtime_error("tensor byte buffer overflow");
    }

    const std::uint8_t* ptr = data.data() + byte_offset;
    switch (dtype) {
        case DataType::UINT8: {
            return static_cast<double>(*ptr);
        }
        case DataType::UINT16: {
            std::uint16_t val = 0;
            std::memcpy(&val, ptr, 2);
            return static_cast<double>(val);
        }
        case DataType::FLOAT32: {
            float val = 0.0f;
            std::memcpy(&val, ptr, 4);
            return static_cast<double>(val);
        }
        case DataType::FLOAT64: {
            double val = 0.0;
            std::memcpy(&val, ptr, 8);
            return val;
        }
    }
    return 0.0;
}

void MultichannelTensor::extract_normalized_patch_into(std::size_t origin_x,
                                                       std::size_t origin_y,
                                                       std::span<double> out_buf) const {
    const std::size_t expected_size = 8 * 8 * channels;
    if (out_buf.size() != expected_size) {
        throw std::invalid_argument("output span size mismatch for tensor patch");
    }
    if (origin_x + 8 > width || origin_y + 8 > height) {
        throw std::out_of_range("patch exceeds tensor boundaries");
    }

    for (std::size_t wy = 0; wy < 8; ++wy) {
        const std::size_t y = origin_y + wy;
        for (std::size_t wx = 0; wx < 8; ++wx) {
            const std::size_t x = origin_x + wx;
            const std::size_t patch_px_idx = (wy * 8 + wx) * channels;

            for (std::size_t c = 0; c < channels; ++c) {
                const double raw_val = get_raw_value(x, y, c);
                double normalized = raw_val;

                if (c < schema.channel_specs.size()) {
                    const auto& norm = schema.channel_specs[c].normalization;
                    if (norm.type == NormalizationType::UINT8_DIV_255) {
                        normalized = raw_val / 255.0;
                    } else if (norm.type == NormalizationType::LINEAR) {
                        normalized = raw_val * norm.scale + norm.offset;
                    }
                }
                out_buf[patch_px_idx + c] = normalized;
            }
        }
    }
}

void MultichannelTensor::save_tvp(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open file for writing: " + path.string());

    // 64-byte Header
    char header[64]{0};
    std::memcpy(header, "TVP\0", 4);
    const std::uint32_t version = 1;
    const std::uint32_t w = static_cast<std::uint32_t>(width);
    const std::uint32_t h = static_cast<std::uint32_t>(height);
    const std::uint32_t ch = static_cast<std::uint32_t>(channels);
    const std::uint32_t dt = static_cast<std::uint32_t>(dtype);
    const std::uint32_t lay = static_cast<std::uint32_t>(layout);
    const std::uint64_t payload_bytes = data.size();

    std::memcpy(header + 4, &version, 4);
    std::memcpy(header + 8, &w, 4);
    std::memcpy(header + 12, &h, 4);
    std::memcpy(header + 16, &ch, 4);
    std::memcpy(header + 20, &dt, 4);
    std::memcpy(header + 24, &lay, 4);
    std::memcpy(header + 28, &payload_bytes, 8);

    out.write(header, 64);
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
}

MultichannelTensor MultichannelTensor::load_tvp(const std::filesystem::path& path, const InputSchema& schema) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path.string());

    char header[64]{0};
    in.read(header, 64);
    if (std::memcmp(header, "TVP\0", 4) != 0) {
        throw std::runtime_error("invalid TVP magic in " + path.string());
    }

    std::uint32_t version = 0;
    std::uint32_t w = 0, h = 0, ch = 0, dt = 0, lay = 0;
    std::uint64_t payload_bytes = 0;

    std::memcpy(&version, header + 4, 4);
    std::memcpy(&w, header + 8, 4);
    std::memcpy(&h, header + 12, 4);
    std::memcpy(&ch, header + 16, 4);
    std::memcpy(&dt, header + 20, 4);
    std::memcpy(&lay, header + 24, 4);
    std::memcpy(&payload_bytes, header + 28, 8);

    if (version != 1) {
        throw std::runtime_error("unsupported TVP version " + std::to_string(version));
    }

    MultichannelTensor tensor;
    tensor.width = w;
    tensor.height = h;
    tensor.channels = ch;
    tensor.dtype = static_cast<DataType>(dt);
    tensor.layout = static_cast<DataLayout>(lay);
    tensor.schema = schema;

    const std::size_t expected_bytes = tensor.total_bytes();
    if (payload_bytes != expected_bytes) {
        throw std::runtime_error("corrupt TVP payload size");
    }

    tensor.data.resize(expected_bytes);
    in.read(reinterpret_cast<char*>(tensor.data.data()), expected_bytes);
    return tensor;
}

} // namespace tinyvision
