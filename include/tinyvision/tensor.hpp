#pragma once

#include "tinyvision/schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace tinyvision {

enum class DataType {
    UINT8,
    UINT16,
    FLOAT32,
    FLOAT64
};

inline std::string data_type_to_string(DataType dt) {
    switch (dt) {
        case DataType::UINT8: return "UINT8";
        case DataType::UINT16: return "UINT16";
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::FLOAT64: return "FLOAT64";
    }
    return "UINT8";
}

inline DataType string_to_data_type(const std::string& str) {
    if (str == "UINT16") return DataType::UINT16;
    if (str == "FLOAT32") return DataType::FLOAT32;
    if (str == "FLOAT64") return DataType::FLOAT64;
    return DataType::UINT8;
}

struct MultichannelTensor {
    std::size_t width{0};
    std::size_t height{0};
    std::size_t channels{0};
    DataType dtype{DataType::UINT8};
    DataLayout layout{DataLayout::INTERLEAVED};
    InputSchema schema;
    std::vector<std::uint8_t> data;

    std::size_t element_size() const {
        switch (dtype) {
            case DataType::UINT8: return 1;
            case DataType::UINT16: return 2;
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT64: return 8;
        }
        return 1;
    }

    std::size_t total_elements() const noexcept { return width * height * channels; }
    std::size_t total_bytes() const noexcept { return total_elements() * element_size(); }

    double get_raw_value(std::size_t x, std::size_t y, std::size_t c) const;
    void extract_normalized_patch_into(std::size_t origin_x, std::size_t origin_y, std::span<double> out_buf) const;

    void save_tvp(const std::filesystem::path& path) const;
    static MultichannelTensor load_tvp(const std::filesystem::path& path, const InputSchema& schema = {});
};

} // namespace tinyvision
