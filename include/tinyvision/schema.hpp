#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinyvision {

enum class Modality {
    RGB,
    MULTICHANNEL,
    SENTINEL2_MULTIBAND
};

inline std::string modality_to_string(Modality m) {
    switch (m) {
        case Modality::RGB: return "RGB";
        case Modality::MULTICHANNEL: return "MULTICHANNEL";
        case Modality::SENTINEL2_MULTIBAND: return "SENTINEL2_MULTIBAND";
    }
    return "UNKNOWN";
}

inline Modality string_to_modality(const std::string& str) {
    if (str == "RGB") return Modality::RGB;
    if (str == "MULTICHANNEL") return Modality::MULTICHANNEL;
    if (str == "SENTINEL2_MULTIBAND") return Modality::SENTINEL2_MULTIBAND;
    return Modality::MULTICHANNEL;
}

enum class NormalizationType {
    UINT8_DIV_255,
    LINEAR
};

inline std::string normalization_to_string(NormalizationType n) {
    switch (n) {
        case NormalizationType::UINT8_DIV_255: return "uint8_div_255";
        case NormalizationType::LINEAR: return "linear";
    }
    return "unknown";
}

inline NormalizationType string_to_normalization(const std::string& str) {
    if (str == "uint8_div_255") return NormalizationType::UINT8_DIV_255;
    if (str == "linear") return NormalizationType::LINEAR;
    return NormalizationType::LINEAR;
}

enum class DataLayout {
    INTERLEAVED,
    PLANAR
};

inline std::string layout_to_string(DataLayout l) {
    switch (l) {
        case DataLayout::INTERLEAVED: return "INTERLEAVED";
        case DataLayout::PLANAR: return "PLANAR";
    }
    return "INTERLEAVED";
}

inline DataLayout string_to_layout(const std::string& str) {
    if (str == "PLANAR") return DataLayout::PLANAR;
    return DataLayout::INTERLEAVED;
}

struct NormalizationSpec {
    NormalizationType type{NormalizationType::UINT8_DIV_255};
    double scale{1.0 / 255.0};
    double offset{0.0};

    bool operator==(const NormalizationSpec& other) const = default;
};

struct ChannelSpec {
    std::string id;       // e.g. "R", "G", "B", "B2", "B3", "B4", "B8"
    std::string name;     // e.g. "red", "green", "blue", "nir_10m"
    std::string role;     // e.g. "COLOR_RED", "SPECTRAL_NIR"
    std::string unit;     // e.g. "DN", "reflectance_scaled"
    NormalizationSpec normalization;

    bool operator==(const ChannelSpec& other) const = default;
};

struct InputSchema {
    std::size_t width{8};
    std::size_t height{8};
    std::size_t channels{3};
    DataLayout layout{DataLayout::INTERLEAVED};
    Modality modality{Modality::RGB};
    std::vector<ChannelSpec> channel_specs;

    std::size_t input_size() const noexcept {
        return width * height * channels;
    }

    bool operator==(const InputSchema& other) const = default;

    static InputSchema create_canonical_rgb(std::size_t width = 8, std::size_t height = 8) {
        InputSchema schema;
        schema.width = width;
        schema.height = height;
        schema.channels = 3;
        schema.layout = DataLayout::INTERLEAVED;
        schema.modality = Modality::RGB;
        schema.channel_specs = {
            {"R", "red", "COLOR_RED", "DN", {NormalizationType::UINT8_DIV_255, 1.0 / 255.0, 0.0}},
            {"G", "green", "COLOR_GREEN", "DN", {NormalizationType::UINT8_DIV_255, 1.0 / 255.0, 0.0}},
            {"B", "blue", "COLOR_BLUE", "DN", {NormalizationType::UINT8_DIV_255, 1.0 / 255.0, 0.0}}
        };
        return schema;
    }

    static InputSchema create_sentinel2_10m(std::size_t width = 8, std::size_t height = 8) {
        InputSchema schema;
        schema.width = width;
        schema.height = height;
        schema.channels = 4;
        schema.layout = DataLayout::INTERLEAVED;
        schema.modality = Modality::SENTINEL2_MULTIBAND;
        schema.channel_specs = {
            {"B2", "blue_10m", "SPECTRAL_OPTICAL", "reflectance_scaled", {NormalizationType::LINEAR, 0.0001, 0.0}},
            {"B3", "green_10m", "SPECTRAL_OPTICAL", "reflectance_scaled", {NormalizationType::LINEAR, 0.0001, 0.0}},
            {"B4", "red_10m", "SPECTRAL_OPTICAL", "reflectance_scaled", {NormalizationType::LINEAR, 0.0001, 0.0}},
            {"B8", "nir_10m", "SPECTRAL_NIR", "reflectance_scaled", {NormalizationType::LINEAR, 0.0001, 0.0}}
        };
        return schema;
    }
};

} // namespace tinyvision
