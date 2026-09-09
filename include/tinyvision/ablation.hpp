#pragma once

#include "tinyvision/application.hpp"
#include "tinyvision/schema.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tinyvision {

enum class SpectralMask {
    NONE,
    MASK_B2,
    MASK_B3,
    MASK_B4,
    MASK_B8
};

inline std::string spectral_mask_to_string(SpectralMask m) {
    switch (m) {
        case SpectralMask::NONE: return "FULL_4BAND";
        case SpectralMask::MASK_B2: return "MASK_B2";
        case SpectralMask::MASK_B3: return "MASK_B3";
        case SpectralMask::MASK_B4: return "MASK_B4";
        case SpectralMask::MASK_B8: return "MASK_B8";
    }
    return "NONE";
}

// Applies neutral masking to a 256-element patch (8x8x4, interleaved) without altering input dimensions
inline void apply_spectral_mask(std::span<double> patch_256, SpectralMask mask, double neutral_value = 0.0) {
    if (mask == SpectralMask::NONE) return;
    if (patch_256.size() != 256) return;

    std::size_t target_channel = 0;
    if (mask == SpectralMask::MASK_B2) target_channel = 0;
    else if (mask == SpectralMask::MASK_B3) target_channel = 1;
    else if (mask == SpectralMask::MASK_B4) target_channel = 2;
    else if (mask == SpectralMask::MASK_B8) target_channel = 3;

    for (std::size_t p = 0; p < 64; ++p) {
        patch_256[p * 4 + target_channel] = neutral_value;
    }
}

} // namespace tinyvision
