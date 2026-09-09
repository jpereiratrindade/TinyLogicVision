#include "tinyvision/ablation.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running ablation_protocol_test...\n";

    // 1. Create a 256-element mock 4-channel patch (8x8x4)
    std::vector<double> base_patch(256);
    for (std::size_t p = 0; p < 64; ++p) {
        base_patch[p * 4 + 0] = 0.1; // B2
        base_patch[p * 4 + 1] = 0.2; // B3
        base_patch[p * 4 + 2] = 0.3; // B4
        base_patch[p * 4 + 3] = 0.8; // B8
    }

    // 2. Test MASK_B8
    auto patch_mask_b8 = base_patch;
    tinyvision::apply_spectral_mask(patch_mask_b8, tinyvision::SpectralMask::MASK_B8, 0.0);

    for (std::size_t p = 0; p < 64; ++p) {
        if (patch_mask_b8[p * 4 + 0] != 0.1 ||
            patch_mask_b8[p * 4 + 1] != 0.2 ||
            patch_mask_b8[p * 4 + 2] != 0.3) {
            fail("MASK_B8 altered active channels (B2, B3, B4)");
        }
        if (patch_mask_b8[p * 4 + 3] != 0.0) {
            fail("MASK_B8 did not neutralize B8 to 0.0");
        }
    }

    // 3. Test MASK_B2
    auto patch_mask_b2 = base_patch;
    tinyvision::apply_spectral_mask(patch_mask_b2, tinyvision::SpectralMask::MASK_B2, 0.0);

    for (std::size_t p = 0; p < 64; ++p) {
        if (patch_mask_b2[p * 4 + 0] != 0.0) {
            fail("MASK_B2 did not neutralize B2 to 0.0");
        }
        if (patch_mask_b2[p * 4 + 1] != 0.2 ||
            patch_mask_b2[p * 4 + 2] != 0.3 ||
            patch_mask_b2[p * 4 + 3] != 0.8) {
            fail("MASK_B2 altered active channels (B3, B4, B8)");
        }
    }

    // Vector size remains unchanged (256)
    if (patch_mask_b8.size() != 256 || patch_mask_b2.size() != 256) {
        fail("Spectral masking altered input dimensions");
    }

    std::cout << "PASS: ablation_protocol_test (neutral masking preserves 256 inputs & parameter capacity)\n";
    return 0;
}
