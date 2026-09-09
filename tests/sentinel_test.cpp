#include "tinyvision/sentinel.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

void touch_file(const std::filesystem::path& p) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p);
    f << "data";
}

} // namespace

int main() {
    std::cout << "Running sentinel_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_sentinel_test";
    std::filesystem::create_directories(tmp_dir);

    // 1. Create dummy SAFE product structure
    const auto safe_dir = tmp_dir / "S2A_MSIL2A_TEST.SAFE";
    const auto r10m_dir = safe_dir / "GRANULE" / "L2A_T22JCS" / "IMG_DATA" / "R10m";

    const auto b2_file = r10m_dir / "T22JCS_20240101_B02_10m.jp2";
    const auto b3_file = r10m_dir / "T22JCS_20240101_B03_10m.jp2";
    const auto b4_file = r10m_dir / "T22JCS_20240101_B04_10m.jp2";
    const auto b8_file = r10m_dir / "T22JCS_20240101_B08_10m.jp2";

    touch_file(b2_file);
    touch_file(b3_file);
    touch_file(b4_file);
    touch_file(b8_file);

    // 2. Discover SAFE product
    const auto prod = tinyvision::Sentinel2Product::discover_from_safe(safe_dir);
    if (prod.b2_path != b2_file || prod.b3_path != b3_file ||
        prod.b4_path != b4_file || prod.b8_path != b8_file) {
        fail("SAFE discovery matched incorrect paths");
    }
    if (prod.schema.channels != 4 || prod.schema.modality != tinyvision::Modality::SENTINEL2_MULTIBAND) {
        fail("SAFE product schema mismatch");
    }

    // 3. Test ambiguous rejection
    const auto b2_dup = r10m_dir / "T22JCS_20240101_copy_B02_10m.jp2";
    touch_file(b2_dup);
    bool threw_ambig = false;
    try {
        tinyvision::Sentinel2Product::discover_from_safe(safe_dir);
    } catch (const std::invalid_argument&) {
        threw_ambig = true;
    }
    if (!threw_ambig) {
        fail("Ambiguous bands in SAFE product were not rejected");
    }
    std::filesystem::remove(b2_dup);

    // 4. Test missing band rejection
    std::filesystem::remove(b8_file);
    bool threw_missing = false;
    try {
        tinyvision::Sentinel2Product::discover_from_safe(safe_dir);
    } catch (const std::invalid_argument&) {
        threw_missing = true;
    }
    if (!threw_missing) {
        fail("Missing band in SAFE product was not rejected");
    }

    // 5. Test RGB preview generation from 4-band tensor
    tinyvision::MultichannelTensor tensor;
    tensor.width = 4;
    tensor.height = 4;
    tensor.channels = 4;
    tensor.dtype = tinyvision::DataType::UINT16;
    tensor.layout = tinyvision::DataLayout::INTERLEAVED;
    tensor.schema = tinyvision::InputSchema::create_sentinel2_10m(8, 8);
    tensor.data.resize(tensor.total_bytes());

    auto* u16 = reinterpret_cast<std::uint16_t*>(tensor.data.data());
    // Pixel (0,0): B2=1000, B3=2000, B4=3000, B8=4000
    u16[0] = 1000; // B2 (Blue)
    u16[1] = 2000; // B3 (Green)
    u16[2] = 3000; // B4 (Red)
    u16[3] = 4000; // B8 (NIR)

    const auto preview = tinyvision::Sentinel2Product::generate_rgb_preview(tensor, 1.0);
    if (preview.width != 4 || preview.height != 4) {
        fail("Preview dimensions mismatch");
    }
    // Expected at (0,0):
    // R = 3000 * 0.0001 * 255.0 = 76.5 -> 76
    // G = 2000 * 0.0001 * 255.0 = 51.0 -> 51
    // B = 1000 * 0.0001 * 255.0 = 25.5 -> 25
    if (preview.pixels[0] != 76 || preview.pixels[1] != 51 || preview.pixels[2] != 25) {
        fail("Preview RGB channel mapping mismatch: got (" +
             std::to_string(preview.pixels[0]) + "," +
             std::to_string(preview.pixels[1]) + "," +
             std::to_string(preview.pixels[2]) + ")");
    }

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: sentinel_test (B2/B3/B4/B8 SAFE discovery, rejection gates, RGB preview)\n";
    return 0;
}
