#include "tinyvision/application.hpp"
#include "tinyvision/schema.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running schema_model_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_schema_test";
    std::filesystem::create_directories(tmp_dir);

    // 1. Create and save a v1 format model manually
    const auto v1_path = tmp_dir / "test_model_v1.tlv";
    {
        std::ofstream out(v1_path);
        out << "TINYLOGICVISION_APP_MODEL 1\n"
            << "8 8 16 3\n"
            << "3\n"
            << "\"campo\"\n\"floresta\"\n\"solo\"\n";
        const std::size_t param_count = 16 * 8 * 8 * 3 + 16 + 3 * 16 + 3;
        out << param_count << '\n';
        for (std::size_t i = 0; i < param_count; ++i) {
            out << (static_cast<double>(i % 31) / 31.0 - 0.5) << '\n';
        }
    }

    // Load v1 model
    const auto loaded_v1 = tinyvision::load_application_model(v1_path);
    if (loaded_v1.class_names.size() != 3 || loaded_v1.class_names[0] != "campo") {
        fail("v1 model class names mismatch");
    }
    if (loaded_v1.schema.channels != 3 || loaded_v1.schema.modality != tinyvision::Modality::RGB) {
        fail("v1 model schema modality mismatch");
    }
    if (loaded_v1.network.input_size() != 192) {
        fail("v1 model input size mismatch");
    }

    // 2. Save as v2 model and test exact roundtrip
    const auto v2_path = tmp_dir / "test_model_v2.tlv";
    tinyvision::save_application_model(loaded_v1, v2_path);

    const auto loaded_v2 = tinyvision::load_application_model(v2_path);
    if (loaded_v2.class_names != loaded_v1.class_names) {
        fail("v2 roundtrip class names mismatch");
    }
    if (loaded_v2.schema != loaded_v1.schema) {
        fail("v2 roundtrip schema mismatch");
    }
    const auto p1 = loaded_v1.network.parameters();
    const auto p2 = loaded_v2.network.parameters();
    if (p1.size() != p2.size()) {
        fail("v2 roundtrip parameter size mismatch");
    }
    for (std::size_t i = 0; i < p1.size(); ++i) {
        if (std::abs(p1[i] - p2[i]) > 1e-15) {
            fail("v2 roundtrip parameter value mismatch at " + std::to_string(i));
        }
    }

    // 3. Test Sentinel-2 4-band schema model (B2, B3, B4, B8 -> 256 inputs)
    const auto s2_schema = tinyvision::InputSchema::create_sentinel2_10m(8, 8);
    if (s2_schema.channels != 4 || s2_schema.input_size() != 256) {
        fail("Sentinel-2 schema input size mismatch: got " + std::to_string(s2_schema.input_size()));
    }
    tinyvision::MLP s2_mlp(256, 24, 3, 42);
    tinyvision::ApplicationModel s2_model({"campo", "floresta", "solo"}, s2_schema, std::move(s2_mlp));

    const auto s2_path = tmp_dir / "sentinel2_model.tlv";
    tinyvision::save_application_model(s2_model, s2_path);

    const auto loaded_s2 = tinyvision::load_application_model(s2_path);
    if (loaded_s2.schema != s2_schema) {
        fail("Sentinel-2 loaded schema mismatch");
    }
    if (loaded_s2.network.input_size() != 256) {
        fail("Sentinel-2 loaded network input size mismatch");
    }

    // 4. Schema equality & rejection tests
    auto altered_schema = s2_schema;
    altered_schema.channel_specs[0].id = "B4"; // swap band order
    if (s2_schema == altered_schema) {
        fail("Schema comparison failed to detect channel order change");
    }

    auto altered_norm = s2_schema;
    altered_norm.channel_specs[0].normalization.scale = 0.0002;
    if (s2_schema == altered_norm) {
        fail("Schema comparison failed to detect normalization change");
    }

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: schema_model_test (v1 compatibility + v2 roundtrip + schema rejection)\n";
    return 0;
}
