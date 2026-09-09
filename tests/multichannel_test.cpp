#include "tinyvision/application.hpp"
#include "tinyvision/schema.hpp"
#include "tinyvision/tensor.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running multichannel_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_multichannel_test";
    std::filesystem::create_directories(tmp_dir);

    // 1. Create a 4-channel UINT16 Tensor (e.g., Sentinel-2 10m B2, B3, B4, B8)
    const auto s2_schema = tinyvision::InputSchema::create_sentinel2_10m(8, 8);
    tinyvision::MultichannelTensor tensor;
    tensor.width = 16;
    tensor.height = 16;
    tensor.channels = 4;
    tensor.dtype = tinyvision::DataType::UINT16;
    tensor.layout = tinyvision::DataLayout::INTERLEAVED;
    tensor.schema = s2_schema;
    tensor.data.resize(tensor.total_bytes());

    auto* u16_ptr = reinterpret_cast<std::uint16_t*>(tensor.data.data());
    for (std::size_t i = 0; i < tensor.total_elements(); ++i) {
        u16_ptr[i] = static_cast<std::uint16_t>((i * 100) % 10000); // 0..10000 reflectance scaled
    }

    // 2. Extract 8x8 patch into normalized vector
    std::vector<double> patch_buf(256);
    tensor.extract_normalized_patch_into(2, 3, patch_buf);

    // Verify first pixel normalized value (scale = 0.0001)
    const std::size_t elem0 = (3 * 16 + 2) * 4 + 0;
    const double expected_val0 = static_cast<double>(u16_ptr[elem0]) * 0.0001;
    if (std::abs(patch_buf[0] - expected_val0) > 1e-9) {
        fail("Patch normalization mismatch: got " + std::to_string(patch_buf[0]) + " expected " + std::to_string(expected_val0));
    }

    // 3. Save and load TVP patch
    const auto tvp_path = tmp_dir / "patch.tvp";
    tensor.save_tvp(tvp_path);

    const auto loaded_tvp = tinyvision::MultichannelTensor::load_tvp(tvp_path, s2_schema);
    if (loaded_tvp.width != 16 || loaded_tvp.height != 16 || loaded_tvp.channels != 4) {
        fail("Loaded TVP dimensions mismatch");
    }
    if (loaded_tvp.data != tensor.data) {
        fail("Loaded TVP byte data mismatch");
    }

    // 4. Train a 256-input model on synthetic 4-channel dataset
    tinyvision::ApplicationSplit train_split;
    train_split.class_names = {"agua", "vegetacao", "solo"};
    train_split.schema = s2_schema;

    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t sample_idx = 0; sample_idx < 10; ++sample_idx) {
            std::vector<double> sample_input(256);
            for (std::size_t i = 0; i < 256; ++i) {
                sample_input[i] = 0.1 * (c + 1) + 0.01 * (i % 5);
            }
            train_split.samples.push_back({sample_input, c, "synthetic"});
        }
    }

    tinyvision::ApplicationTrainingConfig cfg;
    cfg.hidden_size = 24;
    cfg.epochs = 20;
    cfg.learning_rate = 0.05;

    const auto res = tinyvision::train_application(train_split, train_split, cfg);
    if (res.model.schema.channels != 4 || res.model.network.input_size() != 256) {
        fail("Trained model schema mismatch");
    }
    if (res.final_train.accuracy < 0.9) {
        fail("4-channel synthetic training failed to achieve expected convergence");
    }

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: multichannel_test (4-channel tensor, TVP roundtrip, 256-input model training)\n";
    return 0;
}
