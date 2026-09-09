#include "tinyvision/application.hpp"
#include "tinyvision/dense.hpp"
#include "tinyvision/geo.hpp"
#include "tinyvision/h3_index.hpp"
#include "tinyvision/image.hpp"

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

} // namespace

int main() {
    std::cout << "Running h3_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_h3_test";
    std::filesystem::create_directories(tmp_dir);

    // 1. Test deterministic hex index calculation
    const std::string idx1 = tinyvision::latlon_to_h3_index(-23.5505, -46.6333, 9);
    const std::string idx2 = tinyvision::latlon_to_h3_index(-23.5505, -46.6333, 9);
    if (idx1 != idx2 || idx1.empty()) {
        fail("latlon_to_h3_index is not deterministic");
    }

    // 2. Test cell split assignment (1 cell = exactly 1 split)
    std::vector<std::string> cells;
    for (int i = 0; i < 20; ++i) {
        cells.push_back(tinyvision::latlon_to_h3_index(-23.0 + i * 0.05, -46.0 + i * 0.05, 8));
    }
    const auto split_map = tinyvision::assign_h3_cell_splits(cells, 0.6, 0.2, 7);
    if (split_map.size() != cells.size()) {
        fail("Split map cell count mismatch");
    }
    std::size_t train_cnt = 0, dev_cnt = 0, probe_cnt = 0;
    for (const auto& [cell, sp] : split_map) {
        if (sp == tinyvision::H3SplitType::TRAIN) train_cnt++;
        else if (sp == tinyvision::H3SplitType::DEV) dev_cnt++;
        else if (sp == tinyvision::H3SplitType::PROBE) probe_cnt++;
    }
    if (train_cnt + dev_cnt + probe_cnt != cells.size() || train_cnt == 0) {
        fail("Split partition failed to assign valid disjoint subsets");
    }

    // 3. Test Dense Classification Aggregation by H3
    std::vector<std::string> classes{"floresta", "campo", "solo"};
    tinyvision::MLP mlp(192, 24, 3, 42);
    tinyvision::ApplicationModel model(classes, 8, 8, mlp);

    tinyvision::RgbImage img;
    img.width = 32;
    img.height = 32;
    img.pixels.resize(32 * 32 * 3, 150);

    tinyvision::GeoMetadata meta;
    meta.has_geo = true;
    meta.crs = "EPSG:32722";
    meta.geotransform = {500000.0, 10.0, 0.0, 7500000.0, 0.0, -10.0};
    meta.pixel_size_x = 10.0;
    meta.pixel_size_y = 10.0;

    tinyvision::DenseMapConfig cfg;
    cfg.stride = 2;
    cfg.confidence_threshold = 0.5;

    const auto res = tinyvision::classify_dense(model, img, cfg, meta);
    const auto aggregates = tinyvision::aggregate_dense_run_h3(res, 9, "test_run_01", "hash_model_01");

    if (aggregates.empty()) {
        fail("H3 aggregation produced 0 cells");
    }

    std::size_t total_agg_decisions = 0;
    std::size_t total_agg_classified = 0;
    std::size_t total_agg_uncertain = 0;
    for (const auto& a : aggregates) {
        total_agg_decisions += a.number_of_decisions;
        total_agg_classified += a.classified_count;
        total_agg_uncertain += a.uncertain_count;
        if (a.number_of_decisions != a.classified_count + a.uncertain_count) {
            fail("Cell decision count != classified + uncertain");
        }
    }

    if (total_agg_decisions != res.total_decisions) {
        fail("Aggregated decision sum (" + std::to_string(total_agg_decisions) +
             ") != result total decisions (" + std::to_string(res.total_decisions) + ")");
    }
    if (total_agg_classified != res.classified_count || total_agg_uncertain != res.uncertain_count) {
        fail("Aggregated classified/uncertain totals mismatch");
    }

    // 4. Test CSV Export
    const auto h3_csv = tmp_dir / "classification_h3.csv";
    tinyvision::export_h3_aggregation_csv(aggregates, classes, h3_csv);

    if (!std::filesystem::exists(h3_csv) || std::filesystem::file_size(h3_csv) == 0) {
        fail("classification_h3.csv was not created");
    }

    std::ifstream csv_f(h3_csv);
    std::string header;
    std::getline(csv_f, header);
    if (header.find("h3_index,resolution,number_of_decisions") == std::string::npos ||
        header.find("class_count_floresta") == std::string::npos ||
        header.find("dominant_class") == std::string::npos) {
        fail("classification_h3.csv header format mismatch");
    }

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: h3_test (deterministic hex indexing, cell split integrity, aggregation counts, classification_h3.csv)\n";
    return 0;
}
