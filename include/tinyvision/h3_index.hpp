#pragma once

#include "tinyvision/dense.hpp"
#include "tinyvision/geo.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tinyvision {

// Computes an official H3 cell index for WGS84 latitude/longitude (degrees).
std::string latlon_to_h3_index(double lat_deg, double lon_deg, int resolution);
bool h3_available() noexcept;
bool h3_aggregation_available() noexcept;

struct H3CellAggregate {
    std::string h3_index;
    int resolution{9};
    std::size_t number_of_decisions{0};
    std::size_t classified_count{0};
    std::size_t uncertain_count{0};
    std::map<std::string, std::size_t> class_counts;
    std::map<std::string, double> class_proportions;
    double mean_top1_probability{0.0};
    double mean_margin{0.0};
    std::string dominant_class;
    std::string source_run;
    std::string model_hash;
};

std::vector<H3CellAggregate> aggregate_dense_run_h3(const DenseMapResult& result,
                                                    int resolution = 9,
                                                    const std::string& source_run = "",
                                                    const std::string& model_hash = "");

void export_h3_aggregation_csv(const std::vector<H3CellAggregate>& aggregates,
                               const std::vector<std::string>& class_names,
                               const std::filesystem::path& path);

enum class H3SplitType {
    TRAIN,
    DEV,
    PROBE
};

struct H3SplitAssignment {
    std::string h3_index;
    H3SplitType split{H3SplitType::TRAIN};
};

std::map<std::string, H3SplitType> assign_h3_cell_splits(const std::vector<std::string>& h3_cells,
                                                         double train_ratio = 0.7,
                                                         double dev_ratio = 0.15,
                                                         std::uint32_t seed = 42);

} // namespace tinyvision
