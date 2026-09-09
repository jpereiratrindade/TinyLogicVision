#include "tinyvision/h3_index.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace tinyvision {
namespace {

// Deterministic cell discretization for geospatial hexagonal indexing
std::uint64_t compute_hex_cell_id(double lat_deg, double lon_deg, int res) {
    res = std::clamp(res, 0, 15);
    // Base cell scale at resolution res: spacing ~ 360.0 / (2^res * 100)
    const double cell_scale = 100.0 * std::pow(1.5, static_cast<double>(res));
    const double x = (lon_deg + 180.0) * cell_scale;
    const double y = (lat_deg + 90.0) * cell_scale;

    // Hexagonal coordinate quantization (skewed axial coordinates)
    const double q = (std::sqrt(3.0) / 3.0 * x - 1.0 / 3.0 * y);
    const double r = (2.0 / 3.0 * y);

    const auto qi = static_cast<std::int64_t>(std::round(q));
    const auto ri = static_cast<std::int64_t>(std::round(r));

    // Construct 64-bit index: [mode(4 bits) | res(4 bits) | base_cell(8 bits) | coord_q(24 bits) | coord_r(24 bits)]
    const std::uint64_t mode = 1ULL; // H3 index mode
    const std::uint64_t res_bits = static_cast<std::uint64_t>(res) & 0x0fULL;
    const std::uint64_t q_bits = static_cast<std::uint64_t>(qi) & 0x00ffffffULL;
    const std::uint64_t r_bits = static_cast<std::uint64_t>(ri) & 0x00ffffffULL;

    return (mode << 59) | (res_bits << 52) | (q_bits << 24) | r_bits;
}

} // namespace

std::string latlon_to_h3_index(double lat_deg, double lon_deg, int resolution) {
    const std::uint64_t cell_id = compute_hex_cell_id(lat_deg, lon_deg, resolution);
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(15) << cell_id;
    return ss.str();
}

std::vector<H3CellAggregate> aggregate_dense_run_h3(const DenseMapResult& result,
                                                    int resolution,
                                                    const std::string& source_run,
                                                    const std::string& model_hash) {
    struct Accumulator {
        std::size_t total{0};
        std::size_t classified{0};
        std::size_t uncertain{0};
        std::map<std::string, std::size_t> class_counts;
        double sum_top1{0.0};
        double sum_margin{0.0};
    };

    std::map<std::string, Accumulator> cell_acc;

    for (const auto& d : result.decisions) {
        // Map decision center (map_x, map_y) to lat/lon approximation (or directly if geographic)
        // If CRS is projected (e.g. UTM meters), map_x/map_y are continuous coordinates
        double lat = d.map_y;
        double lon = d.map_x;

        if (result.metadata.has_geo) {
            // If UTM (e.g. northing ~ 7000000, easting ~ 500000), approximate latitude/longitude degrees
            if (std::abs(lat) > 360.0 || std::abs(lon) > 360.0) {
                lat = (d.map_y - 10000000.0) / 111320.0;
                lon = (d.map_x - 500000.0) / 111320.0;
            }
        } else {
            // Default continuous grid space
            lat = d.center_y;
            lon = d.center_x;
        }

        const std::string h3_cell = latlon_to_h3_index(lat, lon, resolution);
        auto& acc = cell_acc[h3_cell];
        acc.total++;
        if (d.is_uncertain) {
            acc.uncertain++;
        } else {
            acc.classified++;
            acc.class_counts[d.predicted_class]++;
        }
        acc.sum_top1 += d.probability;
        acc.sum_margin += d.margin;
    }

    std::vector<H3CellAggregate> aggregates;
    aggregates.reserve(cell_acc.size());

    for (const auto& [cell, acc] : cell_acc) {
        H3CellAggregate agg;
        agg.h3_index = cell;
        agg.resolution = resolution;
        agg.number_of_decisions = acc.total;
        agg.classified_count = acc.classified;
        agg.uncertain_count = acc.uncertain;
        agg.class_counts = acc.class_counts;
        agg.mean_top1_probability = acc.total > 0 ? (acc.sum_top1 / static_cast<double>(acc.total)) : 0.0;
        agg.mean_margin = acc.total > 0 ? (acc.sum_margin / static_cast<double>(acc.total)) : 0.0;
        agg.source_run = source_run;
        agg.model_hash = model_hash;

        std::string dom_class = "UNCERTAIN";
        std::size_t max_count = 0;
        for (const auto& [cname, cnt] : acc.class_counts) {
            const double prop = acc.total > 0 ? (static_cast<double>(cnt) / static_cast<double>(acc.total)) : 0.0;
            agg.class_proportions[cname] = prop;
            if (cnt > max_count) {
                max_count = cnt;
                dom_class = cname;
            }
        }
        agg.dominant_class = dom_class;
        aggregates.push_back(std::move(agg));
    }

    return aggregates;
}

void export_h3_aggregation_csv(const std::vector<H3CellAggregate>& aggregates,
                               const std::vector<std::string>& class_names,
                               const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open file for writing: " + path.string());

    // CSV Header: h3_index,resolution,number_of_decisions,classified_count,uncertain_count,class_count.*,class_proportion.*,mean_top1_probability,mean_margin,dominant_class,source_run,model_hash
    out << "h3_index,resolution,number_of_decisions,classified_count,uncertain_count";
    for (const auto& c : class_names) out << ",class_count_" << c;
    for (const auto& c : class_names) out << ",class_proportion_" << c;
    out << ",mean_top1_probability,mean_margin,dominant_class,source_run,model_hash\n";

    for (const auto& a : aggregates) {
        out << a.h3_index << ',' << a.resolution << ','
            << a.number_of_decisions << ',' << a.classified_count << ',' << a.uncertain_count;
        for (const auto& c : class_names) {
            const auto it = a.class_counts.find(c);
            out << ',' << (it != a.class_counts.end() ? it->second : 0);
        }
        for (const auto& c : class_names) {
            const auto it = a.class_proportions.find(c);
            out << ',' << std::fixed << std::setprecision(4) << (it != a.class_proportions.end() ? it->second : 0.0);
        }
        out << ',' << std::setprecision(4) << a.mean_top1_probability
            << ',' << a.mean_margin
            << ',' << a.dominant_class
            << ',' << a.source_run
            << ',' << a.model_hash << '\n';
    }
}

std::map<std::string, H3SplitType> assign_h3_cell_splits(const std::vector<std::string>& h3_cells,
                                                         double train_ratio,
                                                         double dev_ratio,
                                                         std::uint32_t seed) {
    std::vector<std::string> unique_cells = h3_cells;
    std::sort(unique_cells.begin(), unique_cells.end());
    unique_cells.erase(std::unique(unique_cells.begin(), unique_cells.end()), unique_cells.end());

    std::mt19937 rng(seed);
    std::shuffle(unique_cells.begin(), unique_cells.end(), rng);

    const std::size_t total = unique_cells.size();
    const std::size_t n_train = static_cast<std::size_t>(std::round(static_cast<double>(total) * train_ratio));
    const std::size_t n_dev = static_cast<std::size_t>(std::round(static_cast<double>(total) * dev_ratio));

    std::map<std::string, H3SplitType> split_map;
    for (std::size_t i = 0; i < total; ++i) {
        if (i < n_train) {
            split_map[unique_cells[i]] = H3SplitType::TRAIN;
        } else if (i < n_train + n_dev) {
            split_map[unique_cells[i]] = H3SplitType::DEV;
        } else {
            split_map[unique_cells[i]] = H3SplitType::PROBE;
        }
    }
    return split_map;
}

} // namespace tinyvision
