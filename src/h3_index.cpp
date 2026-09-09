#include "tinyvision/h3_index.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef TINYVISION_WITH_H3
#include <h3/h3api.h>
#endif

#ifdef TINYVISION_WITH_GDAL
#include <ogr_spatialref.h>
#endif

namespace tinyvision {

bool h3_available() noexcept {
#ifdef TINYVISION_WITH_H3
    return true;
#else
    return false;
#endif
}

bool h3_aggregation_available() noexcept {
#if defined(TINYVISION_WITH_H3) && defined(TINYVISION_WITH_GDAL)
    return true;
#else
    return false;
#endif
}

std::string latlon_to_h3_index(double lat_deg, double lon_deg, int resolution) {
#ifdef TINYVISION_WITH_H3
    if (!std::isfinite(lat_deg) || !std::isfinite(lon_deg) ||
        lat_deg < -90.0 || lat_deg > 90.0 || lon_deg < -180.0 || lon_deg > 180.0) {
        throw std::invalid_argument("H3 coordinates must be finite WGS84 latitude/longitude degrees");
    }
    if (resolution < 0 || resolution > 15) {
        throw std::invalid_argument("H3 resolution must be in [0, 15]");
    }
    const LatLng coordinate{degsToRads(lat_deg), degsToRads(lon_deg)};
    H3Index cell = H3_NULL;
    if (latLngToCell(&coordinate, resolution, &cell) != E_SUCCESS || !isValidCell(cell)) {
        throw std::runtime_error("official H3 library failed to index WGS84 coordinate");
    }
    char text[17]{}; // 64-bit H3 index: at most 16 hexadecimal digits plus NUL.
    if (h3ToString(cell, text, sizeof(text)) != E_SUCCESS) {
        throw std::runtime_error("official H3 library failed to format cell index");
    }
    return text;
#else
    (void)lat_deg;
    (void)lon_deg;
    (void)resolution;
    throw std::runtime_error("H3 support is unavailable; rebuild with the official H3 library");
#endif
}

std::vector<H3CellAggregate> aggregate_dense_run_h3(const DenseMapResult& result,
                                                    int resolution,
                                                    const std::string& source_run,
                                                    const std::string& model_hash) {
#if !defined(TINYVISION_WITH_H3) || !defined(TINYVISION_WITH_GDAL)
    (void)result;
    (void)resolution;
    (void)source_run;
    (void)model_hash;
    throw std::runtime_error("H3 aggregation requires official H3 and GDAL support");
#else
    if (!result.metadata.has_geo || result.metadata.crs.empty()) {
        throw std::invalid_argument("H3 aggregation requires a georeferenced source CRS");
    }

    OGRSpatialReference source_reference;
    OGRSpatialReference wgs84_reference;
    source_reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    wgs84_reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    if (source_reference.SetFromUserInput(result.metadata.crs.c_str()) != OGRERR_NONE ||
        wgs84_reference.SetWellKnownGeogCS("WGS84") != OGRERR_NONE) {
        throw std::invalid_argument("cannot parse source CRS for H3 aggregation: " + result.metadata.crs);
    }
    auto* transformation = OGRCreateCoordinateTransformation(&source_reference, &wgs84_reference);
    if (transformation == nullptr) {
        throw std::runtime_error("cannot transform source CRS to WGS84 for H3 aggregation");
    }

    struct Accumulator {
        std::size_t total{0};
        std::size_t classified{0};
        std::size_t uncertain{0};
        std::map<std::string, std::size_t> class_counts;
        double sum_top1{0.0};
        double sum_margin{0.0};
    };

    std::map<std::string, Accumulator> cell_acc;

    try {
        for (const auto& d : result.decisions) {
            double lon = d.map_x;
            double lat = d.map_y;
            if (!transformation->Transform(1, &lon, &lat)) {
                throw std::runtime_error("GDAL failed to transform a decision point to WGS84");
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
        OCTDestroyCoordinateTransformation(transformation);
    } catch (...) {
        OCTDestroyCoordinateTransformation(transformation);
        throw;
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
#endif
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
