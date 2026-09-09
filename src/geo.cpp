#include "tinyvision/geo.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace tinyvision {

void validate_multiband_alignment(const std::vector<GeoMetadata>& band_metas) {
    if (band_metas.empty()) {
        throw std::invalid_argument("no band metadata provided for alignment check");
    }

    const auto& base = band_metas[0];
    for (std::size_t i = 1; i < band_metas.size(); ++i) {
        const auto& other = band_metas[i];

        if (base.raster_width != other.raster_width || base.raster_height != other.raster_height) {
            std::ostringstream ss;
            ss << "multiband dimension mismatch between band 0 (" << base.raster_width << 'x' << base.raster_height
               << ") and band " << i << " (" << other.raster_width << 'x' << other.raster_height << ")";
            throw std::invalid_argument(ss.str());
        }

        if (base.has_geo && other.has_geo) {
            if (base.crs != other.crs) {
                std::ostringstream ss;
                ss << "multiband CRS mismatch between band 0 (" << base.crs << ") and band " << i << " (" << other.crs << ")";
                throw std::invalid_argument(ss.str());
            }

            for (std::size_t gt_idx = 0; gt_idx < 6; ++gt_idx) {
                if (std::abs(base.geotransform[gt_idx] - other.geotransform[gt_idx]) > 1e-6) {
                    std::ostringstream ss;
                    ss << "multiband geotransform mismatch at index " << gt_idx << " between band 0 and band " << i;
                    throw std::invalid_argument(ss.str());
                }
            }
        }
    }
}

} // namespace tinyvision
