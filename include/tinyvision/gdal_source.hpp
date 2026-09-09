#pragma once

#include "tinyvision/geo.hpp"

#include <filesystem>
#include <memory>

namespace tinyvision {

class GdalMultibandSource final : public InputSource {
public:
    GdalMultibandSource(const std::filesystem::path& b2,
                        const std::filesystem::path& b3,
                        const std::filesystem::path& b4,
                        const std::filesystem::path& b8);
    ~GdalMultibandSource() override;

    GdalMultibandSource(const GdalMultibandSource&) = delete;
    GdalMultibandSource& operator=(const GdalMultibandSource&) = delete;

    static bool available() noexcept;

    std::size_t width() const override;
    std::size_t height() const override;
    std::size_t channels() const override;
    const InputSchema& schema() const override;
    const GeoMetadata& spatial_metadata() const override;
    void read_window_into(std::size_t origin_x,
                          std::size_t origin_y,
                          std::span<double> out_buf) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tinyvision
