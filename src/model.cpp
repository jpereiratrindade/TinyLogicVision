#include "tinyvision/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tinyvision {
namespace {

double clamp_probability(double value) {
    return std::clamp(value, 1e-12, 1.0);
}

} // namespace

MLP::MLP(std::size_t input_size,
         std::size_t hidden_size,
         std::size_t output_size,
         std::uint32_t seed)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      w1_(hidden_size * input_size),
      b1_(hidden_size, 0.0),
      w2_(output_size * hidden_size),
      b2_(output_size, 0.0) {
    if (input_size == 0 || hidden_size == 0 || output_size < 2) {
        throw std::invalid_argument("invalid MLP dimensions");
    }

    std::mt19937 rng(seed);
    const double limit1 = std::sqrt(6.0 / static_cast<double>(input_size + hidden_size));
    const double limit2 = std::sqrt(6.0 / static_cast<double>(hidden_size + output_size));
    std::uniform_real_distribution<double> dist1(-limit1, limit1);
    std::uniform_real_distribution<double> dist2(-limit2, limit2);
    for (double& value : w1_) value = dist1(rng);
    for (double& value : w2_) value = dist2(rng);
}

void MLP::validate_input(const std::vector<double>& input) const {
    if (input.size() != input_size_) throw std::invalid_argument("input size mismatch");
}

MLP::Cache MLP::forward(const std::vector<double>& input) const {
    validate_input(input);
    Cache cache;
    cache.hidden.assign(hidden_size_, 0.0);
    cache.logits.assign(output_size_, 0.0);

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        double z = b1_[h];
        const std::size_t offset = h * input_size_;
        for (std::size_t i = 0; i < input_size_; ++i) z += w1_[offset + i] * input[i];
        cache.hidden[h] = std::tanh(z);
    }

    for (std::size_t o = 0; o < output_size_; ++o) {
        double z = b2_[o];
        const std::size_t offset = o * hidden_size_;
        for (std::size_t h = 0; h < hidden_size_; ++h) z += w2_[offset + h] * cache.hidden[h];
        cache.logits[o] = z;
    }

    const double max_logit = *std::max_element(cache.logits.begin(), cache.logits.end());
    cache.probabilities.resize(output_size_);
    double sum = 0.0;
    for (std::size_t o = 0; o < output_size_; ++o) {
        cache.probabilities[o] = std::exp(cache.logits[o] - max_logit);
        sum += cache.probabilities[o];
    }
    for (double& value : cache.probabilities) value /= sum;
    return cache;
}

std::vector<double> MLP::predict(const std::vector<double>& input) const {
    return forward(input).probabilities;
}

std::size_t MLP::predict_class(const std::vector<double>& input) const {
    const auto probabilities = predict(input);
    return static_cast<std::size_t>(std::distance(
        probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
}

double MLP::loss(const std::vector<double>& input, std::size_t label) const {
    if (label >= output_size_) throw std::invalid_argument("label out of range");
    const auto cache = forward(input);
    return -std::log(clamp_probability(cache.probabilities[label]));
}

std::vector<double> MLP::gradient(const std::vector<double>& input, std::size_t label) const {
    if (label >= output_size_) throw std::invalid_argument("label out of range");
    const auto cache = forward(input);

    std::vector<double> dz2 = cache.probabilities;
    dz2[label] -= 1.0;

    std::vector<double> gw2(w2_.size(), 0.0);
    std::vector<double> gb2 = dz2;
    std::vector<double> dh(hidden_size_, 0.0);

    for (std::size_t o = 0; o < output_size_; ++o) {
        const std::size_t offset = o * hidden_size_;
        for (std::size_t h = 0; h < hidden_size_; ++h) {
            gw2[offset + h] = dz2[o] * cache.hidden[h];
            dh[h] += w2_[offset + h] * dz2[o];
        }
    }

    std::vector<double> dz1(hidden_size_, 0.0);
    for (std::size_t h = 0; h < hidden_size_; ++h) {
        dz1[h] = dh[h] * (1.0 - cache.hidden[h] * cache.hidden[h]);
    }

    std::vector<double> gw1(w1_.size(), 0.0);
    std::vector<double> gb1 = dz1;
    for (std::size_t h = 0; h < hidden_size_; ++h) {
        const std::size_t offset = h * input_size_;
        for (std::size_t i = 0; i < input_size_; ++i) {
            gw1[offset + i] = dz1[h] * input[i];
        }
    }

    std::vector<double> all;
    all.reserve(gw1.size() + gb1.size() + gw2.size() + gb2.size());
    all.insert(all.end(), gw1.begin(), gw1.end());
    all.insert(all.end(), gb1.begin(), gb1.end());
    all.insert(all.end(), gw2.begin(), gw2.end());
    all.insert(all.end(), gb2.begin(), gb2.end());
    return all;
}

double MLP::train_one(const std::vector<double>& input, std::size_t label, double learning_rate) {
    if (!(learning_rate > 0.0) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("learning rate must be finite and positive");
    }
    const double current_loss = loss(input, label);
    const auto grad = gradient(input, label);
    auto params = parameters();
    for (std::size_t i = 0; i < params.size(); ++i) params[i] -= learning_rate * grad[i];
    set_parameters(params);
    return current_loss;
}

std::vector<double> MLP::parameters() const {
    std::vector<double> all;
    all.reserve(w1_.size() + b1_.size() + w2_.size() + b2_.size());
    all.insert(all.end(), w1_.begin(), w1_.end());
    all.insert(all.end(), b1_.begin(), b1_.end());
    all.insert(all.end(), w2_.begin(), w2_.end());
    all.insert(all.end(), b2_.begin(), b2_.end());
    return all;
}

void MLP::set_parameters(const std::vector<double>& values) {
    const std::size_t expected = w1_.size() + b1_.size() + w2_.size() + b2_.size();
    if (values.size() != expected) throw std::invalid_argument("parameter size mismatch");
    std::size_t offset = 0;
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), w1_.size(), w1_.begin());
    offset += w1_.size();
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), b1_.size(), b1_.begin());
    offset += b1_.size();
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), w2_.size(), w2_.begin());
    offset += w2_.size();
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(offset), b2_.size(), b2_.begin());
}

} // namespace tinyvision
