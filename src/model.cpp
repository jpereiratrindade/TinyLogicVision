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

void MLPWorkspace::ensure_dimensions(std::size_t input_size, std::size_t hidden_size, std::size_t output_size) {
    if (hidden.size() != hidden_size) hidden.assign(hidden_size, 0.0);
    if (logits.size() != output_size) logits.assign(output_size, 0.0);
    if (probabilities.size() != output_size) probabilities.assign(output_size, 0.0);
    if (dz2.size() != output_size) dz2.assign(output_size, 0.0);
    if (gw2.size() != output_size * hidden_size) gw2.assign(output_size * hidden_size, 0.0);
    if (gb2.size() != output_size) gb2.assign(output_size, 0.0);
    if (dh.size() != hidden_size) dh.assign(hidden_size, 0.0);
    if (dz1.size() != hidden_size) dz1.assign(hidden_size, 0.0);
    if (gw1.size() != hidden_size * input_size) gw1.assign(hidden_size * input_size, 0.0);
    if (gb1.size() != hidden_size) gb1.assign(hidden_size, 0.0);
}

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

void MLP::validate_input(std::span<const double> input) const {
    if (input.size() != input_size_) throw std::invalid_argument("input size mismatch");
}

void MLP::forward_into(std::span<const double> input, MLPWorkspace& ws) const {
    validate_input(input);
    ws.ensure_dimensions(input_size_, hidden_size_, output_size_);

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        double z = b1_[h];
        const std::size_t offset = h * input_size_;
        for (std::size_t i = 0; i < input_size_; ++i) z += w1_[offset + i] * input[i];
        ws.hidden[h] = std::tanh(z);
    }

    for (std::size_t o = 0; o < output_size_; ++o) {
        double z = b2_[o];
        const std::size_t offset = o * hidden_size_;
        for (std::size_t h = 0; h < hidden_size_; ++h) z += w2_[offset + h] * ws.hidden[h];
        ws.logits[o] = z;
    }

    const double max_logit = *std::max_element(ws.logits.begin(), ws.logits.end());
    double sum = 0.0;
    for (std::size_t o = 0; o < output_size_; ++o) {
        ws.probabilities[o] = std::exp(ws.logits[o] - max_logit);
        sum += ws.probabilities[o];
    }
    for (std::size_t o = 0; o < output_size_; ++o) {
        ws.probabilities[o] /= sum;
    }
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

void MLP::predict_into(std::span<const double> input, std::span<double> output_probs, MLPWorkspace& ws) const {
    if (output_probs.size() != output_size_) {
        throw std::invalid_argument("output span size mismatch");
    }
    forward_into(input, ws);
    std::copy_n(ws.probabilities.data(), output_size_, output_probs.data());
}

std::size_t MLP::predict_class(std::span<const double> input, MLPWorkspace& ws) const {
    forward_into(input, ws);
    return static_cast<std::size_t>(std::distance(
        ws.probabilities.begin(), std::max_element(ws.probabilities.begin(), ws.probabilities.end())));
}

double MLP::loss(std::span<const double> input, std::size_t label, MLPWorkspace& ws) const {
    if (label >= output_size_) throw std::invalid_argument("label out of range");
    forward_into(input, ws);
    return -std::log(clamp_probability(ws.probabilities[label]));
}

void MLP::gradient_into(std::span<const double> input, std::size_t label, std::span<double> out_grad, MLPWorkspace& ws) const {
    if (label >= output_size_) throw std::invalid_argument("label out of range");
    if (out_grad.size() != parameter_count()) throw std::invalid_argument("gradient buffer size mismatch");
    forward_into(input, ws);

    for (std::size_t o = 0; o < output_size_; ++o) {
        ws.dz2[o] = ws.probabilities[o];
    }
    ws.dz2[label] -= 1.0;
    ws.gb2 = ws.dz2;

    std::fill(ws.dh.begin(), ws.dh.end(), 0.0);
    for (std::size_t o = 0; o < output_size_; ++o) {
        const std::size_t offset = o * hidden_size_;
        for (std::size_t h = 0; h < hidden_size_; ++h) {
            ws.gw2[offset + h] = ws.dz2[o] * ws.hidden[h];
            ws.dh[h] += w2_[offset + h] * ws.dz2[o];
        }
    }

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        ws.dz1[h] = ws.dh[h] * (1.0 - ws.hidden[h] * ws.hidden[h]);
    }
    ws.gb1 = ws.dz1;

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        const std::size_t offset = h * input_size_;
        for (std::size_t i = 0; i < input_size_; ++i) {
            ws.gw1[offset + i] = ws.dz1[h] * input[i];
        }
    }

    std::size_t dst_idx = 0;
    std::copy_n(ws.gw1.data(), ws.gw1.size(), out_grad.data() + dst_idx);
    dst_idx += ws.gw1.size();
    std::copy_n(ws.gb1.data(), ws.gb1.size(), out_grad.data() + dst_idx);
    dst_idx += ws.gb1.size();
    std::copy_n(ws.gw2.data(), ws.gw2.size(), out_grad.data() + dst_idx);
    dst_idx += ws.gw2.size();
    std::copy_n(ws.gb2.data(), ws.gb2.size(), out_grad.data() + dst_idx);
}

double MLP::train_one_inplace(std::span<const double> input, std::size_t label, double learning_rate, MLPWorkspace& ws) {
    if (!(learning_rate > 0.0) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("learning rate must be finite and positive");
    }
    if (label >= output_size_) throw std::invalid_argument("label out of range");

    forward_into(input, ws);
    const double current_loss = -std::log(clamp_probability(ws.probabilities[label]));

    for (std::size_t o = 0; o < output_size_; ++o) {
        ws.dz2[o] = ws.probabilities[o];
    }
    ws.dz2[label] -= 1.0;
    ws.gb2 = ws.dz2;

    std::fill(ws.dh.begin(), ws.dh.end(), 0.0);
    for (std::size_t o = 0; o < output_size_; ++o) {
        const std::size_t offset = o * hidden_size_;
        for (std::size_t h = 0; h < hidden_size_; ++h) {
            ws.gw2[offset + h] = ws.dz2[o] * ws.hidden[h];
            ws.dh[h] += w2_[offset + h] * ws.dz2[o];
        }
    }

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        ws.dz1[h] = ws.dh[h] * (1.0 - ws.hidden[h] * ws.hidden[h]);
    }
    ws.gb1 = ws.dz1;

    for (std::size_t h = 0; h < hidden_size_; ++h) {
        const std::size_t offset = h * input_size_;
        for (std::size_t i = 0; i < input_size_; ++i) {
            ws.gw1[offset + i] = ws.dz1[h] * input[i];
        }
    }

    for (std::size_t i = 0; i < w1_.size(); ++i) w1_[i] -= learning_rate * ws.gw1[i];
    for (std::size_t i = 0; i < b1_.size(); ++i) b1_[i] -= learning_rate * ws.gb1[i];
    for (std::size_t i = 0; i < w2_.size(); ++i) w2_[i] -= learning_rate * ws.gw2[i];
    for (std::size_t i = 0; i < b2_.size(); ++i) b2_[i] -= learning_rate * ws.gb2[i];

    return current_loss;
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
    set_parameters(std::span<const double>(values));
}

void MLP::set_parameters(std::span<const double> values) {
    const std::size_t expected = w1_.size() + b1_.size() + w2_.size() + b2_.size();
    if (values.size() != expected) throw std::invalid_argument("parameter size mismatch");
    std::size_t offset = 0;
    std::copy_n(values.data() + offset, w1_.size(), w1_.data());
    offset += w1_.size();
    std::copy_n(values.data() + offset, b1_.size(), b1_.data());
    offset += b1_.size();
    std::copy_n(values.data() + offset, w2_.size(), w2_.data());
    offset += w2_.size();
    std::copy_n(values.data() + offset, b2_.size(), b2_.data());
}

} // namespace tinyvision
