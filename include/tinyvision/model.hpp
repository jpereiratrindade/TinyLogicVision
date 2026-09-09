#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tinyvision {

struct MLPWorkspace {
    std::vector<double> hidden;
    std::vector<double> logits;
    std::vector<double> probabilities;
    std::vector<double> dz2;
    std::vector<double> gw2;
    std::vector<double> gb2;
    std::vector<double> dh;
    std::vector<double> dz1;
    std::vector<double> gw1;
    std::vector<double> gb1;

    void ensure_dimensions(std::size_t input_size, std::size_t hidden_size, std::size_t output_size);
};

using InferenceWorkspace = MLPWorkspace;
using TrainingWorkspace = MLPWorkspace;

class MLP {
public:
    MLP(std::size_t input_size,
        std::size_t hidden_size,
        std::size_t output_size,
        std::uint32_t seed = 7);

    // Existing vector-based APIs (preserving backward compatibility)
    std::vector<double> predict(const std::vector<double>& input) const;
    std::size_t predict_class(const std::vector<double>& input) const;
    double loss(const std::vector<double>& input, std::size_t label) const;
    double train_one(const std::vector<double>& input, std::size_t label, double learning_rate);
    std::vector<double> gradient(const std::vector<double>& input, std::size_t label) const;

    // Optimized span & workspace APIs (zero heap allocation in hot loops after warmup)
    void forward_into(std::span<const double> input, MLPWorkspace& ws) const;
    void predict_into(std::span<const double> input, std::span<double> output_probs, MLPWorkspace& ws) const;
    std::size_t predict_class(std::span<const double> input, MLPWorkspace& ws) const;
    double loss(std::span<const double> input, std::size_t label, MLPWorkspace& ws) const;
    void gradient_into(std::span<const double> input, std::size_t label, std::span<double> out_grad, MLPWorkspace& ws) const;
    double train_one_inplace(std::span<const double> input, std::size_t label, double learning_rate, MLPWorkspace& ws);

    std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    void set_parameters(std::span<const double> values);

    std::size_t input_size() const noexcept { return input_size_; }
    std::size_t hidden_size() const noexcept { return hidden_size_; }
    std::size_t output_size() const noexcept { return output_size_; }
    std::size_t parameter_count() const noexcept {
        return w1_.size() + b1_.size() + w2_.size() + b2_.size();
    }

private:
    struct Cache {
        std::vector<double> hidden;
        std::vector<double> logits;
        std::vector<double> probabilities;
    };

    Cache forward(const std::vector<double>& input) const;
    void validate_input(std::span<const double> input) const;

    std::size_t input_size_;
    std::size_t hidden_size_;
    std::size_t output_size_;
    std::vector<double> w1_;
    std::vector<double> b1_;
    std::vector<double> w2_;
    std::vector<double> b2_;
};

} // namespace tinyvision
