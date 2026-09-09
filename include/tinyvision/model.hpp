#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyvision {

class MLP {
public:
    MLP(std::size_t input_size,
        std::size_t hidden_size,
        std::size_t output_size,
        std::uint32_t seed = 7);

    std::vector<double> predict(const std::vector<double>& input) const;
    std::size_t predict_class(const std::vector<double>& input) const;
    double loss(const std::vector<double>& input, std::size_t label) const;
    double train_one(const std::vector<double>& input, std::size_t label, double learning_rate);

    std::vector<double> gradient(const std::vector<double>& input, std::size_t label) const;
    std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);

    std::size_t input_size() const noexcept { return input_size_; }
    std::size_t hidden_size() const noexcept { return hidden_size_; }
    std::size_t output_size() const noexcept { return output_size_; }

private:
    struct Cache {
        std::vector<double> hidden;
        std::vector<double> logits;
        std::vector<double> probabilities;
    };

    Cache forward(const std::vector<double>& input) const;
    void validate_input(const std::vector<double>& input) const;

    std::size_t input_size_;
    std::size_t hidden_size_;
    std::size_t output_size_;
    std::vector<double> w1_;
    std::vector<double> b1_;
    std::vector<double> w2_;
    std::vector<double> b2_;
};

} // namespace tinyvision
