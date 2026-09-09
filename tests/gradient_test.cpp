#include "tinyvision/model.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    tinyvision::MLP model(6, 4, 3, 5);
    const std::vector<double> input = {0.10, 0.90, 0.25, 0.75, 0.40, 0.60};
    constexpr std::size_t label = 1;
    constexpr double epsilon = 1e-6;

    const auto analytic = model.gradient(input, label);
    auto params = model.parameters();
    double max_relative_error = 0.0;

    for (std::size_t i = 0; i < params.size(); ++i) {
        auto plus = params;
        auto minus = params;
        plus[i] += epsilon;
        minus[i] -= epsilon;

        model.set_parameters(plus);
        const double loss_plus = model.loss(input, label);
        model.set_parameters(minus);
        const double loss_minus = model.loss(input, label);

        const double numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        const double denom = std::max({1.0, std::abs(numeric), std::abs(analytic[i])});
        max_relative_error = std::max(max_relative_error, std::abs(numeric - analytic[i]) / denom);
    }
    model.set_parameters(params);

    std::cout << "max_relative_error=" << max_relative_error << '\n';
    return max_relative_error < 1e-6 ? 0 : 1;
}
