#include "tinyvision/model.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

namespace {

static thread_local bool g_track_allocations = false;
static thread_local std::size_t g_allocation_count = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

} // namespace

void* operator new(std::size_t size) {
    if (g_track_allocations) {
        ++g_allocation_count;
    }
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

int main() {
    std::cout << "Running core_perf_test...\n";

    constexpr std::size_t kInputSize = 192;
    constexpr std::size_t kHiddenSize = 24;
    constexpr std::size_t kOutputSize = 3;

    tinyvision::MLP model1(kInputSize, kHiddenSize, kOutputSize, 42);
    tinyvision::MLP model2(kInputSize, kHiddenSize, kOutputSize, 42);

    // Verify initial parameters are identical
    const auto params1 = model1.parameters();
    const auto params2 = model2.parameters();
    if (params1 != params2) {
        fail("Initial parameters mismatch");
    }

    std::vector<double> input(kInputSize);
    for (std::size_t i = 0; i < kInputSize; ++i) {
        input[i] = static_cast<double>(i % 17) / 17.0;
    }

    // 1. Equivalence of predict vs predict_into / forward_into
    tinyvision::MLPWorkspace ws;
    const auto pred_legacy = model1.predict(input);
    std::vector<double> pred_opt(kOutputSize, 0.0);
    model1.predict_into(input, pred_opt, ws);

    for (std::size_t i = 0; i < kOutputSize; ++i) {
        if (std::abs(pred_legacy[i] - pred_opt[i]) > 1e-15) {
            fail("predict vs predict_into mismatch at index " + std::to_string(i));
        }
    }

    // 2. Equivalence of gradient vs gradient_into
    const auto grad_legacy = model1.gradient(input, 1);
    std::vector<double> grad_opt(model1.parameter_count(), 0.0);
    model1.gradient_into(input, 1, grad_opt, ws);

    if (grad_legacy.size() != grad_opt.size()) {
        fail("gradient size mismatch");
    }
    for (std::size_t i = 0; i < grad_legacy.size(); ++i) {
        if (std::abs(grad_legacy[i] - grad_opt[i]) > 1e-15) {
            fail("gradient vs gradient_into mismatch at index " + std::to_string(i));
        }
    }

    // 3. Equivalence of train_one vs train_one_inplace over multiple updates
    tinyvision::MLPWorkspace train_ws;
    for (std::size_t step = 0; step < 50; ++step) {
        const std::size_t label = step % kOutputSize;
        for (std::size_t i = 0; i < kInputSize; ++i) {
            input[i] = static_cast<double>((i + step) % 19) / 19.0;
        }

        const double loss1 = model1.train_one(input, label, 0.05);
        const double loss2 = model2.train_one_inplace(input, label, 0.05, train_ws);

        if (std::abs(loss1 - loss2) > 1e-14) {
            fail("Loss mismatch at step " + std::to_string(step));
        }
    }

    const auto final_params1 = model1.parameters();
    const auto final_params2 = model2.parameters();
    for (std::size_t i = 0; i < final_params1.size(); ++i) {
        if (std::abs(final_params1[i] - final_params2[i]) > 1e-14) {
            fail("Final parameters mismatch at parameter index " + std::to_string(i));
        }
    }

    // 4. Zero allocation witness in hot inference loop
    tinyvision::MLPWorkspace hot_ws;
    std::vector<double> out_buf(kOutputSize, 0.0);

    // Warm up workspace buffers
    model1.predict_into(input, out_buf, hot_ws);

    // Start tracking allocations
    g_allocation_count = 0;
    g_track_allocations = true;

    for (std::size_t iter = 0; iter < 1000; ++iter) {
        model1.predict_into(input, out_buf, hot_ws);
    }

    g_track_allocations = false;

    if (g_allocation_count != 0) {
        fail("Expected 0 allocations in hot inference loop, got " + std::to_string(g_allocation_count));
    }

    std::cout << "PASS: core_perf_test (exact math equivalence + 0 hot-loop heap allocations)\n";
    return 0;
}
