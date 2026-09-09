# TinyLogicVision

TinyLogicVision is a deliberately tiny RGB vision-learning experiment written
from scratch in C++23.

TV-00 contains:

- fixed-size 8x8 RGB inputs;
- one explicit hidden layer using `tanh`;
- softmax output;
- cross-entropy loss;
- manual backward pass;
- SGD on CPU;
- deterministic synthetic spatial-pattern dataset;
- numerical gradient checking;
- deterministic training gate.

No ML/tensor framework, pretrained model, image library, SisTer integration,
OBCE code, geospatial assumption, or web frontend is used in TV-00.

## Build and verify

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
./build/tinyvision
```

Expected TV-00 gate:

- `gradient_test`: PASS
- `training_test`: PASS with >= 98% training accuracy
- canonical current run: 100% training accuracy

TV-00 only proves that the explicit learning machinery is mathematically
consistent and can fit controlled RGB spatial patterns. It does not establish
generalization to natural images.

## TV-01A — Generalization Observatory

TV-01A preserves the TV-00 architecture, model seed, learning rate, epoch count,
optimizer and training dataset. It adds a deterministic held-out validation
trajectory with stronger RGB/noise variation and spatial shifts, plus explicit
metrics for probability, margin, confusion, learning events and forgetting.

Run the observational witness with:

```bash
./build/tinyvision_generalization
```

The test split is deliberately not evaluated in TV-01A. Validation may inform a
later REA successor experiment; the sealed test may not be used for model or
procedure selection. See `docs/TV-01A-generalization-observatory.md`.
