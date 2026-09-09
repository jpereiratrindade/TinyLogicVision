# TinyLogicVision

TinyLogicVision is a deliberately tiny RGB vision-learning experiment written
from scratch in C++23. Its current synthetic baseline is complete, deterministic,
and independently verifiable from a clean clone.

## Current status

| Stage | Purpose | Status |
| --- | --- | --- |
| TV-00 | Explicit RGB learning core | PASS |
| TV-01A | Generalization observatory | PASS — validation failure observed |
| TV-01B | Spatial-coverage intervention | STRONG SUPPORT |
| TV-01C | Sealed synthetic TEST | NOT EXECUTED |

TV-00 contains fixed-size 8x8 RGB inputs, one 24-unit `tanh` hidden layer,
four softmax outputs, cross-entropy loss, a manual backward pass, and
per-sample SGD on CPU. The model has 4,732 trainable parameters. Numerical
gradient checking and deterministic training tests exercise the implementation.

No ML/tensor framework, pretrained model, image library, SisTer integration,
OBCE code, geospatial assumption, or web frontend is used.

## Canonical verification

After cloning, run the complete synthetic baseline verification with:

```bash
./scripts/verify.sh
```

The command configures and builds the Release/Ninja tree, runs all four tests,
and checks the TV-00, TV-01A, and TV-01B witnesses. It does not evaluate TEST.

The equivalent manual build commands are:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

## Evidence sequence

### TV-00 — Learning core

TV-00 established that the explicit learning machinery is numerically
consistent with the tested loss and can fit the controlled RGB spatial-pattern
training set. Its canonical trajectory reaches 100% training accuracy.

This is a fitting result, not evidence of generalization to unseen or natural
images.

### TV-01A — Generalization observatory

TV-01A preserved the TV-00 architecture and training trajectory, then observed
a deterministic held-out validation distribution with independent spatial
shifts, stronger RGB/noise variation, and brightness variation.

```text
final training accuracy      100.00%
final validation accuracy     46.88%
final validation loss          4.421994
```

The model fit TRAIN immediately while validation remained near 47% and became
increasingly confident in wrong predictions.

### TV-01B — Spatial-coverage intervention

TV-01B was preregistered before implementation. Its paired comparison preserved
the architecture, parameter count, seeds, optimizer, learning rate, epoch count,
sample count, RNG draws, presentation order, and validation distribution. It
changed only deterministic TRAIN spatial coverage.

```text
                              Control A    Intervention B
final training accuracy         100.00%          100.00%
final validation accuracy        46.88%          100.00%
final validation loss          4.421994         0.004919
minimum final margin          -0.999398         0.966333
final-30 validation minimum      46.88%          100.00%
```

Both trajectories reproduced exactly, including their complete 4,732-parameter
vectors. The result satisfied the preregistered `STRONG SUPPORT` boundary.

Under this fixed architecture, seed, synthetic family, and training trajectory,
broader spatial coverage was sufficient to eliminate the observed validation
failure.

VALIDATION remained frozen and never updated model weights, but it participated
in the formulation of the TV-01B intervention. It is therefore an experimental
development distribution. Independent confirmation remains reserved for the
sealed TEST split.

## Evidence limits

The current evidence does not establish:

- independent TEST generalization;
- natural-image recognition or remote-sensing performance;
- abstract translation, scale, or rotation invariance;
- robustness across model/training seeds or architectures;
- CNN superiority or SisTer integration.

TV-01C is the next question eligible for separate preregistration. TEST remains
sealed and has not been executed, directly or indirectly.

See [TV-01A](docs/TV-01A-generalization-observatory.md), the
[TV-01B preregistration](docs/TV-01B-spatial-coverage-preregistration.md), the
[TV-01B result](docs/TV-01B-spatial-coverage-result.md), and the current
[project status](docs/STATUS.md).
