# TinyLogicVision — Current Status

## Current authority

```text
experimental evidence          cd0a591 — TV-01B spatial-coverage result
consolidated baseline          TV-READY-01 — baseline frozen
application surface           TV-APP-00 — infrastructure and protocol implemented
application CLI                v0.1 READY — canonical CLI entry point and test suite
Local Web GUI                  READY — 127.0.0.1 browser application for dataset and dense classification
Dataset authoring              READY — manifest.csv, dataset.json, and spatial ROI management
ROI-level split integrity      READY — 1 ROI = 1 Split and backend spatial disjointness validation across splits
Dense patch classification     READY — in-memory C++ sliding window classification engine
Uncertainty & margin maps      READY — top-1 probability, top-1/top-2 margin, and threshold-based UNCERTAIN
Single overlay authority       READY — C++ canonical overlay.png centered on decisions in source image space (W×H)
Grid rasters geometry          READY — class_map.png, confidence.png, margin.png in grid space (Nx×Ny)
Canonical palette authority    READY — single palette defined in C++ engine and registered in run.json
Nominal resolution scale       READY — operator-declared nominal 10m/px scale (80x80m support)
NATURAL APP PROBE              EXPLORATORY / NOT CLAIM-BEARING
SYNTHETIC TEST                 SEALED / NOT_EVALUATED
H3 integration                 PLANNED / NOT IMPLEMENTED
```

## Demonstrated

- explicit C++23 learning core with 4,732 trainable parameters;
- numerical agreement between the manual backward pass and finite differences
  on the tested gradient case;
- deterministic fitting of the TV-00 synthetic TRAIN distribution;
- held-out validation failure under the original spatial coverage;
- a preregistered, paired, one-variable spatial-coverage intervention;
- exact repeated trajectories, including complete final parameter vectors;
- final validation improvement from 46.88% to 100.00%;
- broader TRAIN spatial coverage was sufficient to eliminate the observed
  validation failure under the fixed TV-01B conditions;
- PNG/JPEG RGB decoding and deterministic bilinear resize to 8x8;
- dynamic `192→24→N` application models, deterministic training, exact model
  persistence, individual classification, and labeled-split evaluation;
- an end-to-end three-class raster-fixture test with model round-trip and a
  held-out fixture split;
- a canonical application CLI entry point (`./bin/tinyvision`) orchestrating
  training, classification, evaluation, dense mapping, and verification with full path independence
  and CLI test coverage;
- a local Web GUI (`./bin/tinyvision web`) executing on `127.0.0.1` enabling interactive
  image inspection, nominal 10m/px resolution declaration vs display resolution tagging, strict ROI-level split integrity annotation,
  exact 8x8 patch extraction with provenance manifest generation, and integrated training,
  classification, dense mapping with interactive spatial inspection, and split evaluation;
- in-memory C++ dense classification engine (`tinyvision map`) sliding an exact 8x8 window
  without interpolation (`EXACT_RGB_INPUT_VECTOR`), with configurable strides (1, 2, 4, 8), producing `classification.csv`, `run.json`,
  `class_map.png` (grid space), `confidence.png` (grid space), `margin.png` (grid space), and `overlay.png` (source image space centered on decisions)
  with sub-millisecond execution per window;
- formal `DenseDecision` spatial semantics separating support window $[x, x+7] \times [y, y+7]$, geometric center $(x+3.5, y+3.5)$,
  grid coordinates $(gx, gy)$, and visual display cell;
- single palette authority registered in `run.json` and consumed by GUI;
- single overlay authority produced directly by C++ engine with centered decision cells;
- direct witness of `EXACT_RGB_INPUT_VECTOR` extraction without interpolation;
- backend enforcement of `ROI-LEVEL SPLIT INTEGRITY + NON-OVERLAPPING 8x8 SAMPLE SUPPORTS ACROSS SPLITS`.



VALIDATION never updated weights and remained frozen during TV-01B, but its
TV-01A observations informed the intervention question. It is an experimental
development distribution, not an independent confirmation split.

## Not demonstrated

- independent generalization on the sealed TEST split;
- classification of natural RGB images;
- a claim-bearing TV-APP-00 natural-image probe;
- abstract translation, scale, or rotation invariance;
- remote-sensing performance;
- robustness across model or training seeds;
- robustness across architectures or execution platforms;
- CNN superiority;
- integration with `sister-image`, SisTer, or dense vision.

## Canonical verification

```bash
./scripts/verify.sh
```

The command verifies TV-00, TV-01A, TV-01B, and the TV-APP-00 engineering
pipeline while leaving synthetic TEST and the natural-image probe unevaluated.

## Next authorized questions

TV-01C may ask whether the frozen Intervention B procedure generalizes to the
previously sealed synthetic TEST split without further selection or adjustment.

TV-01C must be preregistered and published independently before TEST is opened.
No TEST execution is authorized by this status document.

In the parallel application track, TV-APP-00 may ask whether approximately
homogeneous patches from real RGB imagery contain useful class signal for the
tiny MLP. A claim-bearing run requires a separately committed dataset record
with provenance, split membership, duplicate controls, metrics, and decision
boundaries.

The current synthetic baseline is frozen: new optimizations, additional seeds,
or architectures belong to later, separately authorized synthetic work. The
application pipeline does not revise the existing synthetic conclusions.
