# TinyLogicVision — Current Status

## Current authority

```text
experimental evidence    cd0a591 — TV-01B spatial-coverage result
consolidated baseline    TV-READY-01 — commit containing this document
application surface     TV-APP-00 — infrastructure and protocol implemented
application CLI          v0.1 READY — canonical CLI entry point and test suite
Local Web GUI            READY — 127.0.0.1 browser application for patch authoring and workflow
Dataset authoring        READY — manifest.csv, dataset.json, and spatial ROI management
Patch extraction         READY — exact 8x8 source pixel sampling with stride 8
NATURAL APP PROBE        NOT_EVALUATED
sealed TEST              NOT_EVALUATED
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
  training, classification, evaluation, and verification with full path independence
  and CLI test coverage;
- a local Web GUI (`./bin/tinyvision web`) executing on `127.0.0.1` enabling interactive
  image inspection, Sentinel 10m vs display resolution tagging, ROI annotation, exact
  8x8 patch extraction with provenance manifest generation, and integrated training,
  classification, and split evaluation without modifying core ML mechanics or baseline data.



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
