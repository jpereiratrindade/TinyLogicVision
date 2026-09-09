# TinyLogicVision — Current Status

## Current authority

```text
experimental evidence    cd0a591 — TV-01B spatial-coverage result
consolidated baseline    TV-READY-01 — commit containing this document
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
  validation failure under the fixed TV-01B conditions.

VALIDATION never updated weights and remained frozen during TV-01B, but its
TV-01A observations informed the intervention question. It is an experimental
development distribution, not an independent confirmation split.

## Not demonstrated

- independent generalization on the sealed TEST split;
- classification of natural RGB images;
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

The command verifies TV-00, TV-01A, and TV-01B while leaving TEST sealed.

## Next authorized question

TV-01C may ask whether the frozen Intervention B procedure generalizes to the
previously sealed synthetic TEST split without further selection or adjustment.

TV-01C must be preregistered and published independently before TEST is opened.
No TEST execution is authorized by this status document.

The current synthetic baseline is frozen: new optimizations, additional seeds,
architectures, image formats, or real-image experiments belong to later,
separately authorized work.
