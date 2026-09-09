# TV-01B — REA Spatial-Coverage Intervention

## Status

Pre-registered successor experiment. This document authorizes no implementation
or execution until it is committed and published independently of the
intervention code.

## Authority and observed surprise

TV-01B starts from the published TV-01A authority:

```text
fdbcb71
experiment(vision): establish generalization observatory
```

TV-01A preserved the TV-00 training trajectory and observed:

```text
final train accuracy          100.00%
final validation accuracy      46.88%
best validation accuracy       47.92% at epoch 1
validation loss epoch 1         1.795085
validation loss epoch 180       4.421994
validation minimum margin      -0.999398
sealed TEST                     NOT_EVALUATED
```

The validation trajectory did not show a useful generalization state that was
later lost. Training accuracy reached 100% immediately while validation stayed
near 47% and became increasingly confident in wrong decisions.

## REA successor question

> Is insufficient spatial-phase coverage in TRAIN the primary bottleneck behind
> the TV-01A validation failure?

The intervention tests whether exposing the same tiny model to a broader set of
spatial shifts for the same four concepts is sufficient to improve validation
generalization without changing model capacity or optimization hyperparameters.

## Paired comparison

Both paths use exactly:

```text
input                     8 x 8 x 3 = 192
hidden                    24 tanh units
output                    4 softmax classes
model seed                7
learning rate             0.025
epochs                    180
optimizer                 per-sample SGD
training samples          96 total, 24/class
training RNG seed         11
validation samples        96 total, 24/class
validation RNG seed       101
validation distribution   frozen TV-01A generator
```

### Control A

Control A is the exact TV-01A training path:

```text
TRAIN spatial phase:
phase = sample_index % 2
```

All existing TV-00/TV-01A behavior must reproduce.

### Intervention B

Intervention B changes only the spatial placement rule used to construct TRAIN:

```text
shift_x = sample_index % 4
shift_y = (sample_index / 4) % 4
```

The 24 examples per class therefore cover a deterministic set of independent
horizontal and vertical shifts while retaining the same total sample count.

The intervention must preserve the TV-00 training distributions for:

```text
foreground level
background level
RGB tint jitter
pixel noise
RNG seed
number and order of RNG draws
final deterministic shuffle
```

No additional random draw may be introduced by the spatial-coverage rule.
Therefore the only intended causal difference between paired paths is the
spatial foreground arrangement of TRAIN examples.

## Frozen variables

TV-01B must not change:

- model architecture or parameter count;
- initialization seed;
- learning rate or schedule;
- epoch count;
- optimizer implementation;
- backward implementation;
- loss;
- number of training examples;
- validation generator or validation seed;
- validation observation metrics;
- class definitions;
- presentation-order mechanism;
- TEST generator or TEST visibility.

No early stopping, checkpoint selection, scheduler, data-size increase, hidden
capacity change, CNN, external ML framework, or image library is authorized.

## Prediction

If insufficient spatial coverage is the primary TV-01A bottleneck:

- Control A will reproduce approximately 46.88% final validation accuracy;
- Intervention B will retain at least 98% training accuracy;
- Intervention B validation accuracy will improve substantially;
- validation loss should fall rather than monotonically diverge;
- mean true-class probability should increase;
- the minimum true-class margin should become less negative or positive;
- the confusion matrix should become more diagonal.

## Decision boundaries

The primary comparison is the change in final validation accuracy relative to
the reproduced Control A trajectory.

```text
strong support
    intervention final validation >= 90%
    AND intervention final train >= 98%

partial support
    validation improvement >= 20 percentage points
    BUT final validation < 90%

refutation
    validation improvement < 5 percentage points

inconclusive
    validation improvement >= 5 and < 20 percentage points
    OR a technical/non-comparable regression prevents the paired claim
```

Secondary observations are validation loss, mean true-class probability,
minimum margin, confusion matrix, learning events, forgetting events, and
minimum validation accuracy in the final 30 epochs.

The classification above is determined by the primary boundary. Secondary
metrics refine interpretation but do not move an outcome across a primary
boundary after observation.

## Reproducibility requirement

Control and intervention must each be executed twice from fresh model
initialization. Their complete milestone metrics, final confusion matrices,
event counts, and final model parameter vectors must reproduce exactly within
the deterministic implementation.

Any failure to reproduce invalidates the causal comparison until resolved.

## Sealed TEST contract

`SyntheticSplit::Test` remains sealed throughout TV-01B.

```text
sealed_test_status=NOT_EVALUATED
```

The TEST split must not be used to choose, tune, reject, or reinterpret the
spatial-coverage intervention.

A successful TV-01B result may justify a later TV-01C sealed generalization
witness. It does not itself authorize opening TEST.

## Evidence limits

Even strong support would establish only that broader deterministic spatial
coverage is sufficient to improve generalization within this synthetic
four-pattern RGB family under this fixed tiny MLP trajectory.

It would not prove translation invariance, natural-image recognition, remote
sensing performance, robustness across architectures/seeds, or that spatial
coverage is the only generalization bottleneck.
