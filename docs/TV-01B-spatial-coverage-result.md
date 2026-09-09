# TV-01B — Spatial-Coverage Intervention Result

## Authority

Pre-registration:

```text
a84d125
docs(vision): preregister spatial coverage intervention
```

Observed baseline authority:

```text
fdbcb71
experiment(vision): establish generalization observatory
```

The experiment was materialized only after the pre-registration was committed
and published.

## Experimental integrity

The paired experiment preserved:

```text
input                     8 x 8 x 3 = 192
hidden                    24 tanh units
output                    4 softmax classes
parameter count           4732
model seed                7
learning rate             0.025
epochs                    180
optimizer                 per-sample SGD
training samples          96 total, 24/class
training RNG seed         11
validation samples        96 total, 24/class
validation RNG seed       101
```

Control A preserved the TV-00/TV-01A spatial rule:

```text
phase = sample_index % 2
```

Intervention B changed only TRAIN spatial coverage:

```text
shift_x = sample_index % 4
shift_y = (sample_index / 4) % 4
```

No additional RNG draw was introduced. The TEST split was not evaluated.

```text
sealed_test_status=NOT_EVALUATED
```

## Control A

```text
final train loss                       0.000398
final train accuracy                    100.00%
final validation loss                  4.421994
final validation accuracy               46.88%
best validation accuracy                47.92%
best validation epoch                        1
validation learning events                  34
validation forgetting events                14
minimum validation accuracy,
  final 30 epochs                       46.88%
mean true-class probability           0.461435
minimum margin                       -0.999398
parameter count                            4732
parameter fingerprint        0x925fca5b8a91c909
```

Final validation confusion matrix:

```text
             horizontal vertical checkerboard diagonal
horizontal           12        4        1        7
vertical              0       12        4        8
checkerboard          0        3        9       12
diagonal              0       11        1       12
```

The repeated Control A trajectory reproduced exactly.

## Intervention B

```text
final train loss                       0.002324
final train accuracy                    100.00%
final validation loss                  0.004919
final validation accuracy              100.00%
best validation accuracy               100.00%
first best validation epoch                 14
validation learning events                  92
validation forgetting events                21
minimum validation accuracy,
  final 30 epochs                      100.00%
mean true-class probability           0.995097
minimum margin                        0.966333
parameter count                            4732
parameter fingerprint        0xc692e4bf33e35eb1
```

Final validation confusion matrix:

```text
             horizontal vertical checkerboard diagonal
horizontal           24        0        0        0
vertical              0       24        0        0
checkerboard          0        0       24        0
diagonal              0        0        0       24
```

The repeated Intervention B trajectory reproduced exactly.

## Primary result

```text
Control A final validation          46.88%
Intervention B final validation    100.00%
improvement                         53.12 percentage points
Intervention B final train         100.00%
preregistered classification       STRONG SUPPORT
```

The pre-registered strong-support boundary was:

```text
intervention final validation >= 90%
AND intervention final train >= 98%
```

The result therefore satisfies the boundary without post-observation
reinterpretation.

## Reachability, stability and final outcome

```text
Intervention B epoch 1 validation      40.62%
Intervention B epoch 10 validation     91.67%
first 100% validation epoch                14
final 30-epoch minimum validation     100.00%
epoch 180 validation                  100.00%
```

Reachability, stability, and the pre-declared final outcome are all supported
on this deterministic trajectory.

## REA interpretation

### Expected

Control A should reproduce the TV-01A failure. If insufficient spatial coverage
were the primary bottleneck, Intervention B should retain at least 98% training
accuracy and substantially improve validation.

### Observed

Control A reproduced exactly. Intervention B reached 100% training and
validation accuracy, low validation loss, a positive minimum margin, and a
perfectly diagonal final confusion matrix.

### Surprise

Intervention B initially underperformed Control A at epoch 1 (40.62% versus
47.92%), then improved rapidly to 91.67% by epoch 10 and first reached 100% at
epoch 14.

### Interpretation

Broader deterministic spatial coverage of TRAIN is sufficient to resolve the
TV-01A validation failure within this fixed synthetic RGB family, seed,
architecture, optimizer, and trajectory.

This does not establish that spatial coverage was the unique or necessary
cause, or that the model learned abstract translation invariance.

### Successor question

> With the Intervention B procedure now frozen, does the model preserve useful
> generalization on the previously sealed TEST split without any further
> selection or adjustment?

No successor intervention is materialized by TV-01B.

## Evidence limits

The result is limited to:

- one synthetic four-pattern RGB family;
- one model initialization seed;
- one training seed;
- one fixed validation distribution;
- one tiny MLP architecture;
- one deterministic execution environment.

It does not establish natural-image recognition, remote-sensing performance,
cross-platform bitwise reproducibility, robustness across seeds or
architectures, or abstract translation invariance.
