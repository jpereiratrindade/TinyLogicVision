# TV-01A — Generalization Observatory

## Status

Pre-registered observational experiment. No adaptive intervention is authorized in TV-01A.

## Frozen baseline

TV-01A starts from commit:

```text
0fd5e71490021c935b9ba4715cc3e0e13a2287e0
feat(vision): establish deterministic RGB learning core
```

The following canonical training choices remain unchanged:

```text
input             8 x 8 x 3 = 192
hidden            24 tanh units
output            4 softmax classes
model seed        7
learning rate     0.025
epochs            180
optimizer         per-sample SGD
train dataset     24 samples/class, seed 11
```

No CNN, image library, augmentation framework, scheduler, checkpoint selection,
early stopping, architecture search, or external ML dependency is introduced.

## Question

> Does the TV-00 model retain useful recognition of the same spatial concepts
> when evaluated on RGB samples not used for weight updates and generated with
> independent spatial shifts, stronger color variation, brightness variation,
> and stronger pixel noise?

## Dataset roles

- `TRAIN`: may update weights.
- `VALIDATION`: is observed after each epoch and may inform a later REA successor experiment.
- `TEST`: generator exists but is not evaluated in TV-01A. It remains sealed from model/procedure selection.

The validation split is deterministic and uses a distinct seed and perturbation profile.

## Observations

TV-01A records, without changing training in response:

- training loss and accuracy;
- validation loss and accuracy;
- validation mean probability assigned to the true class;
- minimum validation margin `P(true) - max(P(other))`;
- four-class confusion matrix;
- epoch-boundary learning events;
- epoch-boundary forgetting events;
- best validation accuracy and epoch;
- minimum validation accuracy in the final 30 epochs.

The existing TV-00 gradient and fitting tests remain normative and must continue to pass.

## REA boundary

TV-01A is observation only. If the trajectory creates a surprise or exposes a
bottleneck, a successor experiment must be registered before changing training.

Allowed successor intervention families are deliberately bounded:

1. one learning-rate schedule intervention;
2. one training-duration intervention;
3. one deterministic presentation-order intervention;
4. one training-data-coverage intervention;
5. one hidden-capacity intervention.

A successor experiment must choose one primary intervention family at a time,
state its causal question and prediction, preserve a control trajectory, and
define support/refutation/inconclusive boundaries before execution.

The sealed test split must not be used to choose that intervention.

## Interpretation limits

Validation performance can support or challenge generalization within this
synthetic pattern family only. It does not establish performance on natural
images, remote sensing, arbitrary RGB objects, translational invariance, or a
general theory of visual representation.
