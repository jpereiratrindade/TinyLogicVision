# TV-APP-00 — Real RGB Probe

## Status

Application infrastructure and protocol are implemented. No natural-image
dataset result is claimed, and no application probe has been executed as
scientific evidence.

The synthetic TV-01C TEST remains sealed. TV-APP-00 is a parallel application
track and does not evaluate or depend on that split.

## Question

> Can the explicit tiny MLP extract useful class signal from approximately
> homogeneous patches sampled from real RGB images?

The first application probe is deliberately narrower than scene recognition.
It concerns local patches whose visual content is dominated by one declared
class, such as vegetation, exposed soil, or water.

## Implemented surface

```text
PNG/JPEG file
    -> RGB decode
    -> deterministic bilinear resize to 8 x 8
    -> normalization to [0, 1]
    -> 192 inputs
    -> 24 tanh units
    -> N declared classes
```

The output layer is dynamic. With three classes, the model has 4,707 trainable
parameters:

```text
(192 * 24) + 24 + (24 * 3) + 3 = 4707
```

The 4,732-parameter count remains specific to the four-output synthetic model.

## Dataset contract

```text
dataset/
├── train/
│   ├── soil/
│   ├── vegetation/
│   └── water/
├── dev/
│   ├── soil/
│   ├── vegetation/
│   └── water/
└── probe/
    ├── soil/
    ├── vegetation/
    └── water/
```

Class-directory names and image paths are sorted deterministically. TRAIN is
shuffled once with seed 11. The current default trajectory uses model seed 7,
24 hidden units, learning rate 0.025, 180 epochs, and per-sample SGD.

`tinyvision_train` reads only `train/` and `dev/`. It does not inspect `probe/`.
The final epoch is always persisted; DEV observation does not select a
checkpoint or change training.

Before a natural-image result is classified, a separate dataset record must
freeze:

- the class definitions and inclusion/exclusion rules;
- image provenance and licenses;
- patch extraction and labeling procedure;
- TRAIN, DEV, and PROBE membership or content hashes;
- duplicate and near-duplicate controls across splits;
- samples per class and class balance;
- primary metric and chance baseline;
- support, refutation, and inconclusive boundaries.

Generated or programmatic raster fixtures may verify decoding and training, but
must not be reported as evidence about natural RGB imagery.

Current preprocessing limitations must be handled by the dataset record:

- JPEG EXIF orientation is not applied automatically;
- transparent PNG backgrounds are not a declared input contract;
- images are read only from the immediate class directory, not recursively;
- bilinear reduction to 8x8 may discard fine spatial structure.

## Commands

Train and persist the final model without opening the application probe:

```bash
./build/tinyvision_train ./dataset ./model.tlv
```

Classify one previously unseen file:

```bash
./build/tinyvision_classify ./model.tlv ./example.jpg
```

Evaluate a labeled split and print accuracy and its confusion matrix:

```bash
./build/tinyvision_evaluate ./model.tlv ./dataset/probe
```

Running the last command is an experimental observation. For a claim-bearing
natural-image probe, it should occur only after the dataset record and decision
boundaries are committed independently.

## Current evidence boundary

The automated application test creates controlled PNG/JPEG fixtures at runtime
and verifies decoding, deterministic preprocessing, three-class training,
model round-trip, and held-out fixture evaluation. It is technical verification
of the pipeline, not functional evidence on natural images.

No synthetic TEST sample is evaluated directly or indirectly.
