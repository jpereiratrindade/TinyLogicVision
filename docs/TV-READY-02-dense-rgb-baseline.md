# TV-READY-02 — Dense RGB Ready Baseline Freeze

## 1. Context & Authority

TinyLogicVision has established a complete, tested, and verifiable RGB dense spatial classification pipeline:

$$\text{Source RGB Image} \longrightarrow \text{Exact } 8 \times 8 \text{ Patch Vector (192 doubles)} \longrightarrow \text{MLP } 192 \to 24 \to N \longrightarrow \text{Dense Decisions } (N_x \times N_y) \longrightarrow \text{Artifacts + GUI}$$

This document freezes the technical baseline before the full multichannel, performance, geospatial, and H3 extension track.

## 2. Frozen Contracts

1. **Synthetic Core**:
   - TV-00: RGB spatial pattern fitting ($100\%$ train).
   - TV-01A: Generalization failure observed ($46.88\%$ dev accuracy, loss $4.421994$).
   - TV-01B: Spatial-coverage intervention with paired one-variable comparison ($100\%$ dev accuracy, loss $0.004919$, minimum margin $0.966333$, `STRONG SUPPORT`).
   - Synthetic `TEST`: **SEALED / NOT_EVALUATED**.

2. **RGB Application Surface (TV-APP-00)**:
   - Dynamic `192 \to 24 \to N` MLP architecture for $N$ classes.
   - Exact input vector extraction (`EXACT_RGB_INPUT_VECTOR`) from $8 \times 8$ scanlines without interpolation.
   - Exact model persistence (`.tlv` format v1).
   - Provenance manifests (`manifest.csv`, `dataset.json`).
   - Split Integrity: `ROI-LEVEL SPLIT INTEGRITY + NON-OVERLAPPING 8x8 SAMPLE SUPPORTS ACROSS SPLITS`.

3. **Dense Spatial Classification**:
   - Formal `DenseDecision` semantics:
     $$\text{SUPPORT } [x, x+7] \times [y, y+7] \neq \text{DECISION POINT } (x+3.5, y+3.5) \neq \text{DISPLAY CELL } [x+4-\lfloor S/2\rfloor, y+4-\lfloor S/2\rfloor]$$
   - Artifacts:
     - `class_map.png`: Grid space ($N_x \times N_y$).
     - `confidence.png`: Grid space ($N_x \times N_y$), uncalibrated Top-1 probability.
     - `margin.png`: Grid space ($N_x \times N_y$), Top-1 minus Top-2 margin.
     - `overlay.png`: Source image space ($W \times H$), single canonical C++ projection centered on decision points.
     - `classification.csv` & `run.json`: Structured execution contract.
   - Single Canonical Palette Authority: Defined in C++ engine and published in `run.json`.

4. **Status of Extended Tracks (At TV-READY-02 Freeze)**:
   - `NATURAL APP PROBE`: EXPLORATORY / NOT CLAIM-BEARING (NOT_EVALUATED).
   - `MULTICHANNEL`: NOT_IMPLEMENTED (Scheduled for TV-MB-00/01).
   - `GEOSPATIAL / GDAL`: NOT_IMPLEMENTED (Scheduled for TV-GEO-00/01).
   - `H3 INDEXING`: NOT_IMPLEMENTED (Scheduled for TV-H3-00).
   - `SPECTRAL ABLATION`: NOT_EVALUATED (Scheduled for TV-SPEC-00).

## 3. Engineering Witness Baseline

- All 8 canonical test suites passing:
  `gradient_test`, `training_test`, `generalization_observatory_test`, `spatial_coverage_experiment_test`, `application_pipeline_test`, `dense_test`, `cli_test`, `web_test`.
- Deterministic execution and exact parameter reproducibility.
