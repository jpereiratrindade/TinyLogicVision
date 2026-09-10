# TinyLogicVision — Current Status

## Current authority

```text
experimental evidence          cd0a591 — TV-01B spatial-coverage result
consolidated baseline          TV-READY-02 — dense RGB baseline frozen
RGB application                READY — RGB application pipeline implemented & preserved
Dense reference engine         READY — in-memory equivalence authority for small test rasters
Sentinel dense streaming       READY — sequential tile+halo engine; memory bounded by tile size
Streaming concurrency/batches  NOT IMPLEMENTED — later optimization after deterministic baseline
CPU/memory optimized path      READY — zero-heap allocation hot inference workspaces (std::span)
Multichannel foundation        READY — N-channel tensor abstraction (UINT8, UINT16, FLOAT32, FLOAT64)
Model v2                       READY — format with explicit InputSchema & ChannelSpec
Model v1 compatibility         READY — legacy .tlv format load & byte-for-byte numerical reproducibility
Sentinel B2/B3/B4/B8           READY — 10m 4-band Sentinel-2 modality (8x8x4 -> 256 inputs)
GDAL GEO source                READY WITH GDAL — native windowed B2/B3/B4/B8 source and strict alignment
Geo dense outputs              READY WITH GDAL — CRS/full-affine GeoTIFF class, confidence, margin rasters
Provenance/RIT graph           READY — real JSON/JSONL lineage loaded by the Web explorer
H3 integration                 READY WITH GDAL + H3 — official cells after source CRS → WGS84 transformation
Web Job Execution              READY — local asynchronous worker execution on 127.0.0.1
Application CLI                READY — unified command-line authority (train, classify, evaluate, map, benchmark, web, verify)
Synthetic TEST                 SEALED / NOT_EVALUATED
Sentinel scientific performance NOT_EVALUATED
Spectral ablation result       NOT_EVALUATED
NATURAL APP PROBE              EXPLORATORY / NOT CLAIM-BEARING
```

## Architectural and Scientific Invariants

- **H3 não é entrada do MLP**: H3 é uma camada geo-espacial para indexação, amostragem e agregação estatística; o MLP processa exclusivamente tensores de entrada $W \times H \times C$.
- **H3 não substitui o raster**: O raster contínuo continua sendo a fonte primária de verdade dos pixels; H3 mapeia centros de decisões para hexágonos discretos.
- **Preview RGB não é multiband tensor**: Visualizações RGB (ex: composição B4/B3/B2) são artefatos de visualização para o operador humano e jamais substituem os dados multiespectrais nativos enviados ao modelo.
- **Geo metadata não é inferida por escala visual**: CRS, pixel size e geotransform são obtidos unicamente de metadados georreferenciados válidos (GeoTIFF/GDAL) ou marcados como desconhecidos.
- **Band order é parte estrita do InputSchema**: A ordem e os nomes dos canais (ex: B2, B3, B4, B8) fazem parte da assinatura do modelo; discrepâncias de ordem ou quantidade de canais provocam rejeição determinística imediata.
- **Normalização é parte do InputSchema**: Escalas e offsets são registrados de forma explícita e determinística; não há normalização implícita ou min/max dependente da imagem.
- **Multichannel não invalida RGB**: Suportes multicanais operam como uma camada sobre o TinyLogicVision sem alterar ou descartar o pipeline RGB legado.
- **Model v2 não invalida Model v1**: O carregador interpreta modelos v1 legados como RGB ($8 \times 8 \times 3$, `uint8_div_255`) com exata reprodução de saídas.

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
- isolated selectable workspace roots, governed by `tinylogicvision.workspace/v1`,
  for uploads, datasets, models, runs, and manifests;
- an optional Qt Quick desktop launcher (`./bin/tinyvision gui [WORKSPACE]`) that
  selects or creates a workspace, owns the local server process, and opens the shared Web workbench;
- individual ROI deletion in dataset authoring, including removal of only the
  patches owned by that ROI and immediate recomputation of split summaries;
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
