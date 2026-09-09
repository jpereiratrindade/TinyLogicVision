# TinyLogicVision

TinyLogicVision is a deliberately tiny RGB vision-learning experiment written
from scratch in C++23. Its current synthetic baseline is complete, deterministic,
and independently verifiable from a clean clone. A separate TV-APP-00 surface
can now train and apply the same tiny MLP to labeled PNG/JPEG patches.

## Current status

| Stage / Capability | Purpose | Status |
| --- | --- | --- |
| TV-00 | Explicit RGB learning core | PASS |
| TV-01A | Generalization observatory | PASS — validation failure observed |
| TV-01B | Spatial-coverage intervention | STRONG SUPPORT |
| TV-01C | Sealed synthetic TEST | NOT EXECUTED / SEALED |
| TV-APP-00 | Real RGB application pipeline | READY — natural-image probe not evaluated |
| Core Performance (TV-PERF-00) | Reusable zero-heap MLP workspaces | READY |
| Tiled Dense Engine (TV-PERF-01) | Tiled streaming inference & binary O(1) index | READY |
| Input Schema (TV-MB-00) | Generic WxHxC schema & Model format v2 | READY |
| Multichannel Tensor (TV-MB-01) | Generic raster tensor & .tvp dataset foundation | READY |
| Geospatial Raster (TV-GEO-00) | Optional GDAL source & strict band alignment | READY WHEN BUILT |
| Sentinel-2 10m (TV-S2-00) | Native B2/B3/B4/B8 10m modality (256 inputs) | READY |
| Georeferenced Outputs (TV-GEO-01) | Dense GeoTIFF maps (class, confidence, margin) | READY WHEN BUILT |
| Provenance & RIT (TV-EVIDENCE-00) | Directed evidence graph (JSON/JSONL lineage) | READY |
| Geospatial H3 (TV-H3-00) | Hex discrete sampling, split partition & aggregation | READY WHEN BUILT |
| Spectral Ablation (TV-SPEC-00) | Neutral channel masking protocol | PREPARED / NOT_EVALUATED |
| Web Async Jobs | Background CLI workers with polling on 127.0.0.1 | READY |
| Application CLI | Unified command line interface & benchmark | READY |

## Quick start

### 1. Aplicação Web Local (Web GUI v2.0)

Inicie a interface web local auto-contida em `127.0.0.1`:

```bash
./bin/tinyvision web
```

O fluxo de trabalho interativo unificado oferece:
1. **Modalidade & Fonte**: Escolha entre **RGB (192 entradas)** ou **Sentinel-2 10m Multibanda (B2, B3, B4, B8 — 256 entradas)** com metadados geoespaciais (CRS, pixel size, geotransform);
2. **Definição de Classes**: Crie classes rotuladas com paleta unificada e canônica (ou preset Cerrado/Sentinel);
3. **Regiões de Interesse (ROIs) & Particionamento H3**: Desenhe ROIs no canvas com validação de **1 ROI = 1 Split** e opção de **Particionamento Espacial H3** (1 célula H3 = 1 split) para prevenir spatial leakage;
4. **Extração de Patches & Proveniência**: Extraia patches exatos de $8 \times 8$ (ou tensores nativos `.tvp`) com manifesto e grafo de proveniência RIT (`manifest.csv`, `dataset.json`, `provenance.json`);
5. **Treinamento Síncrono ou Assíncrono**: Treine modelos v1/v2 em C++ (`./bin/tinyvision train`) em segundo plano com monitoramento em tempo real via aba de Jobs;
6. **Classificação Densa, GeoTIFF & Agregação H3**: Execute o motor de streaming C++ (`./bin/tinyvision map`) com multithreading (`--threads N`), inspeção $O(1)$ por seek binário, download de GeoTIFFs (`class_map.tif`, `confidence.tif`, `margin.tif`) e agregação espacial H3 (`classification_h3.csv`);
7. **Classificar & Avaliar**: Classifique amostras individuais e avalie splits mantendo o `PROBE` isolado.

### 2. Linha de Comando (CLI)

Treine, classifique, avalie, execute benchmarks e gere mapas densos georreferenciados via `./bin/tinyvision`:

```bash
# 1. Build em modo Release
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Treinar modelo em dataset (RGB ou Multicanal Sentinel-2)
./bin/tinyvision train ./dataset ./model.tlv

# 3. Classificação espacial densa em grade (respostas: "onde o modelo vê cada classe?")
./bin/tinyvision map \
    model.tlv \
    sentinel_scene.png \
    .tinyvision/runs/map01 \
    --stride 2 \
    --threads 8 \
    --confidence 0.50 \
    --margin 0.10 \
    --sentinel-10m

# 4. Benchmark de engenharia CPU / memória (zero alocações de heap no hot loop)
./bin/tinyvision benchmark

# 5. Classificar amostra individual (PNG, JPEG ou patch .tvp)
./bin/tinyvision classify ./model.tlv ./example.png

# 6. Avaliar split rotulado
./bin/tinyvision evaluate ./model.tlv ./dataset/dev

# 7. Executar suite de verificação canônica
./bin/tinyvision verify
```

### 3. Conceito da Classificação Espacial Densa & GeoTIFF

- **Suporte Contextual (Support Window)**: $8 \times 8$ pixels ($192$ entradas RGB ou $256$ entradas Sentinel-2 $8 \times 8 \times 4$ $[B2, B3, B4, B8]$ scanline extraídas diretamente sem interpolação).
- **Resolução Nominal Declarada**: $80 \times 80\text{ m}$ nominais quando declarada como escala nominal Sentinel-2 ($10\text{ m/px}$).
- **Espaçamento (Stride)**: Espaçamento entre decisões consecutivas na grade discreta (não altera o suporte $8 \times 8$):
  - Stride 1: decisão a cada pixel ($10\text{ m}$ nominais).
  - Stride 2: decisão a cada 2 pixels ($20\text{ m}$).
  - Stride 4: decisão a cada 4 pixels ($40\text{ m}$).
  - Stride 8: decisão a cada 8 pixels ($80\text{ m}$ — blocos disjuntos).
- **Semântica Espacial & Coordenadas**:
  - `origin_x, origin_y`: Origem da janela na imagem fonte.
  - `center_x, center_y`: Centro geométrico exato $(x + 3.5, y + 3.5)$.
  - `grid_x, grid_y`: Coordenadas na grade discreta de decisões ($N_x \times N_y$).
  - `display_x, display_y`: Âncora visual inteira de exibição $(x + 4, y + 4)$.
  - `map_x, map_y`: Coordenadas no CRS projetado da fonte (quando georreferenciada).
  - `h3_index`: Índice da célula hexadecimal H3 correspondente ao centro do patch.
  - *Regra fundamental*: $\text{SUPPORT} \neq \text{DECISION POINT} \neq \text{DISPLAY CELL} \neq \text{H3 CELL}$.
- **Geometria dos Artefatos**:
  - `class_map.png`, `confidence.png` (Probabilidade Top-1) e `margin.png` (Margem Top-1 − Top-2) são rasters no **GRID SPACE** ($N_x \times N_y$).
  - `class_map.tif`, `confidence.tif` e `margin.tif` são rasters georreferenciados em formato GeoTIFF 6.0 com CRS e geotransform intactos.
  - `overlay.png` é gerado pelo engine C++ como **autoridade única canônica** no espaço da imagem fonte ($W \times H$).
  - `classification_h3.csv` consolida estatísticas agregadas por célula H3 (classe dominante, proporção de classes e incerteza).
  - `decisions.bin` fornece índice binário compacto ($36\text{ bytes/registro}$) para inspeção espacial com tempo de acesso $O(1)$.
- **Interpretação e Papel dos Limiares de Decisão**:

  Na inferência de cada janela $8 \times 8$, a camada softmax produz probabilidades uncalibradas para as $N$ classes:
  - $p_1$ (**Probabilidade Top-1**): Maior probabilidade obtida (classe predita).
  - $p_2$ (**Probabilidade Top-2**): Segunda maior probabilidade obtida.
  - $\text{Margem} = p_1 - p_2$: Diferença entre a primeira e a segunda opção.

  ```text
                    ┌─ p1 < Limiar de Probabilidade  ──┐
  Softmax (p1, p2) ─┤                                  ├──> Status: UNCERTAIN (Cinza #808080)
                    └─ (p1 - p2) < Limiar de Margem ───┘
                                    │
                         (Se ambos forem atendidos)
                                    ↓
                         Status: CLASSIFIED (Cor da Classe)
  ```

  - **Limiar de Probabilidade Top-1 (`confidence_threshold`)**:
    *Pergunta*: "A classe vencedora tem força suficiente?"
    *Função*: Rejeita janelas onde o modelo não tem ativação expressiva para nenhuma classe conhecida.
  - **Limiar de Margem Top-1 − Top-2 (`margin_threshold`)**:
    *Pergunta*: "O modelo está indeciso entre duas classes concorrentes?"
    *Função*: Rejeita **ambiguidades competitivas** em zonas de transição ecológica (ex: borda floresta/campo).
  - **Interpretação Científica Rigorosa**:
    O status `UNCERTAIN` reflete exclusivamente **rejeição por limiar configurado pelo operador** (`UNCERTAIN_BY_CONFIGURED_THRESHOLD`).

### 4. Papel do H3 & Grafo de Proveniência RIT

O H3 **não substitui** o raster nem o patch $8 \times 8$. O fluxo arquitetural canônico é:
$$\text{Raster / Cena Sentinel} \longrightarrow \text{Patches Nativos } 8 \times 8 \longrightarrow \text{Classificação Densa C++} \longrightarrow \text{Centro Geográfico} \longrightarrow \text{Indexação/Agregação H3}$$

O H3 atua em:
1. **Amostragem & Particionamento**: Prevenção de spatial leakage através da política 1 Célula H3 = 1 Split (sem prejuízo à checagem de disjunção geométrica de suportes $8 \times 8$).
2. **Agregação Espacial**: Agrupamento estatístico em `classification_h3.csv` com proporção de classes e incertezas por hexágono.
3. **Linhagem Temporal (RIT)**: Rastreabilidade explícita através de `provenance.json` e `evidence.jsonl` ligando fontes, ROIs, modelos e artefatos.

## Canonical verification

After cloning, run the complete project and synthetic baseline verification with:

```bash
./bin/tinyvision verify
# or: ./scripts/verify.sh
```

The command configures and builds the Release/Ninja tree, runs all 8 test suites
(including the CLI, dense spatial classification, and Web test suites), and checks the TV-00, TV-01A, TV-01B, and TV-APP-00
engineering witnesses. It does not evaluate synthetic TEST or a natural-image application probe.


The equivalent manual build commands are:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Build prerequisites are a C++23 compiler, CMake, Ninja, and development packages
for libpng and libjpeg.


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

## TV-APP-00 — Real RGB application surface

TV-APP-00 is a parallel application track. It does not open TV-01C or alter the
frozen synthetic evidence.

Expected dataset structure:

```text
dataset/
├── train/<class>/*.{png,jpg,jpeg}
├── dev/<class>/*.{png,jpg,jpeg}
└── probe/<class>/*.{png,jpg,jpeg}
```

Train on `train/`, observe `dev/`, and persist the final epoch:

```bash
./bin/tinyvision train ./dataset ./model.tlv
# or direct binary: ./build/tinyvision_train ./dataset ./model.tlv
```

Classify one image or explicitly evaluate a labeled probe:

```bash
./bin/tinyvision classify ./model.tlv ./example.jpg
./bin/tinyvision evaluate ./model.tlv ./dataset/probe
```

The training command never reads `probe/`. The output layer follows the number
of sorted class directories: a three-class `192→24→3` model has 4,707 trainable
parameters, while the four-class synthetic model has 4,732.

The automated test uses controlled PNG/JPEG fixtures to verify the technical
pipeline. It is not evidence of useful classification on natural images. See
the [TV-APP-00 protocol](docs/TV-APP-00-real-rgb-probe.md) before constructing a
claim-bearing real dataset.

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

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).
See the [LICENSE](LICENSE) file for the full license text.
