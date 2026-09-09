# TinyLogicVision

TinyLogicVision is a deliberately tiny RGB vision-learning experiment written
from scratch in C++23. Its current synthetic baseline is complete, deterministic,
and independently verifiable from a clean clone. A separate TV-APP-00 surface
can now train and apply the same tiny MLP to labeled PNG/JPEG patches.

## Current status

| Stage | Purpose | Status |
| --- | --- | --- |
| TV-00 | Explicit RGB learning core | PASS |
| TV-01A | Generalization observatory | PASS — validation failure observed |
| TV-01B | Spatial-coverage intervention | STRONG SUPPORT |
| TV-01C | Sealed synthetic TEST | NOT EXECUTED / SEALED |
| TV-APP-00 | Real RGB application pipeline | READY — natural-image probe not evaluated |
| Application CLI | Canonical v0.1 CLI interface | READY |
| Local Web GUI | Dataset authoring, patch extraction & UI workflow | READY |
| Split Integrity | ROI-level split integrity & backend spatial disjointness validation | READY |
| Dense patch classification | C++ sliding-window spatial map engine | READY |
| Uncertainty & Margin | Top-1 probability and top-1/top-2 margin thresholding | READY |
| Sentinel nominal resolution | 1px=10m nominal scale tagging (80x80m support) | READY |
| H3 integration | Geospatial indexing preparation | PLANNED / NOT IMPLEMENTED |

## Quick start

### 1. Local Web Application (Recommended)

Launch the self-contained local web interface at `127.0.0.1`:

```bash
./bin/tinyvision web
```

The interactive workflow guides you through:
1. **Abrir Imagem**: Carregue PNG/JPEG local (com indicação se a fonte possui pixels Sentinel-2 nominais de 10 m ou imagem de exibição);
2. **Definir Classes**: Crie classes rotuladas com paleta unificada e canônica;
3. **Marcar Regiões (ROIs)**: Desenhe áreas de interesse no canvas e atribua splits (`TRAIN`, `DEV`, `PROBE`) respeitando a integridade de splits por ROI (**1 ROI = 1 Split** e disjunção espacial entre splits validada no backend);
4. **Gerar Patches 8x8**: Extraia patches exatos de 8x8 pixels sem interpolação com manifesto de proveniência (`manifest.csv` e `dataset.json`);
5. **Treinar**: Execute o treinamento canônico C++ (`./bin/tinyvision train`) com métricas em tempo real;
6. **Classificar Imagem em Grade (Mapa Denso)**: Execute a classificação espacial densa em C++ (`./bin/tinyvision map`) com visualização de classes no espaço da grade, probabilidade top-1, margem top-1 − top-2, overlay canônico C++ e inspeção espacial interativa;
7. **Classificar & Avaliar**: Classifique novas imagens 8x8 e avalie splits mantendo o `PROBE` isolado.

### 2. Linha de Comando (CLI)

Build the release binaries, train, classify, evaluate, and generate dense classification maps via the canonical `./bin/tinyvision` CLI:

```bash
# 1. Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Train on a dataset (trains on train/, observes dev/)
./bin/tinyvision train ./dataset ./model.tlv

# 3. Dense spatial classification map (answers "where does the model see each class?")
./bin/tinyvision map \
    model.tlv \
    sentinel.png \
    .tinyvision/runs/map01 \
    --stride 1 \
    --sentinel-10m

# 4. Classify a single PNG/JPEG 8x8 patch
./bin/tinyvision classify ./model.tlv ./example.jpg

# 5. Evaluate on a labeled split
./bin/tinyvision evaluate ./model.tlv ./dataset/dev

# 6. Verify the entire project suite
./bin/tinyvision verify
```

### 3. Conceito da Classificação Espacial Densa

- **Suporte Contextual (Support Window)**: $8 \times 8$ pixels ($192$ entradas RGB scanline $[x, x+7] \times [y, y+7]$ extraídas diretamente sem interpolação como `EXACT_RGB_INPUT_VECTOR`).
- **Resolução Nominal Declarada**: $80 \times 80\text{ m}$ nominais quando declarada pelo operador como escala nominal Sentinel-2 ($10\text{ m/px}$). O sistema não valida metadata de satélite independente.
- **Espaçamento (Stride)**: Espaçamento entre decisões consecutivas na grade discreta (não altera o tamanho do suporte $8 \times 8$).
  - Stride 1: decisão a cada pixel usando suporte de $8 \times 8$.
  - Stride 2: decisão a cada 2 pixels.
  - Stride 4: decisão a cada 4 pixels.
  - Stride 8: decisão a cada 8 pixels (suportes disjuntos).
- **Semântica Espacial**:
  - `origin_x, origin_y`: Origem da janela na imagem fonte.
  - `center_x, center_y`: Centro geométrico exato $(x + 3.5, y + 3.5)$.
  - `grid_x, grid_y`: Coordenadas na grade discreta de decisões ($N_x \times N_y$).
  - `display_x, display_y`: Âncora visual inteira de exibição $(x + 4, y + 4)$.
  - *Regra fundamental*: $\text{SUPPORT} \neq \text{DECISION POINT} \neq \text{DISPLAY CELL}$.
- **Geometria dos Artefatos**:
  - `class_map.png`, `confidence.png` (Probabilidade Top-1) e `margin.png` (Margem Top-1 − Top-2) são rasters no **GRID SPACE** ($N_x \times N_y$), onde cada pixel é uma decisão.
  - `overlay.png` é gerado pelo engine C++ como **autoridade única canônica** no espaço da imagem fonte ($W \times H$), projetando cada decisão em sua célula $\text{stride} \times \text{stride}$ centrada na âncora de exibição: $[\text{display\_x} - \lfloor\text{stride}/2\rfloor, \text{display\_y} - \lfloor\text{stride}/2\rfloor]$ com clipping determinístico nas bordas.
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
    *Pergunta que responde*: "A classe vencedora tem força suficiente?"
    *Função*: Rejeita janelas onde o modelo não tem ativação expressiva para nenhuma classe conhecida (ex: ruído, nuvens ou padrões fora do domínio de treino).
    *Exemplo*: Em 3 classes, se as saídas forem $[0.36, 0.33, 0.31]$, a vencedora tem apenas $36\%$. Com limiar de $0.50$, a decisão é descartada e marcada como `UNCERTAIN`.

  - **Limiar de Margem Top-1 − Top-2 (`margin_threshold`)**:
    *Pergunta que responde*: "O modelo está indeciso entre duas classes concorrentes?"
    *Função*: Rejeita **ambiguidades competitivas** em zonas de transição ecológica (ex: borda floresta/campo ou transição solo/vegetação rala).
    *Exemplo*: Se $p_1 = 0.51$ (floresta) e $p_2 = 0.49$ (campo), $p_1$ passa no limiar de probabilidade $0.50$, mas a margem é de apenas $0.02$ ($2\%$). Com limiar de margem de $0.10$ ($10\%$), a ambiguidade é detectada e a decisão é marcada como `UNCERTAIN`.

  - **Interpretação Científica Rigorosa**:
    O status `UNCERTAIN` reflete exclusivamente **rejeição por limiar configurado pelo operador** (`UNCERTAIN_BY_CONFIGURED_THRESHOLD`). Não representa incerteza epistemológica nem calibração probabilística Bayesiana.

- **Integridade dos Splits**:
  - Validação estrita de `ROI-LEVEL SPLIT INTEGRITY + NON-OVERLAPPING 8x8 SAMPLE SUPPORTS ACROSS SPLITS` em todos os geradores de dataset. Não se alega independência espacial global irrestrita além dos mecanismos implementados.

### 4. Papel Futuro do H3 (Preparação Arquitetural)

O H3 **não substitui** a grade raster Sentinel nem o patch $8 \times 8$. O fluxo arquitetural planejado é:
$$\text{Raster Sentinel} \longrightarrow \text{Patches Nativos } 8 \times 8 \longrightarrow \text{Classificação Densa C++} \longrightarrow \text{Centro Geográfico} \longrightarrow \text{Indexação/Agregação H3}$$

O H3 será utilizado futuramente para:
1. Seleção espacial balanceada de amostras;
2. Prevenção de spatial leakage (células H3 disjuntas para TRAIN / DEV / PROBE);
3. Agregação espacial (classe dominante, proporção por classe, % de incerteza);
4. Comparação temporal multi-cena e integração com camadas web GIS.

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
