/**
 * TinyLogicVision Web GUI Application Logic v0.1
 */

// State
const state = {
  currentImage: null,
  imagePath: null,
  imageMeta: null,
  classes: [
    { name: 'floresta_natural', color: '#22c55e' },
    { name: 'vegetacao_campestre', color: '#eab308' },
    { name: 'solo_descoberto', color: '#f97316' },
  ],
  rois: [], // { id, x, y, width, height, class, split }
  patches: [], // { id, roi_id, x, y, width: 8, height: 8, class, split, selected: true }
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isDrawing: false,
  drawStartX: 0,
  drawStartY: 0,
  currentRect: null,
  datasets: [],
  models: [],
};

// Elements
const el = {
  tabs: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('image-file-input'),
  imageMeta: document.getElementById('image-meta'),
  metaFilename: document.getElementById('meta-filename'),
  metaDims: document.getElementById('meta-dims'),
  metaSha: document.getElementById('meta-sha'),
  chkSentinel10m: document.getElementById('chk-sentinel-10m'),
  sentinelBadge: document.getElementById('sentinel-badge'),
  newClassName: document.getElementById('new-class-name'),
  newClassColor: document.getElementById('new-class-color'),
  btnAddClass: document.getElementById('btn-add-class'),
  btnPresetSentinel: document.getElementById('btn-preset-sentinel'),
  classesList: document.getElementById('classes-list'),
  selectActiveClass: document.getElementById('select-active-class'),
  selectActiveSplit: document.getElementById('select-active-split'),
  btnClearRois: document.getElementById('btn-clear-rois'),
  roiCount: document.getElementById('roi-count'),
  datasetNameInput: document.getElementById('dataset-name-input'),
  summaryTableBody: document.getElementById('summary-table-body'),
  datasetWarnings: document.getElementById('dataset-warnings'),
  btnAutoSplit: document.getElementById('btn-auto-split'),
  btnGenerateDataset: document.getElementById('btn-generate-dataset'),
  canvas: document.getElementById('main-canvas'),
  canvasContainer: document.getElementById('canvas-container'),
  canvasPlaceholder: document.getElementById('canvas-placeholder'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnZoomReset: document.getElementById('btn-zoom-reset'),
  zoomLevel: document.getElementById('zoom-level'),
  chkShowGrid: document.getElementById('chk-show-grid'),
  chkShowRois: document.getElementById('chk-show-rois'),
  cursorCoords: document.getElementById('cursor-coords'),
  canvasPatchInfo: document.getElementById('canvas-patch-info'),
  // Train
  trainSelectDataset: document.getElementById('train-select-dataset'),
  trainModelName: document.getElementById('train-model-name'),
  btnStartTrain: document.getElementById('btn-start-train'),
  trainStatusBadge: document.getElementById('train-status-badge'),
  metricTrainAcc: document.getElementById('metric-train-acc'),
  metricDevAcc: document.getElementById('metric-dev-acc'),
  metricDevLoss: document.getElementById('metric-dev-loss'),
  metricParams: document.getElementById('metric-params'),
  trainTerminalLog: document.getElementById('train-terminal-log'),
  // Classify & Eval
  classifySelectModel: document.getElementById('classify-select-model'),
  classifyImageFile: document.getElementById('classify-image-file'),
  classifyPreviewImg: document.getElementById('classify-preview-img'),
  btnRunClassify: document.getElementById('btn-run-classify'),
  classifyResults: document.getElementById('classify-results'),

  classifyPredClass: document.getElementById('classify-pred-class'),
  classifyPredScore: document.getElementById('classify-pred-score'),
  classifyProbsContainer: document.getElementById('classify-probs-container'),
  evalSelectModel: document.getElementById('eval-select-model'),
  evalSelectDataset: document.getElementById('eval-select-dataset'),
  btnEvalDev: document.getElementById('btn-eval-dev'),
  btnEvalProbe: document.getElementById('btn-eval-probe'),
  evalResults: document.getElementById('eval-results'),
  evalSplitTag: document.getElementById('eval-split-tag'),
  evalAcc: document.getElementById('eval-acc'),
  evalSamples: document.getElementById('eval-samples'),
  evalLoss: document.getElementById('eval-loss'),
  evalMeanProb: document.getElementById('eval-mean-prob'),
  confusionMatrixContainer: document.getElementById('confusion-matrix-container'),
  // Dense Spatial Classification Elements
  denseSelectModel: document.getElementById('dense-select-model'),
  denseSelectImage: document.getElementById('dense-select-image'),
  denseFileInput: document.getElementById('dense-file-input'),
  btnDenseUpload: document.getElementById('btn-dense-upload'),
  denseSelectStride: document.getElementById('dense-select-stride'),
  denseSliderConfidence: document.getElementById('dense-slider-confidence'),
  denseValConfidence: document.getElementById('dense-val-confidence'),
  denseSliderMargin: document.getElementById('dense-slider-margin'),
  denseValMargin: document.getElementById('dense-val-margin'),
  denseChkSentinel10m: document.getElementById('dense-chk-sentinel-10m'),
  denseSentinelBadge: document.getElementById('dense-sentinel-badge'),
  btnRunDenseMap: document.getElementById('btn-run-dense-map'),
  denseProgress: document.getElementById('dense-progress'),
  denseProgressText: document.getElementById('dense-progress-text'),
  denseResultsPanel: document.getElementById('dense-results-panel'),
  denseCanvas: document.getElementById('dense-canvas'),
  denseCanvasContainer: document.getElementById('dense-canvas-container'),
  denseCrosshair: document.getElementById('dense-crosshair'),
  denseChkOverlay: document.getElementById('dense-chk-overlay'),
  denseSliderOpacity: document.getElementById('dense-slider-opacity'),
  denseOpacityVal: document.getElementById('dense-opacity-val'),
  denseChkShowUncertain: document.getElementById('dense-chk-show-uncertain'),
  inspOrigin: document.getElementById('insp-origin'),
  inspCenter: document.getElementById('insp-center'),
  inspDisplay: document.getElementById('insp-display'),
  inspSupport: document.getElementById('insp-support'),
  inspStatus: document.getElementById('insp-status'),
  inspTop1Class: document.getElementById('insp-top1-class'),
  inspTop1Prob: document.getElementById('insp-top1-prob'),
  inspTop2Class: document.getElementById('insp-top2-class'),
  inspTop2Prob: document.getElementById('insp-top2-prob'),
  inspMargin: document.getElementById('insp-margin'),
  denseDynamicLegend: document.getElementById('dense-dynamic-legend'),
  denseTotalDecisions: document.getElementById('dense-total-decisions'),
  denseContextInfo: document.getElementById('dense-context-info'),
  linkDownloadCsv: document.getElementById('link-download-csv'),
  linkDownloadJson: document.getElementById('link-download-json'),
  linkDownloadOverlay: document.getElementById('link-download-overlay'),
  linkDownloadClassmap: document.getElementById('link-download-classmap'),
  // Browser
  browserDatasetsList: document.getElementById('browser-datasets-list'),
  browserModelsList: document.getElementById('browser-models-list'),
};

const ctx = el.canvas ? el.canvas.getContext('2d') : null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupClassManagement();
  setupImageLoading();
  setupCanvasInteraction();
  setupDatasetGeneration();
  setupTraining();
  setupClassifyAndEval();
  setupDenseMap();
  refreshWorkspaceStatus();
});

// 1. Tabs
function setupTabs() {
  el.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      el.tabs.forEach((t) => t.classList.remove('active'));
      el.tabContents.forEach((c) => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(target).classList.add('active');

      if (target === 'tab-browser') {
        refreshWorkspaceStatus();
      }
    });
  });
}

// 2. Class Management
function setupClassManagement() {
  renderClasses();

  el.btnAddClass.addEventListener('click', () => {
    const rawName = el.newClassName.value.trim();
    if (!rawName) return;
    const name = rawName.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
    const color = el.newClassColor.value;

    if (!state.classes.some((c) => c.name === name)) {
      state.classes.push({ name, color });
      renderClasses();
      el.newClassName.value = '';
    }
  });

  el.btnPresetSentinel.addEventListener('click', () => {
    state.classes = [
      { name: 'floresta_natural', color: '#22c55e' },
      { name: 'vegetacao_campestre', color: '#eab308' },
      { name: 'solo_descoberto', color: '#f97316' },
      { name: 'agua', color: '#3b82f6' },
    ];
    renderClasses();
  });
}

function renderClasses() {
  el.classesList.innerHTML = '';
  el.selectActiveClass.innerHTML = '';

  state.classes.forEach((c) => {
    // Tag
    const tag = document.createElement('div');
    tag.className = 'class-tag';
    tag.innerHTML = `
      <span class="class-dot" style="background-color: ${c.color}"></span>
      <span>${c.name}</span>
      <button class="tag-remove" data-name="${c.name}">&times;</button>
    `;
    tag.querySelector('.tag-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      state.classes = state.classes.filter((item) => item.name !== c.name);
      renderClasses();
      recalculatePatches();
    });
    el.classesList.appendChild(tag);

    // Select option
    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = c.name;
    el.selectActiveClass.appendChild(opt);
  });
}

// 3. Image Loading
function setupImageLoading() {
  el.dropZone.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadImage(e.target.files[0]);
  });

  el.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.dropZone.classList.add('dragover');
  });
  el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
  el.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) uploadImage(e.dataTransfer.files[0]);
  });

  el.chkSentinel10m.addEventListener('change', () => {
    if (el.chkSentinel10m.checked) {
      el.sentinelBadge.className = 'resolution-badge sentinel-native';
      el.sentinelBadge.textContent = '1 PIXEL = 10 m • PATCH 8x8 = 80 x 80 m';
    } else {
      el.sentinelBadge.className = 'resolution-badge display-only';
      el.sentinelBadge.textContent = 'DISPLAY IMAGE — SEM GARANTIA DE 10 m/PIXEL';
    }
  });
}

async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!data.success) {
      alert('Erro ao enviar imagem: ' + data.error);
      return;
    }

    state.imagePath = data.path;
    state.imageMeta = data;

    el.metaFilename.textContent = data.filename;
    el.metaDims.textContent = `${data.width} × ${data.height} px`;
    el.metaSha.textContent = data.sha256;
    el.imageMeta.classList.remove('hidden');

    const img = new Image();
    img.src = `/api/image?path=${encodeURIComponent(data.path)}`;
    img.onload = () => {
      state.currentImage = img;
      el.canvasPlaceholder.classList.add('hidden');
      resizeCanvasToImage();
      state.rois = [];
      state.patches = [];
      recalculatePatches();
      redrawCanvas();
    };
  } catch (err) {
    alert('Erro de conexão ao enviar imagem: ' + err);
  }
}

// 4. Canvas Interaction & Annotation
function resizeCanvasToImage() {
  if (!state.currentImage) return;
  el.canvas.width = state.currentImage.naturalWidth;
  el.canvas.height = state.currentImage.naturalHeight;
  state.zoom = 1.0;
  state.panX = 0;
  state.panY = 0;
  updateCanvasTransform();
}

function updateCanvasTransform() {
  if (!el.canvas) return;
  el.canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  el.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setupCanvasInteraction() {
  el.btnZoomIn.addEventListener('click', () => {
    state.zoom = Math.min(state.zoom * 1.25, 20.0);
    updateCanvasTransform();
  });
  el.btnZoomOut.addEventListener('click', () => {
    state.zoom = Math.max(state.zoom / 1.25, 0.1);
    updateCanvasTransform();
  });
  el.btnZoomReset.addEventListener('click', () => {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    updateCanvasTransform();
  });

  el.chkShowGrid.addEventListener('change', redrawCanvas);
  el.chkShowRois.addEventListener('change', redrawCanvas);
  el.btnClearRois.addEventListener('click', () => {
    state.rois = [];
    state.patches = [];
    recalculatePatches();
    redrawCanvas();
  });

  // Canvas drawing & coordinates
  el.canvasContainer.addEventListener('mousedown', (e) => {
    if (!state.currentImage) return;
    if (e.button !== 0) return; // Only left click

    const coords = getCanvasCoords(e);
    state.isDrawing = true;
    state.drawStartX = coords.x;
    state.drawStartY = coords.y;
    state.currentRect = { x: coords.x, y: coords.y, width: 0, height: 0 };
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.currentImage) return;
    const coords = getCanvasCoords(e);
    el.cursorCoords.textContent = `X: ${Math.round(coords.x)} | Y: ${Math.round(coords.y)}`;

    if (state.isDrawing) {
      const minX = Math.min(state.drawStartX, coords.x);
      const minY = Math.min(state.drawStartY, coords.y);
      const maxX = Math.max(state.drawStartX, coords.x);
      const maxY = Math.max(state.drawStartY, coords.y);

      state.currentRect = {
        x: Math.max(0, Math.floor(minX)),
        y: Math.max(0, Math.floor(minY)),
        width: Math.min(state.currentImage.naturalWidth - minX, Math.floor(maxX - minX)),
        height: Math.min(state.currentImage.naturalHeight - minY, Math.floor(maxY - minY)),
      };
      redrawCanvas();
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.isDrawing && state.currentRect) {
      state.isDrawing = false;
      if (state.currentRect.width >= 8 && state.currentRect.height >= 8) {
        const roi = {
          id: `roi_${Date.now()}_${state.rois.length}`,
          x: state.currentRect.x,
          y: state.currentRect.y,
          width: state.currentRect.width,
          height: state.currentRect.height,
          class: el.selectActiveClass.value || 'unnamed',
          split: el.selectActiveSplit.value || 'train',
        };
        state.rois.push(roi);
        recalculatePatches();
      }
      state.currentRect = null;
      redrawCanvas();
    }
  });
}

function getCanvasCoords(e) {
  const rect = el.canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (el.canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (el.canvas.height / rect.height);
  return { x, y };
}

function recalculatePatches() {
  state.patches = [];
  const stride = 8;

  state.rois.forEach((roi) => {
    const endX = roi.x + roi.width;
    const endY = roi.y + roi.height;

    for (let py = roi.y; py + 8 <= endY; py += stride) {
      for (let px = roi.x; px + 8 <= endX; px += stride) {
        state.patches.push({
          id: `patch_${px}_${py}`,
          roi_id: roi.id,
          x: px,
          y: py,
          width: 8,
          height: 8,
          class: roi.class,
          split: roi.split,
          selected: true,
        });
      }
    }
  });

  updateSummaryTable();
  el.roiCount.textContent = `${state.rois.length} ROIs`;
  el.canvasPatchInfo.textContent = `Patches selecionados: ${state.patches.filter((p) => p.selected).length}`;
}

function redrawCanvas() {
  if (!ctx || !state.currentImage) return;

  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
  ctx.drawImage(state.currentImage, 0, 0);

  // Draw ROIs
  if (el.chkShowRois.checked) {
    state.rois.forEach((roi) => {
      const cls = state.classes.find((c) => c.name === roi.class);
      const color = cls ? cls.color : '#3b82f6';

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);

      ctx.fillStyle = color + '33';
      ctx.fillRect(roi.x, roi.y, roi.width, roi.height);

      // Label tag
      ctx.fillStyle = color;
      ctx.fillRect(roi.x, roi.y - 16, Math.max(60, roi.class.length * 8 + 45), 16);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(`${roi.class} [${roi.split.toUpperCase()}]`, roi.x + 4, roi.y - 4);
    });
  }

  // Draw 8x8 Patch Grid
  if (el.chkShowGrid.checked) {
    state.patches.forEach((p) => {
      const cls = state.classes.find((c) => c.name === p.class);
      const color = cls ? cls.color : '#ffffff';

      ctx.strokeStyle = p.selected ? color : 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(p.x + 0.5, p.y + 0.5, 7, 7);
    });
  }

  // Draw currently drawing rectangle
  if (state.currentRect) {
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(state.currentRect.x, state.currentRect.y, state.currentRect.width, state.currentRect.height);
    ctx.setLineDash([]);
  }
}

// 5. Summary & Dataset Generation
function updateSummaryTable() {
  const counts = {};
  state.classes.forEach((c) => {
    counts[c.name] = { train: 0, dev: 0, probe: 0 };
  });

  state.patches
    .filter((p) => p.selected)
    .forEach((p) => {
      if (!counts[p.class]) counts[p.class] = { train: 0, dev: 0, probe: 0 };
      counts[p.class][p.split] = (counts[p.class][p.split] || 0) + 1;
    });

  const rows = Object.entries(counts);
  if (rows.length === 0 || state.patches.length === 0) {
    el.summaryTableBody.innerHTML = '<tr><td colspan="5" class="empty-msg">Nenhum patch selecionado</td></tr>';
    el.btnGenerateDataset.disabled = true;
    el.datasetWarnings.classList.add('hidden');
    return;
  }

  let tableHtml = '';
  let warnings = [];
  let totalAll = 0;
  let completeClasses = 0;

  rows.forEach(([clsName, splits]) => {
    const total = splits.train + splits.dev + splits.probe;
    totalAll += total;

    if (total === 0) {
      warnings.push(`⚠️ Classe <strong>${clsName}</strong> sem nenhum patch.`);
    } else {
      if (splits.train === 0) {
        warnings.push(`⚠️ Classe <strong>${clsName}</strong> sem patches em <strong>TRAIN</strong>.`);
      }
      if (splits.dev === 0) {
        warnings.push(`⚠️ Classe <strong>${clsName}</strong> sem patches em <strong>DEV</strong> (obrigatório para treinamento).`);
      }
      if (splits.train > 0 && splits.dev > 0) {
        completeClasses += 1;
      }
    }

    tableHtml += `
      <tr>
        <td><strong>${clsName}</strong></td>
        <td><span style="color: ${splits.train > 0 ? '#34d399' : '#fb7185'}">${splits.train}</span></td>
        <td><span style="color: ${splits.dev > 0 ? '#34d399' : '#fb7185'}">${splits.dev}</span></td>
        <td><span>${splits.probe}</span></td>
        <td><strong>${total}</strong></td>
      </tr>
    `;
  });

  el.summaryTableBody.innerHTML = tableHtml;
  el.btnGenerateDataset.disabled = totalAll === 0;

  if (completeClasses < 2 && totalAll > 0) {
    warnings.unshift(`❌ <strong>Requisito de Treinamento:</strong> É necessário ter pelo menos 2 classes com amostras em <strong>TRAIN</strong> e em <strong>DEV</strong>. Use o botão <em>Distribuir Splits</em> abaixo se marcou tudo como TRAIN.`);
  }

  if (warnings.length > 0) {
    el.datasetWarnings.innerHTML = warnings.join('<br>');
    el.datasetWarnings.classList.remove('hidden');
  } else {
    el.datasetWarnings.classList.add('hidden');
  }
}

function setupDatasetGeneration() {
  if (el.btnAutoSplit) {
    el.btnAutoSplit.addEventListener('click', () => {
      if (state.rois.length === 0) {
        alert('Nenhuma ROI criada. Desenhe ROIs na imagem primeiro.');
        return;
      }

      // Group ROIs by class
      const roisByClass = {};
      state.rois.forEach((roi) => {
        if (!roisByClass[roi.class]) roisByClass[roi.class] = [];
        roisByClass[roi.class].push(roi);
      });

      let singleRoiClasses = [];
      Object.entries(roisByClass).forEach(([clsName, classRois]) => {
        const total = classRois.length;
        if (total === 1) {
          classRois[0].split = 'train';
          singleRoiClasses.push(clsName);
        } else {
          // Put roughly 25% of ROIs in DEV (at least 1), remaining in TRAIN
          const devCount = Math.max(1, Math.floor(total * 0.25));
          const trainCount = total - devCount;
          classRois.forEach((roi, idx) => {
            roi.split = idx < trainCount ? 'train' : 'dev';
          });
        }
      });

      recalculatePatches();
      redrawCanvas();

      if (singleRoiClasses.length > 0) {
        alert(
          'Aviso de Independência Espacial (1 ROI = 1 Split):\n' +
          `As seguintes classes possuem apenas 1 ROI: ${singleRoiClasses.join(', ')}.\n` +
          'Para criar o split DEV com independência espacial estrita, desenhe pelo menos uma segunda ROI separada no mapa para essas classes.'
        );
      }
    });
  }


  el.btnGenerateDataset.addEventListener('click', async () => {

    const rawName = el.datasetNameInput.value.trim() || `dataset_${Date.now()}`;
    const datasetName = rawName.replace(/[^a-zA-Z0-9_-]/g, '_');

    const selectedPatches = state.patches
      .filter((p) => p.selected)
      .map((p) => ({
        roi_id: p.roi_id,
        class: p.class,
        split: p.split,
        x: p.x,
        y: p.y,
      }));

    if (selectedPatches.length === 0) {
      alert('Nenhum patch selecionado para geração.');
      return;
    }

    try {
      el.btnGenerateDataset.disabled = true;
      el.btnGenerateDataset.textContent = 'Extraindo patches...';

      const res = await fetch('/api/datasets/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          image_path: state.imagePath,
          is_sentinel_10m: el.chkSentinel10m.checked,
          patches: selectedPatches,
        }),
      });

      const data = await res.json();
      if (!data.success) {
        alert('Erro ao criar dataset: ' + data.error);
        return;
      }

      alert(`Dataset '${datasetName}' criado com sucesso!\nTotal de patches 8x8 extraídos: ${data.total_patches}\nSalvo em: ${data.dataset_path}`);
      refreshWorkspaceStatus();
    } catch (err) {
      alert('Erro na requisição: ' + err);
    } finally {
      el.btnGenerateDataset.disabled = false;
      el.btnGenerateDataset.textContent = '⚡ Extrair Patches 8x8 & Gerar Dataset';
    }
  });
}

// 6. Training Execution
function setupTraining() {
  el.btnStartTrain.addEventListener('click', async () => {
    const datasetName = el.trainSelectDataset.value;
    if (!datasetName) {
      alert('Selecione um dataset para treinar.');
      return;
    }

    const rawModelName = el.trainModelName.value.trim() || `${datasetName}_model`;
    const modelName = rawModelName.replace(/[^a-zA-Z0-9_-]/g, '_');

    try {
      el.btnStartTrain.disabled = true;
      el.trainStatusBadge.textContent = 'Treinando...';
      el.trainStatusBadge.style.color = '#38bdf8';
      el.trainTerminalLog.textContent = 'Iniciando execução de ./bin/tinyvision train...\n';

      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          model_name: modelName,
        }),
      });

      const data = await res.json();
      el.trainTerminalLog.textContent = data.log || '';

      if (data.success) {
        el.trainStatusBadge.textContent = 'Concluído';
        el.trainStatusBadge.style.color = '#34d399';
        el.metricTrainAcc.textContent = data.final_train_acc || '-';
        el.metricDevAcc.textContent = data.final_dev_acc || '-';
        el.metricDevLoss.textContent = data.final_dev_loss ? data.final_dev_loss.toFixed(6) : '-';
        el.metricParams.textContent = data.parameter_count || '-';
        refreshWorkspaceStatus();
      } else {
        el.trainStatusBadge.textContent = 'Falha';
        el.trainStatusBadge.style.color = '#fb7185';
        alert('Treinamento falhou: ' + (data.error || 'Verifique o log'));
      }
    } catch (err) {
      alert('Erro ao executar treinamento: ' + err);
    } finally {
      el.btnStartTrain.disabled = false;
    }
  });
}

// 7. Classify & Evaluate
function setupClassifyAndEval() {
  // Classify
  el.btnRunClassify.addEventListener('click', async () => {
    const modelName = el.classifySelectModel.value;
    if (!modelName) {
      alert('Selecione um modelo.');
      return;
    }

    const files = el.classifyImageFile.files;
    if (files.length === 0) {
      alert('Selecione uma imagem PNG/JPEG.');
      return;
    }

    // First upload the test image
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
      const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
      const upData = await upRes.json();
      if (!upData.success) {
        alert('Erro ao enviar imagem: ' + upData.error);
        return;
      }

      const res = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          image_path: upData.path,
        }),
      });

      const data = await res.json();
      if (!data.success) {
        alert('Erro ao classificar: ' + data.error);
        return;
      }

      el.classifyPredClass.textContent = data.predicted;
      el.classifyPredScore.textContent = `Score: ${(data.score * 100).toFixed(2)}%`;
      if (el.classifyPreviewImg) {
        el.classifyPreviewImg.src = URL.createObjectURL(files[0]);
      }


      el.classifyProbsContainer.innerHTML = '';
      Object.entries(data.probabilities).forEach(([clsName, prob]) => {
        const pct = (prob * 100).toFixed(2);
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
          <div class="prob-header">
            <span>${clsName}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${pct}%"></div>
          </div>
        `;
        el.classifyProbsContainer.appendChild(row);
      });

      el.classifyResults.classList.remove('hidden');
    } catch (err) {
      alert('Erro na requisição: ' + err);
    }
  });

  // Evaluate
  el.btnEvalDev.addEventListener('click', () => runEvaluation('dev'));
  el.btnEvalProbe.addEventListener('click', () => runEvaluation('probe'));
}

async function runEvaluation(split) {
  const modelName = el.evalSelectModel.value;
  const datasetName = el.evalSelectDataset.value;

  if (!modelName || !datasetName) {
    alert('Selecione o modelo e o dataset.');
    return;
  }

  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName,
        dataset_name: datasetName,
        split: split,
      }),
    });

    const data = await res.json();
    if (!data.success) {
      alert(`Erro na avaliação (${split}): ` + data.error);
      return;
    }

    el.evalSplitTag.textContent = split.toUpperCase();
    el.evalAcc.textContent = `${data.accuracy.toFixed(2)}%`;
    el.evalSamples.textContent = data.samples;
    el.evalLoss.textContent = data.loss.toFixed(6);
    el.evalMeanProb.textContent = data.mean_true_probability.toFixed(6);

    // Confusion matrix
    if (data.confusion && data.confusion.classes.length > 0) {
      let matrixHtml = '<table class="confusion-table"><thead><tr><th>Real \\ Pred</th>';
      data.confusion.classes.forEach((c) => {
        matrixHtml += `<th>${c}</th>`;
      });
      matrixHtml += '</tr></thead><tbody>';

      data.confusion.matrix.forEach((row, rowIdx) => {
        matrixHtml += `<tr><th>${row.actual}</th>`;
        row.counts.forEach((cnt, colIdx) => {
          const isDiag = rowIdx === colIdx;
          matrixHtml += `<td class="${isDiag ? 'diagonal' : ''}">${cnt}</td>`;
        });
        matrixHtml += '</tr>';
      });
      matrixHtml += '</tbody></table>';
      el.confusionMatrixContainer.innerHTML = matrixHtml;
    } else {
      el.confusionMatrixContainer.innerHTML = '<em>Matriz de confusão indisponível</em>';
    }

    el.evalResults.classList.remove('hidden');
  } catch (err) {
    alert('Erro ao avaliar split: ' + err);
  }
}

// 8. Dense Spatial Classification Engine (C++ Map)
const denseState = {
  runId: null,
  metadata: null,
  sourceImg: null,
  overlayImg: null,
  classMapImg: null,
  confidenceImg: null,
  currentMode: 'overlay', // 'overlay' | 'class_map' | 'confidence' | 'source'
  inspectCache: {},
};

function setupDenseMap() {
  if (!el.btnRunDenseMap) return;

  // Sliders
  el.denseSliderConfidence.addEventListener('input', (e) => {
    el.denseValConfidence.textContent = parseFloat(e.target.value).toFixed(2);
  });

  el.denseSliderMargin.addEventListener('input', (e) => {
    el.denseValMargin.textContent = parseFloat(e.target.value).toFixed(2);
  });

  el.denseSliderOpacity.addEventListener('input', (e) => {
    el.denseOpacityVal.textContent = `${e.target.value}%`;
    drawDenseCanvas();
  });

  el.denseChkSentinel10m.addEventListener('change', (e) => {
    const isSent = e.target.checked;
    el.denseSentinelBadge.textContent = isSent
      ? 'SENTINEL-2 NATIVO 10 m (80x80m Contexto)'
      : 'DISPLAY IMAGE — NO 10 m GUARANTEE';
    el.denseSentinelBadge.className = 'resolution-badge ' + (isSent ? 'sentinel-native' : 'display-only');
  });

  el.denseChkOverlay.addEventListener('change', () => drawDenseCanvas());
  el.denseChkShowUncertain.addEventListener('change', () => drawDenseCanvas());

  // View Mode buttons
  const modeBtns = document.querySelectorAll('.btn-mode');
  modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      modeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      denseState.currentMode = btn.dataset.mode;
      drawDenseCanvas();
    });
  });

  // Dense Upload Image button
  el.btnDenseUpload.addEventListener('click', () => el.denseFileInput.click());
  el.denseFileInput.addEventListener('change', async () => {
    const files = el.denseFileInput.files;
    if (files.length === 0) return;
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        await refreshWorkspaceStatus();
        el.denseSelectImage.value = data.path;
      } else {
        alert('Erro ao carregar imagem: ' + data.error);
      }
    } catch (err) {
      alert('Erro na requisição: ' + (err.message || err));
    }
  });

  // Run Dense Map button
  el.btnRunDenseMap.addEventListener('click', async () => {
    const modelName = el.denseSelectModel.value;
    const imagePath = el.denseSelectImage.value;

    if (!modelName || !imagePath) {
      alert('Selecione um modelo e uma imagem fonte.');
      return;
    }

    try {
      el.btnRunDenseMap.disabled = true;
      el.denseProgress.classList.remove('hidden');

      const res = await fetch('/api/dense_map', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          image_path: imagePath,
          stride: parseInt(el.denseSelectStride.value, 10),
          confidence: parseFloat(el.denseSliderConfidence.value),
          margin: parseFloat(el.denseSliderMargin.value),
          is_sentinel_10m: el.denseChkSentinel10m.checked,
        }),
      });

      const data = await res.json();
      if (!data.success) {
        alert('Erro na classificação em grade: ' + (data.error || 'Erro desconhecido'));
        return;
      }

      denseState.runId = data.run_id;
      denseState.metadata = data.metadata;
      denseState.inspectCache = {};

      // Load all images asynchronously
      const loadImage = (url) =>
        new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Falha ao carregar imagem: ${url}`));
          img.src = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
        });

      [
        denseState.overlayImg,
        denseState.classMapImg,
        denseState.confidenceImg,
        denseState.sourceImg,
      ] = await Promise.all([
        loadImage(data.artifacts.overlay),
        loadImage(data.artifacts.class_map),
        loadImage(data.artifacts.confidence),
        loadImage(`/api/image?path=${encodeURIComponent(imagePath)}`),
      ]);

      // Update download links
      el.linkDownloadCsv.href = data.artifacts.classification_csv;
      el.linkDownloadJson.href = data.artifacts.run_json;
      el.linkDownloadOverlay.href = data.artifacts.overlay;
      el.linkDownloadClassmap.href = data.artifacts.class_map;

      // Render Dynamic Legend
      renderDynamicLegend(data.metadata);

      // Render Canvas
      drawDenseCanvas();

      el.denseResultsPanel.classList.remove('hidden');
    } catch (err) {
      alert('Erro na execução do mapa: ' + (err.message || err));
    } finally {
      el.btnRunDenseMap.disabled = false;
      el.denseProgress.classList.add('hidden');
    }
  });

  // Canvas Mouse Inspection
  el.denseCanvas.addEventListener('mousemove', (e) => inspectPoint(e));
  el.denseCanvas.addEventListener('click', (e) => inspectPoint(e, true));
}

function renderDynamicLegend(meta) {
  if (!el.denseDynamicLegend || !meta) return;
  const classes = meta.classes || [];
  const total = meta.decision_count || 0;
  el.denseTotalDecisions.textContent = `${total.toLocaleString()} decisões`;

  const isSent = meta.sentinel_native_10m;
  el.denseContextInfo.textContent = isSent
    ? `Janela 8x8 (80x80m nominal) • Stride ${meta.stride} (${meta.nominal_decision_spacing_m}m)`
    : `Janela 8x8 (Espaço de pixel da imagem) • Stride ${meta.stride}`;

  const counts = (meta.summary && meta.summary.class_counts) || {};
  const uncertainCount = (meta.summary && meta.summary.uncertain_count) || 0;

  const palette = ['#22c55e', '#eab308', '#f97316', '#3b82f6', '#ec4899', '#a855f7', '#14b8a6'];
  let html = '';
  classes.forEach((cls, idx) => {
    const color = palette[idx % palette.length];
    const cCount = counts[cls] || 0;
    const pct = total > 0 ? ((cCount / total) * 100).toFixed(1) : '0.0';
    html += `
      <div class="legend-item">
        <div style="display:flex;align-items:center;">
          <span class="legend-color-dot" style="background-color: ${color};"></span>
          <strong>${cls}</strong>
        </div>
        <span class="mono">${cCount.toLocaleString()} (${pct}%)</span>
      </div>
    `;
  });

  // UNCERTAIN item
  const uncPct = total > 0 ? ((uncertainCount / total) * 100).toFixed(1) : '0.0';
  html += `
    <div class="legend-item" style="border-left: 3px solid #808080;">
      <div style="display:flex;align-items:center;">
        <span class="legend-color-dot" style="background-color: #808080;"></span>
        <em>UNCERTAIN (Incerteza)</em>
      </div>
      <span class="mono">${uncertainCount.toLocaleString()} (${uncPct}%)</span>
    </div>
  `;

  el.denseDynamicLegend.innerHTML = html;
}

function drawDenseCanvas() {
  const canvas = el.denseCanvas;
  if (!canvas || !denseState.sourceImg) return;

  const dctx = canvas.getContext('2d');
  const w = denseState.sourceImg.naturalWidth || denseState.sourceImg.width;
  const h = denseState.sourceImg.naturalHeight || denseState.sourceImg.height;

  canvas.width = w;
  canvas.height = h;
  dctx.clearRect(0, 0, w, h);

  const mode = denseState.currentMode;
  if (mode === 'source') {
    dctx.drawImage(denseState.sourceImg, 0, 0);
  } else if (mode === 'class_map') {
    dctx.drawImage(denseState.classMapImg, 0, 0);
  } else if (mode === 'confidence') {
    dctx.drawImage(denseState.confidenceImg, 0, 0);
  } else if (mode === 'overlay') {
    dctx.drawImage(denseState.sourceImg, 0, 0);
    if (el.denseChkOverlay.checked && denseState.classMapImg) {
      const alpha = parseInt(el.denseSliderOpacity.value, 10) / 100.0;
      dctx.globalAlpha = alpha;
      dctx.drawImage(denseState.classMapImg, 0, 0);
      dctx.globalAlpha = 1.0;
    }
  }
}

function updateInspectorUI(decision, originX, originY, centerX, centerY, displayX, displayY, e) {
  if (!decision) return;
  el.inspOrigin.textContent = `(${originX}, ${originY})`;
  el.inspCenter.textContent = `(${centerX.toFixed(1)}, ${centerY.toFixed(1)})`;
  el.inspDisplay.textContent = `(${displayX}, ${displayY})`;
  el.inspSupport.textContent = denseState.metadata.sentinel_native_10m
    ? '8x8 px (80x80 m nominal)'
    : '8x8 px (Espaço de pixel)';

  const isUncertain = decision.status === 'UNCERTAIN';
  el.inspStatus.textContent = decision.status;
  el.inspStatus.className = 'badge ' + (isUncertain ? 'badge-uncertain' : 'badge-classified');

  el.inspTop1Class.textContent = decision.predicted_class;
  el.inspTop1Prob.textContent = `${(parseFloat(decision.probability) * 100).toFixed(2)}%`;
  el.inspTop2Class.textContent = decision.second_class || '-';
  el.inspTop2Prob.textContent = decision.second_probability
    ? `${(parseFloat(decision.second_probability) * 100).toFixed(2)}%`
    : '-';
  el.inspMargin.textContent = decision.margin
    ? `${(parseFloat(decision.margin) * 100).toFixed(2)}%`
    : '-';

  // Position crosshair
  if (el.denseCrosshair && e) {
    const canvasWrapper = el.denseCanvasContainer;
    const wrapRect = canvasWrapper.getBoundingClientRect();
    const crossX = e.clientX - wrapRect.left;
    const crossY = e.clientY - wrapRect.top;
    el.denseCrosshair.style.left = `${crossX}px`;
    el.denseCrosshair.style.top = `${crossY}px`;
    el.denseCrosshair.classList.remove('hidden');
  }
}

let inspectThrottle = null;
async function inspectPoint(e, isClick = false) {
  const canvas = el.denseCanvas;
  if (!canvas || !denseState.metadata) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const clickX = Math.floor((e.clientX - rect.left) * scaleX);
  const clickY = Math.floor((e.clientY - rect.top) * scaleY);

  const stride = denseState.metadata.stride || 1;
  const w = denseState.metadata.source_width || canvas.width;
  const h = denseState.metadata.source_height || canvas.height;

  // Grid origins
  const originX = Math.floor(clickX / stride) * stride;
  const originY = Math.floor(clickY / stride) * stride;

  const nx = Math.floor((w - 8) / stride) + 1;
  const ny = Math.floor((h - 8) / stride) + 1;

  const gx = Math.floor(originX / stride);
  const gy = Math.floor(originY / stride);

  if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) return;

  const centerX = originX + 3.5;
  const centerY = originY + 3.5;
  const displayX = originX + 4;
  const displayY = originY + 4;

  const cacheKey = `${originX}_${originY}`;
  if (denseState.inspectCache[cacheKey]) {
    updateInspectorUI(denseState.inspectCache[cacheKey], originX, originY, centerX, centerY, displayX, displayY, e);
    return;
  }

  if (inspectThrottle) return;
  inspectThrottle = setTimeout(() => { inspectThrottle = null; }, 50);

  if (denseState.runId) {
    try {
      const res = await fetch(`/api/runs/${denseState.runId}/inspect?x=${originX}&y=${originY}`);
      const data = await res.json();
      if (data.success && data.decision) {
        denseState.inspectCache[cacheKey] = data.decision;
        updateInspectorUI(data.decision, originX, originY, centerX, centerY, displayX, displayY, e);
      }
    } catch (err) {
      console.warn('Inspect fetch failed', err);
    }
  }
}

// 9. Refresh Workspace Status
async function refreshWorkspaceStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!data.success) return;

    state.datasets = data.datasets || [];
    state.models = data.models || [];
    const uploads = data.uploads || [];

    // Populate dataset selects
    const datasetOpts = state.datasets
      .map((d) => `<option value="${d.dataset_id}">${d.dataset_id} (${d.total_patches} patches)</option>`)
      .join('');
    el.trainSelectDataset.innerHTML = datasetOpts;
    el.evalSelectDataset.innerHTML = datasetOpts;

    // Populate model selects
    const modelOpts = state.models
      .map((m) => `<option value="${m.name}">${m.name} (${Math.round(m.size_bytes / 1024)} KB)</option>`)
      .join('');
    el.classifySelectModel.innerHTML = modelOpts;
    el.evalSelectModel.innerHTML = modelOpts;
    if (el.denseSelectModel) {
      el.denseSelectModel.innerHTML = modelOpts;
    }

    // Populate dense image select
    if (el.denseSelectImage) {
      const imgOpts = uploads
        .map((u) => `<option value="${u.path}">${u.name} (${u.width}x${u.height})</option>`)
        .join('');
      el.denseSelectImage.innerHTML = imgOpts;
    }

    // Browser lists
    el.browserDatasetsList.innerHTML = state.datasets.length
      ? state.datasets
          .map(
            (d) => `
        <div class="item-card">
          <div class="item-title">${d.dataset_id}</div>
          <div class="item-sub">Total: ${d.total_patches} patches • Fonte: ${d.source_image.name} • ${d.is_sentinel_10m ? 'Sentinel 10m' : 'Display-only'}</div>
        </div>
      `
          )
          .join('')
      : '<em>Nenhum dataset gerado em .tinyvision/datasets/</em>';

    el.browserModelsList.innerHTML = state.models.length
      ? state.models
          .map(
            (m) => `
        <div class="item-card">
          <div class="item-title">${m.name}</div>
          <div class="item-sub">${m.path}</div>
        </div>
      `
          )
          .join('')
      : '<em>Nenhum modelo salvo em .tinyvision/models/</em>';
  } catch (err) {
    console.error('Failed to refresh status', err);
  }
}
