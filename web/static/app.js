/**
 * TinyLogicVision Web GUI Application Logic v2.0
 * Supports RGB, Multichannel Sentinel-2 10m (B2,B3,B4,B8), GeoTIFF, H3 Indexing, RIT Provenance & Async Jobs
 */

// State
const state = {
  modality: 'RGB', // 'RGB' or 'SENTINEL2_MULTIBAND'
  currentImage: null,
  imagePath: null,
  imageMeta: null,
  classes: [
    { name: 'floresta_natural', color: '#22c55e' },
    { name: 'vegetacao_campestre', color: '#eab308' },
    { name: 'solo_descoberto', color: '#f97316' },
  ],
  rois: [], // { id, x, y, width, height, class, split }
  nextRoiId: 1,
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
  activeJobId: null,
  jobsPollingInterval: null,
};

// Elements
const el = {
  tabs: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  navJobsBadge: document.getElementById('nav-jobs-badge'),
  activeWorkspacePath: document.getElementById('active-workspace-path'),
  dropZone: document.getElementById('drop-zone'),
  fileInput: document.getElementById('image-file-input'),
  uploadPromptText: document.getElementById('upload-prompt-text'),
  imageMeta: document.getElementById('image-meta'),
  metaFilename: document.getElementById('meta-filename'),
  metaDims: document.getElementById('meta-dims'),
  metaSchemaModality: document.getElementById('meta-schema-modality'),
  metaSha: document.getElementById('meta-sha'),
  bandsSchemaBox: document.getElementById('bands-schema-box'),
  geoMetaCard: document.getElementById('geo-meta-card'),
  metaGeoCrs: document.getElementById('meta-geo-crs'),
  metaGeoPixel: document.getElementById('meta-geo-pixel'),
  chkSentinel10m: document.getElementById('chk-sentinel-10m'),
  sentinelBadge: document.getElementById('sentinel-badge'),
  modalityRgbCard: document.getElementById('modality-rgb-card'),
  modalityS2Card: document.getElementById('modality-s2-card'),
  newClassName: document.getElementById('new-class-name'),
  newClassColor: document.getElementById('new-class-color'),
  btnAddClass: document.getElementById('btn-add-class'),
  btnPresetSentinel: document.getElementById('btn-preset-sentinel'),
  classesList: document.getElementById('classes-list'),
  selectActiveClass: document.getElementById('select-active-class'),
  selectActiveSplit: document.getElementById('select-active-split'),
  chkH3SplitPartition: document.getElementById('chk-h3-split-partition'),
  h3SplitControls: document.getElementById('h3-split-controls'),
  selectH3Resolution: document.getElementById('select-h3-resolution'),
  btnClearRois: document.getElementById('btn-clear-rois'),
  roiCount: document.getElementById('roi-count'),
  roiList: document.getElementById('roi-list'),
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
  chkShowH3Overlay: document.getElementById('chk-show-h3-overlay'),
  cursorCoords: document.getElementById('cursor-coords'),
  cursorGeoCoords: document.getElementById('cursor-geo-coords'),
  canvasPatchInfo: document.getElementById('canvas-patch-info'),
  // Train
  trainSelectDataset: document.getElementById('train-select-dataset'),
  trainModelName: document.getElementById('train-model-name'),
  chkTrainAsync: document.getElementById('chk-train-async'),
  btnStartTrain: document.getElementById('btn-start-train'),
  trainStatusBadge: document.getElementById('train-status-badge'),
  metricTrainAcc: document.getElementById('metric-train-acc'),
  metricDevAcc: document.getElementById('metric-dev-acc'),
  metricDevLoss: document.getElementById('metric-dev-loss'),
  metricParams: document.getElementById('metric-params'),
  trainTerminalLog: document.getElementById('train-terminal-log'),
  // Classify & Eval & Dense
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
  denseSelectModel: document.getElementById('dense-select-model'),
  denseSelectImage: document.getElementById('dense-select-image'),
  denseFileInput: document.getElementById('dense-file-input'),
  btnDenseUpload: document.getElementById('btn-dense-upload'),
  denseSelectStride: document.getElementById('dense-select-stride'),
  denseSelectThreads: document.getElementById('dense-select-threads'),
  denseSelectTileSize: document.getElementById('dense-select-tile-size'),
  denseSchemaCompatibility: document.getElementById('dense-schema-compatibility'),
  denseModelSchema: document.getElementById('dense-model-schema'),
  denseSourceSchema: document.getElementById('dense-source-schema'),
  denseSchemaStatus: document.getElementById('dense-schema-status'),
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
  denseZoomOut: document.getElementById('dense-zoom-out'),
  denseZoomIn: document.getElementById('dense-zoom-in'),
  denseZoomFit: document.getElementById('dense-zoom-fit'),
  denseZoomFill: document.getElementById('dense-zoom-fill'),
  denseZoomActual: document.getElementById('dense-zoom-actual'),
  denseZoomLevel: document.getElementById('dense-zoom-level'),
  denseToggleExpand: document.getElementById('dense-toggle-expand'),
  inspGrid: document.getElementById('insp-grid'),
  inspOrigin: document.getElementById('insp-origin'),
  inspCenter: document.getElementById('insp-center'),
  inspMapCoords: document.getElementById('insp-map-coords'),
  inspLatLon: document.getElementById('insp-latlon'),
  inspH3Index: document.getElementById('insp-h3-index'),
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
  linkDownloadConfidencemap: document.getElementById('link-download-confidencemap'),
  linkDownloadMarginmap: document.getElementById('link-download-marginmap'),
  linkDownloadGeotiffClass: document.getElementById('link-download-geotiff-class'),
  linkDownloadGeotiffConf: document.getElementById('link-download-geotiff-conf'),
  linkDownloadH3Csv: document.getElementById('link-download-h3-csv'),
  linkDownloadProvenance: document.getElementById('link-download-provenance'),
  // Jobs
  btnRefreshJobs: document.getElementById('btn-refresh-jobs'),
  quickJobType: document.getElementById('quick-job-type'),
  btnSubmitQuickJob: document.getElementById('btn-submit-quick-job'),
  jobsCountBadge: document.getElementById('jobs-count-badge'),
  jobsListContainer: document.getElementById('jobs-list-container'),
  jobActiveDetail: document.getElementById('job-active-detail'),
  activeJobId: document.getElementById('active-job-id'),
  activeJobStatus: document.getElementById('active-job-status'),
  activeJobLog: document.getElementById('active-job-log'),
  // Browser & RIT
  browserDatasetsCount: document.getElementById('browser-datasets-count'),
  browserDatasetsList: document.getElementById('browser-datasets-list'),
  browserModelsCount: document.getElementById('browser-models-count'),
  browserModelsList: document.getElementById('browser-models-list'),
  ritNodesList: document.getElementById('rit-nodes-list'),
};

const ctx = el.canvas ? el.canvas.getContext('2d') : null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupModalitySelection();
  setupClassManagement();
  setupImageLoading();
  setupCanvasInteraction();
  setupDatasetGeneration();
  setupTraining();
  setupClassifyAndEval();
  setupDenseMap();
  setupJobsMonitor();
  refreshWorkspaceStatus();
});

// 1. Navigation Tabs
function setupTabs() {
  el.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      el.tabs.forEach((t) => t.classList.remove('active'));
      el.tabContents.forEach((c) => c.classList.remove('active'));
      tab.classList.add('active');
      const targetEl = document.getElementById(target);
      if (targetEl) targetEl.classList.add('active');

      if (target === 'tab-browser') {
        refreshWorkspaceStatus();
        renderRitProvenance();
      } else if (target === 'tab-jobs') {
        fetchAndRenderJobs();
      }
    });
  });
}

// 2. Modality & Multichannel Setup
function setupModalitySelection() {
  const radios = document.querySelectorAll('input[name="input-modality"]');
  const tab1SourceTabs = document.getElementById('tab1-source-mode-tabs');
  const panelFolder = document.getElementById('src-mode-folder');
  const panelMultiband = document.getElementById('src-mode-multiband');
  const panelSingle = document.getElementById('src-mode-single');

  const switchTab = (mode) => {
    if (tab1SourceTabs) {
      tab1SourceTabs.querySelectorAll('.tab1-src-btn').forEach((b) => {
        if (b.dataset.srcMode === mode) b.classList.add('active');
        else b.classList.remove('active');
      });
    }
    if (panelFolder) {
      if (mode === 'folder') { panelFolder.classList.remove('hidden'); panelFolder.style.display = 'flex'; }
      else { panelFolder.classList.add('hidden'); panelFolder.style.display = 'none'; }
    }
    if (panelMultiband) {
      if (mode === 'multiband') { panelMultiband.classList.remove('hidden'); panelMultiband.style.display = 'flex'; }
      else { panelMultiband.classList.add('hidden'); panelMultiband.style.display = 'none'; }
    }
    if (panelSingle) {
      if (mode === 'single') { panelSingle.classList.remove('hidden'); panelSingle.style.display = 'flex'; }
      else { panelSingle.classList.add('hidden'); panelSingle.style.display = 'none'; }
    }
  };

  radios.forEach((r) => {
    r.addEventListener('change', (e) => {
      state.modality = e.target.value;
      if (state.modality === 'SENTINEL2_MULTIBAND') {
        el.modalityS2Card.classList.add('active');
        el.modalityRgbCard.classList.remove('active');
        el.bandsSchemaBox.classList.remove('hidden');
        el.metaSchemaModality.textContent = 'SENTINEL2_MULTIBAND (8x8x4)';
        el.chkSentinel10m.checked = true;
        el.denseChkSentinel10m.checked = true;
        updateSentinelBadges();
        switchTab('folder');
      } else {
        el.modalityRgbCard.classList.add('active');
        el.modalityS2Card.classList.remove('active');
        el.bandsSchemaBox.classList.add('hidden');
        el.metaSchemaModality.textContent = 'RGB (8x8x3)';
        switchTab('single');
      }
    });
  });

  if (el.chkH3SplitPartition) {
    el.chkH3SplitPartition.addEventListener('change', (e) => {
      if (e.target.checked) {
        el.h3SplitControls.classList.remove('hidden');
      } else {
        el.h3SplitControls.classList.add('hidden');
      }
      redrawMainCanvas();
    });
  }

  if (el.chkShowH3Overlay) {
    el.chkShowH3Overlay.addEventListener('change', () => {
      redrawMainCanvas();
    });
  }
}

// 3. Class Management
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
      { name: 'agua_corpos_hidricos', color: '#3b82f6' },
    ];
    renderClasses();
  });
}

function renderClasses() {
  el.classesList.innerHTML = '';
  el.selectActiveClass.innerHTML = '';

  state.classes.forEach((c, idx) => {
    const tag = document.createElement('div');
    tag.className = 'class-tag';
    tag.style.borderColor = c.color;
    tag.innerHTML = `
      <span class="class-dot" style="background:${c.color}"></span>
      <span class="class-name">${c.name}</span>
      <button class="btn-remove-class" title="Remover">&times;</button>
    `;
    tag.querySelector('.btn-remove-class').addEventListener('click', () => {
      state.classes.splice(idx, 1);
      renderClasses();
      updateSummaryTable();
    });
    el.classesList.appendChild(tag);

    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = c.name;
    el.selectActiveClass.appendChild(opt);
  });

  updateSummaryTable();
}

// 4. Image & Sentinel Source Loading
function setupImageLoading() {
  // Source Mode Tabs (Folder vs 4-Bands vs Single)
  const modeBtns = document.querySelectorAll('#tab1-source-mode-tabs .tab1-src-btn, #tab1-source-mode-tabs button');
  const modePanels = {
    folder: document.getElementById('src-mode-folder'),
    multiband: document.getElementById('src-mode-multiband'),
    single: document.getElementById('src-mode-single'),
  };

  modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      modeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const targetMode = btn.dataset.srcMode;
      Object.entries(modePanels).forEach(([k, panel]) => {
        if (panel) {
          if (k === targetMode) {
            panel.classList.remove('hidden');
            panel.style.display = 'flex';
          } else {
            panel.classList.add('hidden');
            panel.style.display = 'none';
          }
        }
      });

      const s2Radio = document.querySelector('input[name="input-modality"][value="SENTINEL2_MULTIBAND"]');
      const rgbRadio = document.querySelector('input[name="input-modality"][value="RGB"]');
      if (targetMode === 'folder' || targetMode === 'multiband') {
        if (s2Radio) s2Radio.checked = true;
        state.modality = 'SENTINEL2_MULTIBAND';
        if (el.modalityS2Card) el.modalityS2Card.classList.add('active');
        if (el.modalityRgbCard) el.modalityRgbCard.classList.remove('active');
        if (el.bandsSchemaBox) el.bandsSchemaBox.classList.remove('hidden');
        if (el.metaSchemaModality) el.metaSchemaModality.textContent = 'SENTINEL2_MULTIBAND (8x8x4)';
        if (el.chkSentinel10m) el.chkSentinel10m.checked = true;
        if (el.denseChkSentinel10m) el.denseChkSentinel10m.checked = true;
        updateSentinelBadges();
      } else if (targetMode === 'single') {
        if (rgbRadio) rgbRadio.checked = true;
        state.modality = 'RGB';
        if (el.modalityRgbCard) el.modalityRgbCard.classList.add('active');
        if (el.modalityS2Card) el.modalityS2Card.classList.remove('active');
        if (el.bandsSchemaBox) el.bandsSchemaBox.classList.add('hidden');
        if (el.metaSchemaModality) el.metaSchemaModality.textContent = 'RGB (8x8x3)';
      }
    });
  });

  // Mode 1: Sentinel Folder (.SAFE / R10m)
  const s2FolderInput = document.getElementById('s2-folder-input');
  const btnSelectS2Folder = document.getElementById('btn-select-s2-folder');
  const s2FolderPathInput = document.getElementById('s2-folder-path-input');
  const btnOpenS2Path = document.getElementById('btn-open-s2-path');

  if (btnSelectS2Folder && s2FolderInput) {
    btnSelectS2Folder.addEventListener('click', () => s2FolderInput.click());
    s2FolderInput.addEventListener('change', async (e) => {
      const allFiles = Array.from(e.target.files);
      if (allFiles.length === 0) return;

      btnSelectS2Folder.textContent = 'Localizando bandas 10m...';
      btnSelectS2Folder.disabled = true;

      try {
        // Filter specifically for 10m bands (B02, B03, B04, B08, TCI) and ignore 20m/60m
        const isNot20_60 = (f) => !/(20m|60m)/i.test(f.webkitRelativePath || f.name);
        const b2File = allFiles.find((f) => /(^|[_.-])(B02|B2|B02_10m|B2_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b3File = allFiles.find((f) => /(^|[_.-])(B03|B3|B03_10m|B3_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b4File = allFiles.find((f) => /(^|[_.-])(B04|B4|B04_10m|B4_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b8File = allFiles.find((f) => /(^|[_.-])(B08|B8|B08_10m|B8_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const tciFile = allFiles.find((f) => /(^|[_.-])(TCI|TCI_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));

        if (!b2File || !b3File || !b4File || !b8File) {
          throw new Error('Não foram encontradas todas as 4 bandas de 10m (B02, B03, B04, B08) na pasta selecionada.');
        }

        btnSelectS2Folder.textContent = 'Enviando 4 bandas 10m...';
        const formData = new FormData();
        formData.append('file_b2', b2File, b2File.name);
        formData.append('file_b3', b3File, b3File.name);
        formData.append('file_b4', b4File, b4File.name);
        formData.append('file_b8', b8File, b8File.name);
        if (tciFile) formData.append('file_tci', tciFile, tciFile.name);

        const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const upData = await upRes.json();
        if (!upData.success) throw new Error(upData.error || 'Erro no upload das bandas');

        // Automatically discover from the uploaded band files
        const savedFiles = upData.files || [upData];
        const findSaved = (re) => {
          const m = savedFiles.find((f) => re.test(f.original_name || f.filename));
          return m ? m.path : null;
        };

        const p2 = findSaved(/(^|[_.-])(B02|B2|B02_10m|B2_10m)\./i);
        const p3 = findSaved(/(^|[_.-])(B03|B3|B03_10m|B3_10m)\./i);
        const p4 = findSaved(/(^|[_.-])(B04|B4|B04_10m|B4_10m)\./i);
        const p8 = findSaved(/(^|[_.-])(B08|B8|B08_10m|B8_10m)\./i);

        if (p2 && p3 && p4 && p8) {
          const sRes = await fetch('/api/sentinel/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bands: { b2: p2, b3: p3, b4: p4, b8: p8 } }),
          });
          const sData = await sRes.json();
          if (sData.success) {
            applySentinelDescriptor(sData);
            return;
          }
        }
        handleLoadedSource(upData);
      } catch (err) {
        alert(`Erro ao abrir pasta Sentinel: ${err.message}`);
      } finally {
        btnSelectS2Folder.textContent = '📂 Selecionar Pasta do Computador';
        btnSelectS2Folder.disabled = false;
      }
    });
  }

  if (btnOpenS2Path && s2FolderPathInput) {
    btnOpenS2Path.addEventListener('click', async () => {
      const folderPath = s2FolderPathInput.value.trim();
      if (!folderPath) {
        alert('Digite ou cole o caminho da pasta Sentinel (.SAFE ou R10m)');
        return;
      }
      btnOpenS2Path.disabled = true;
      btnOpenS2Path.textContent = 'Abrindo...';
      try {
        const res = await fetch('/api/sentinel/open', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder_path: folderPath }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Erro ao abrir pasta');
        applySentinelDescriptor(data);
      } catch (err) {
        alert(`Erro ao abrir pasta Sentinel: ${err.message}`);
      } finally {
        btnOpenS2Path.disabled = false;
        btnOpenS2Path.textContent = 'Abrir';
      }
    });
  }

  // Mode 2: 4 Individual Bands (B2, B3, B4, B8)
  const btnLoad4Bands = document.getElementById('btn-load-4-bands');
  const b2Input = document.getElementById('band-file-b2');
  const b3Input = document.getElementById('band-file-b3');
  const b4Input = document.getElementById('band-file-b4');
  const b8Input = document.getElementById('band-file-b8');
  const b2Path = document.getElementById('band-path-b2');
  const b3Path = document.getElementById('band-path-b3');
  const b4Path = document.getElementById('band-path-b4');
  const b8Path = document.getElementById('band-path-b8');

  if (btnLoad4Bands) {
    btnLoad4Bands.addEventListener('click', async () => {
      btnLoad4Bands.disabled = true;
      btnLoad4Bands.textContent = 'Carregando 4 bandas...';
      try {
        const uploadFileIfSelected = async (input, pathInput) => {
          if (input.files && input.files[0]) {
            const fd = new FormData();
            fd.append('file', input.files[0]);
            const r = await fetch('/api/upload', { method: 'POST', body: fd });
            const d = await r.json();
            if (!d.success) throw new Error('Falha no upload da banda');
            return d.path;
          }
          return pathInput.value.trim();
        };

        const p2 = await uploadFileIfSelected(b2Input, b2Path);
        const p3 = await uploadFileIfSelected(b3Input, b3Path);
        const p4 = await uploadFileIfSelected(b4Input, b4Path);
        const p8 = await uploadFileIfSelected(b8Input, b8Path);

        if (!p2 || !p3 || !p4 || !p8) {
          throw new Error('Forneça os arquivos ou caminhos para as 4 bandas (B2, B3, B4, B8)');
        }

        const res = await fetch('/api/sentinel/open', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bands: { b2: p2, b3: p3, b4: p4, b8: p8 } }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Erro ao processar as 4 bandas');

        applySentinelDescriptor(data);
      } catch (err) {
        alert(`Erro nas 4 bandas: ${err.message}`);
      } finally {
        btnLoad4Bands.disabled = false;
        btnLoad4Bands.textContent = '⚡ Carregar Conjunto de 4 Bandas Sentinel 10m';
      }
    });
  }

  // Mode 3: Single File (Drop zone + Local Path)
  el.dropZone.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImageUpload(file);
  });

  const btnOpenSinglePath = document.getElementById('btn-open-single-path');
  const singleFilePathInput = document.getElementById('single-file-path-input');
  if (btnOpenSinglePath && singleFilePathInput) {
    btnOpenSinglePath.addEventListener('click', async () => {
      const p = singleFilePathInput.value.trim();
      if (!p) return;
      try {
        const res = await fetch('/api/source/open_local', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: p }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Erro ao abrir caminho');
        handleLoadedSource(data);
      } catch (err) {
        alert(err.message);
      }
    });
  }

  el.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.dropZone.classList.add('dragover');
  });
  el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
  el.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleImageUpload(file);
  });

  // Composition mode switcher (True Color / False Color NIR / False Color Green / NDVI)
  const selectViewComp = document.getElementById('select-view-composition');
  if (selectViewComp) {
    selectViewComp.addEventListener('change', (e) => {
      const mode = e.target.value;
      if (state.sentinelDescriptor && state.sentinelDescriptor.previews && state.sentinelDescriptor.previews[mode]) {
        const url = state.sentinelDescriptor.previews[mode];
        const img = new Image();
        img.onload = () => {
          state.currentImage = img;
          redrawMainCanvas();
        };
        img.src = url;
      }
    });
  }

  el.chkSentinel10m.addEventListener('change', () => updateSentinelBadges());
  el.denseChkSentinel10m.addEventListener('change', () => updateSentinelBadges());
}

function applySentinelDescriptor(data) {
  state.modality = 'SENTINEL2_MULTIBAND';
  state.imagePath = data.preview_path;
  state.imageMeta = data;
  state.sentinelDescriptor = data;

  el.modalityS2Card.classList.add('active');
  el.modalityRgbCard.classList.remove('active');
  el.metaFilename.textContent = `Sentinel-2 10m Multibanda (4 Bandas: B2, B3, B4, B8)`;
  el.metaDims.textContent = `${data.width} × ${data.height} px`;
  el.metaSchemaModality.textContent = 'SENTINEL2_MULTIBAND (8x8x4)';
  el.metaSha.textContent = '4_BANDS_ALIGNED_10M';
  el.imageMeta.classList.remove('hidden');
  el.bandsSchemaBox.classList.remove('hidden');

  // Update table band descriptions
  if (data.bands) {
    const descB2 = document.getElementById('band-desc-b2');
    const descB3 = document.getElementById('band-desc-b3');
    const descB4 = document.getElementById('band-desc-b4');
    const descB8 = document.getElementById('band-desc-b8');
    if (descB2 && data.bands.b2) descB2.textContent = `${data.bands.b2.filename} (${(data.bands.b2.size_bytes / 1024 / 1024).toFixed(1)} MB)`;
    if (descB3 && data.bands.b3) descB3.textContent = `${data.bands.b3.filename} (${(data.bands.b3.size_bytes / 1024 / 1024).toFixed(1)} MB)`;
    if (descB4 && data.bands.b4) descB4.textContent = `${data.bands.b4.filename} (${(data.bands.b4.size_bytes / 1024 / 1024).toFixed(1)} MB)`;
    if (descB8 && data.bands.b8) descB8.textContent = `${data.bands.b8.filename} (${(data.bands.b8.size_bytes / 1024 / 1024).toFixed(1)} MB)`;
  }

  // Geo metadata
  if (data.crs) {
    el.geoMetaCard.classList.remove('hidden');
    el.metaGeoCrs.textContent = data.crs;
    el.metaGeoPixel.textContent = `${(data.pixel_size_m || 10).toFixed(2)} m × ${(data.pixel_size_m || 10).toFixed(2)} m`;
  } else {
    el.geoMetaCard.classList.add('hidden');
  }
  el.chkSentinel10m.checked = true;
  el.denseChkSentinel10m.checked = true;
  updateSentinelBadges();

  // Load preview into canvas and automatically fit to window
  const img = new Image();
  img.onload = () => {
    state.currentImage = img;
    el.canvasPlaceholder.classList.add('hidden');
    el.canvas.width = img.width;
    el.canvas.height = img.height;
    fitToScreen();
    redrawMainCanvas();
  };
  img.src = data.preview_url || `/api/image?path=${encodeURIComponent(data.preview_path)}`;

  selectDenseActiveSource('Sentinel-2 B2/B3/B4/B8');
  refreshWorkspaceStatus();
  updateDenseSchemaCompatibility();
}

function handleLoadedSource(data) {
  state.modality = 'RGB';
  state.sentinelDescriptor = null;
  state.imagePath = data.path;
  state.imageMeta = data;

  el.metaFilename.textContent = data.filename;
  el.metaDims.textContent = `${data.width} × ${data.height} px`;
  el.metaSha.textContent = data.sha256;
  el.imageMeta.classList.remove('hidden');

  if (data.crs) {
    el.geoMetaCard.classList.remove('hidden');
    el.metaGeoCrs.textContent = data.crs;
    el.metaGeoPixel.textContent = data.pixel_size_m ? `${data.pixel_size_m.toFixed(2)} m × ${data.pixel_size_m.toFixed(2)} m` : 'NOT_AVAILABLE';
    el.chkSentinel10m.checked = true;
    el.denseChkSentinel10m.checked = true;
    updateSentinelBadges();
  } else {
    el.geoMetaCard.classList.add('hidden');
  }

  const img = new Image();
  img.onload = () => {
    state.currentImage = img;
    el.canvasPlaceholder.classList.add('hidden');
    el.canvas.width = img.width;
    el.canvas.height = img.height;
    fitToScreen();
    redrawMainCanvas();
  };
  img.src = `/api/image?path=${encodeURIComponent(data.path)}`;

  selectDenseActiveSource(data.filename || 'Imagem RGB');
  refreshWorkspaceStatus();
  updateDenseSchemaCompatibility();
}

function selectDenseActiveSource(label) {
  if (!el.denseSelectImage || !state.imagePath) return;
  let option = Array.from(el.denseSelectImage.options).find((item) => item.value === state.imagePath);
  if (!option) {
    option = document.createElement('option');
    option.value = state.imagePath;
    el.denseSelectImage.prepend(option);
  }
  option.textContent = `⭐ Cena Ativa no Canvas (${label})`;
  el.denseSelectImage.value = state.imagePath;
}

function updateSentinelBadges() {
  const is10m = el.chkSentinel10m.checked;
  if (is10m) {
    el.sentinelBadge.textContent = 'RESOLUÇÃO NOMINAL DECLARADA: 10 m/px (80x80m Contexto)';
    el.sentinelBadge.className = 'resolution-badge nominal-10m';
  } else {
    el.sentinelBadge.textContent = 'ESPAÇO DE PIXEL — SEM ESCALA MÉTRICA DECLARADA';
    el.sentinelBadge.className = 'resolution-badge display-only';
  }

  const isDense10m = el.denseChkSentinel10m.checked;
  if (isDense10m) {
    el.denseSentinelBadge.textContent = 'RESOLUÇÃO NOMINAL DECLARADA: 10 m/px (80x80m Contexto)';
    el.denseSentinelBadge.className = 'resolution-badge nominal-10m';
  } else {
    el.denseSentinelBadge.textContent = 'ESPAÇO DE PIXEL — SEM ESCALA MÉTRICA DECLARADA';
    el.denseSentinelBadge.className = 'resolution-badge display-only';
  }
}

async function handleImageUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || 'Erro ao carregar imagem');
    handleLoadedSource(data);
  } catch (err) {
    alert(`Erro no upload: ${err.message}`);
  }
}

// 5. Canvas Interaction, Pan & Zoom
function setupCanvasInteraction() {
  const btnZoomFit = document.getElementById('btn-zoom-fit');

  el.btnZoomIn.addEventListener('click', () => setZoom(state.zoom * 1.25));
  el.btnZoomOut.addEventListener('click', () => setZoom(state.zoom / 1.25));
  if (btnZoomFit) btnZoomFit.addEventListener('click', () => fitToScreen());
  el.btnZoomReset.addEventListener('click', () => resetZoom());

  el.chkShowGrid.addEventListener('change', () => redrawMainCanvas());
  el.chkShowRois.addEventListener('change', () => redrawMainCanvas());
  el.btnClearRois.addEventListener('click', () => {
    state.rois = [];
    state.patches = [];
    state.nextRoiId = 1;
    el.roiCount.textContent = '0 ROIs';
    renderRoiList();
    updateSummaryTable();
    redrawMainCanvas();
  });

  let isPanning = false;
  let panStartX = 0;
  let panStartY = 0;

  const getCanvasCoords = (e) => {
    const rect = el.canvasContainer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    return {
      x: Math.floor((mouseX - state.panX) / state.zoom),
      y: Math.floor((mouseY - state.panY) / state.zoom),
    };
  };

  el.canvasContainer.addEventListener('contextmenu', (e) => e.preventDefault());

  el.canvasContainer.addEventListener('mousedown', (e) => {
    if (!state.currentImage) return;

    // Pan with Right Click (button 2) or Middle Click (button 1) or Alt/Space key
    if (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey) {
      e.preventDefault();
      isPanning = true;
      panStartX = e.clientX - state.panX;
      panStartY = e.clientY - state.panY;
      el.canvasContainer.style.cursor = 'grabbing';
      return;
    }

    // Left Click: Start drawing ROI if inside canvas
    if (e.button === 0) {
      const { x, y } = getCanvasCoords(e);
      if (x >= 0 && x <= el.canvas.width && y >= 0 && y <= el.canvas.height) {
        state.isDrawing = true;
        state.drawStartX = x;
        state.drawStartY = y;
        state.currentRect = { x, y, width: 0, height: 0 };
      }
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (isPanning) {
      state.panX = e.clientX - panStartX;
      state.panY = e.clientY - panStartY;
      applyTransform();
      return;
    }

    if (!state.currentImage) return;
    const { x, y } = getCanvasCoords(e);
    el.cursorCoords.textContent = `Pixel X: ${x} | Y: ${y}`;

    if (state.sentinelDescriptor && state.sentinelDescriptor.geotransform && state.sentinelDescriptor.geotransform.length === 6) {
      const gt = state.sentinelDescriptor.geotransform;
      const scaleX = (state.sentinelDescriptor.width || el.canvas.width) / el.canvas.width;
      const scaleY = (state.sentinelDescriptor.height || el.canvas.height) / el.canvas.height;
      const natX = x * scaleX;
      const natY = y * scaleY;
      const easting = gt[0] + natX * gt[1] + natY * gt[2];
      const northing = gt[3] + natX * gt[4] + natY * gt[5];
      el.cursorGeoCoords.textContent = `Map: ${easting.toFixed(1)}, ${northing.toFixed(1)}`;
    } else {
      el.cursorGeoCoords.textContent = 'Geo: NOT_AVAILABLE';
    }

    if (state.isDrawing) {
      const curX = Math.max(0, Math.min(x, el.canvas.width));
      const curY = Math.max(0, Math.min(y, el.canvas.height));
      state.currentRect = {
        x: Math.min(state.drawStartX, curX),
        y: Math.min(state.drawStartY, curY),
        width: Math.abs(curX - state.drawStartX),
        height: Math.abs(curY - state.drawStartY),
      };
      redrawMainCanvas();
    }
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      el.canvasContainer.style.cursor = 'crosshair';
    }
    if (state.isDrawing) {
      state.isDrawing = false;
      if (state.currentRect && state.currentRect.width >= 8 && state.currentRect.height >= 8) {
        addRoi(state.currentRect);
      }
      state.currentRect = null;
      redrawMainCanvas();
    }
  });

  // Wheel zoom centered on cursor
  el.canvasContainer.addEventListener('wheel', (e) => {
    if (!state.currentImage) return;
    e.preventDefault();
    const rect = el.canvasContainer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
    const oldZoom = state.zoom;
    const newZoom = Math.max(0.05, Math.min(oldZoom * zoomFactor, 16.0));

    state.panX = mouseX - (mouseX - state.panX) * (newZoom / oldZoom);
    state.panY = mouseY - (mouseY - state.panY) * (newZoom / oldZoom);
    state.zoom = newZoom;
    applyTransform();
  }, { passive: false });
}

function fitToScreen() {
  if (!state.currentImage || !el.canvasContainer) return;
  const cw = el.canvasContainer.clientWidth || 800;
  const ch = el.canvasContainer.clientHeight || 600;
  const imgW = el.canvas.width || 1;
  const imgH = el.canvas.height || 1;

  const scale = Math.min((cw - 32) / imgW, (ch - 32) / imgH);
  state.zoom = Math.max(0.01, Math.min(scale, 4.0));
  state.panX = Math.round((cw - imgW * state.zoom) / 2);
  state.panY = Math.round((ch - imgH * state.zoom) / 2);
  applyTransform();
}

function setZoom(newZoom) {
  state.zoom = Math.max(0.05, Math.min(newZoom, 16.0));
  applyTransform();
}

function resetZoom() {
  if (!state.currentImage || !el.canvasContainer) return;
  const cw = el.canvasContainer.clientWidth || 800;
  const ch = el.canvasContainer.clientHeight || 600;
  state.zoom = 1.0;
  state.panX = Math.round((cw - el.canvas.width) / 2);
  state.panY = Math.round((ch - el.canvas.height) / 2);
  applyTransform();
}

function applyTransform() {
  el.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
  el.canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  el.canvas.style.transformOrigin = '0 0';
}

function addRoi(rect) {
  const activeClass = el.selectActiveClass.value;
  const activeSplit = el.selectActiveSplit.value;
  if (!activeClass) return;

  const roiId = `roi_${state.nextRoiId++}`;
  const roi = {
    id: roiId,
    x: Math.floor(rect.x / 8) * 8,
    y: Math.floor(rect.y / 8) * 8,
    width: Math.ceil(rect.width / 8) * 8,
    height: Math.ceil(rect.height / 8) * 8,
    class: activeClass,
    split: activeSplit,
  };
  state.rois.push(roi);
  el.roiCount.textContent = `${state.rois.length} ROIs`;

  // Generate 8x8 patches inside ROI
  for (let py = roi.y; py + 8 <= roi.y + roi.height; py += 8) {
    for (let px = roi.x; px + 8 <= roi.x + roi.width; px += 8) {
      if (px + 8 <= el.canvas.width && py + 8 <= el.canvas.height) {
        state.patches.push({
          id: `p_${state.patches.length}`,
          roi_id: roiId,
          x: px,
          y: py,
          width: 8,
          height: 8,
          class: activeClass,
          split: activeSplit,
        });
      }
    }
  }

  updateSummaryTable();
  renderRoiList();
  redrawMainCanvas();
}

function deleteRoi(roiId) {
  state.rois = state.rois.filter((roi) => roi.id !== roiId);
  state.patches = state.patches.filter((patch) => patch.roi_id !== roiId);
  el.roiCount.textContent = `${state.rois.length} ROIs`;
  renderRoiList();
  updateSummaryTable();
  redrawMainCanvas();
}

function renderRoiList() {
  if (!el.roiList) return;
  el.roiList.innerHTML = '';
  if (state.rois.length === 0) {
    const empty = document.createElement('span');
    empty.className = 'empty-msg';
    empty.textContent = 'Nenhuma ROI desenhada';
    el.roiList.appendChild(empty);
    return;
  }

  state.rois.forEach((roi) => {
    const patchCount = state.patches.filter((patch) => patch.roi_id === roi.id).length;
    const item = document.createElement('div');
    item.className = 'roi-list-item';
    const main = document.createElement('div');
    main.className = 'roi-list-main';
    const title = document.createElement('span');
    title.className = 'roi-list-title';
    title.textContent = `${roi.id} · ${roi.class} · ${roi.split.toUpperCase()}`;
    const meta = document.createElement('span');
    meta.className = 'roi-list-meta';
    meta.textContent = `x=${roi.x}, y=${roi.y}, ${roi.width}×${roi.height} · ${patchCount} patches`;
    main.append(title, meta);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'btn btn-xs btn-danger-outline roi-delete-button';
    remove.title = `Excluir somente ${roi.id}`;
    remove.setAttribute('aria-label', remove.title);
    remove.textContent = '✕';
    remove.addEventListener('click', () => deleteRoi(roi.id));
    item.append(main, remove);
    el.roiList.appendChild(item);
  });
}

function redrawMainCanvas() {
  if (!ctx || !state.currentImage) return;
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
  ctx.drawImage(state.currentImage, 0, 0);

  // Draw 8x8 grid
  if (el.chkShowGrid.checked) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= el.canvas.width; x += 8) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, el.canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= el.canvas.height; y += 8) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(el.canvas.width, y);
      ctx.stroke();
    }
  }

  // Draw Hexagonal H3 Overlay if enabled
  if (el.chkShowH3Overlay && el.chkShowH3Overlay.checked) {
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
    ctx.lineWidth = 1.0;
    const hexSize = 24;
    for (let y = 0; y < el.canvas.height + hexSize; y += hexSize * 1.5) {
      for (let x = 0; x < el.canvas.width + hexSize; x += hexSize * Math.sqrt(3)) {
        drawHexagon(ctx, x, y, hexSize * 0.55);
      }
    }
  }

  // Draw Patches
  state.patches.forEach((p) => {
    const clsObj = state.classes.find((c) => c.name === p.class);
    ctx.fillStyle = clsObj ? `${clsObj.color}55` : 'rgba(34, 197, 94, 0.35)';
    ctx.fillRect(p.x, p.y, p.width, p.height);
    ctx.strokeStyle = clsObj ? clsObj.color : '#22c55e';
    ctx.lineWidth = 1;
    ctx.strokeRect(p.x, p.y, p.width, p.height);
  });

  // Draw ROIs
  if (el.chkShowRois.checked) {
    state.rois.forEach((r) => {
      const clsObj = state.classes.find((c) => c.name === r.class);
      ctx.strokeStyle = clsObj ? clsObj.color : '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 2]);
      ctx.strokeRect(r.x, r.y, r.width, r.height);
      ctx.setLineDash([]);
    });
  }

  // Draw Current Selection Rect
  if (state.currentRect) {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.fillRect(state.currentRect.x, state.currentRect.y, state.currentRect.width, state.currentRect.height);
    ctx.strokeRect(state.currentRect.x, state.currentRect.y, state.currentRect.width, state.currentRect.height);
  }

  el.canvasPatchInfo.textContent = `Patches selecionados: ${state.patches.length}`;
}

function drawHexagon(context, cx, cy, r) {
  context.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.closePath();
  context.stroke();
}

// 6. Dataset Generation
function setupDatasetGeneration() {
  el.btnAutoSplit.addEventListener('click', () => {
    state.rois.forEach((r, idx) => {
      r.split = idx % 5 === 0 ? 'dev' : 'train';
    });
    state.patches.forEach((p) => {
      const parentRoi = state.rois.find((r) => r.id === p.roi_id);
      if (parentRoi) p.split = parentRoi.split;
    });
    renderRoiList();
    updateSummaryTable();
    redrawMainCanvas();
  });

  el.btnGenerateDataset.addEventListener('click', async () => {
    const datasetName = el.datasetNameInput.value.trim() || `dataset_${Date.now()}`;
    const payload = {
      dataset_name: datasetName,
      image_path: state.imagePath,
      is_sentinel_10m: el.chkSentinel10m.checked,
      patches: state.patches,
      bands: (state.modality === 'SENTINEL2_MULTIBAND' && state.sentinelDescriptor) ? state.sentinelDescriptor.bands : null,
      preview_width: el.canvas ? el.canvas.width : (state.sentinelDescriptor ? state.sentinelDescriptor.preview_width : null),
      preview_height: el.canvas ? el.canvas.height : (state.sentinelDescriptor ? state.sentinelDescriptor.preview_height : null),
      native_width: state.sentinelDescriptor ? state.sentinelDescriptor.width : null,
      native_height: state.sentinelDescriptor ? state.sentinelDescriptor.height : null,
    };

    try {
      el.btnGenerateDataset.disabled = true;
      el.btnGenerateDataset.textContent = 'Gerando Dataset...';
      const res = await fetch('/api/datasets/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Erro ao gerar dataset');

      alert(`Dataset '${data.dataset_name}' criado com sucesso! (${data.total_patches} patches extraídos)`);
      refreshWorkspaceStatus();
    } catch (err) {
      alert(`Erro: ${err.message}`);
    } finally {
      el.btnGenerateDataset.disabled = false;
      el.btnGenerateDataset.textContent = '⚡ Extrair Patches & Gerar Dataset (v2)';
    }
  });
}

function updateSummaryTable() {
  const counts = {};
  state.classes.forEach((c) => {
    counts[c.name] = { train: 0, dev: 0, probe: 0, total: 0 };
  });

  state.patches.forEach((p) => {
    if (counts[p.class]) {
      counts[p.class][p.split] = (counts[p.class][p.split] || 0) + 1;
      counts[p.class].total += 1;
    }
  });

  el.summaryTableBody.innerHTML = '';
  let totalTrain = 0;
  let totalDev = 0;

  state.classes.forEach((c) => {
    const row = document.createElement('tr');
    const cnt = counts[c.name];
    totalTrain += cnt.train;
    totalDev += cnt.dev;
    row.innerHTML = `
      <td><span class="class-dot" style="background:${c.color}"></span> ${c.name}</td>
      <td>${cnt.train}</td>
      <td>${cnt.dev}</td>
      <td>${cnt.probe}</td>
      <td><strong>${cnt.total}</strong></td>
    `;
    el.summaryTableBody.appendChild(row);
  });

  const canGenerate = state.patches.length > 0 && totalTrain > 0 && totalDev > 0;
  el.btnGenerateDataset.disabled = !canGenerate;

  if (state.patches.length > 0 && totalDev === 0) {
    el.datasetWarnings.classList.remove('hidden');
    el.datasetWarnings.textContent = 'Atenção: Nenhum patch atribuído ao split DEV. O TinyLogicVision exige amostras de DEV para monitorar a generalização.';
  } else {
    el.datasetWarnings.classList.add('hidden');
  }
}

// 7. Training Setup (Supports Async Jobs)
function setupTraining() {
  el.btnStartTrain.addEventListener('click', async () => {
    const dataset = el.trainSelectDataset.value;
    const modelName = el.trainModelName.value.trim() || `model_${dataset}`;
    const isAsync = el.chkTrainAsync ? el.chkTrainAsync.checked : false;

    if (!dataset) {
      alert('Selecione um dataset primeiro');
      return;
    }

    if (isAsync) {
      try {
        const res = await fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: 'train', dataset_id: dataset, model_name: modelName }),
        });
        const data = await res.json();
        if (data.success) {
          alert(`Job de treinamento iniciado em segundo plano! (ID: ${data.job_id})`);
          state.activeJobId = data.job_id;
          fetchAndRenderJobs();
        }
      } catch (err) {
        alert(`Erro ao iniciar job assíncrono: ${err.message}`);
      }
      return;
    }

    el.btnStartTrain.disabled = true;
    el.trainStatusBadge.textContent = 'Treinando em C++...';
    el.trainStatusBadge.className = 'badge badge-warning';

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_name: dataset, model_name: modelName }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Falha no treinamento');

      el.metricTrainAcc.textContent = data.final_train_acc || '100.00%';
      el.metricDevAcc.textContent = data.final_dev_acc || '-';
      el.metricDevLoss.textContent = data.final_dev_loss !== null ? data.final_dev_loss.toFixed(6) : '-';
      el.metricParams.textContent = data.parameter_count || '4707';
      el.trainTerminalLog.textContent = data.log || 'Treinamento concluído com sucesso!';
      el.trainStatusBadge.textContent = 'CONCLUÍDO (READY)';
      el.trainStatusBadge.className = 'badge badge-success';

      refreshWorkspaceStatus();
    } catch (err) {
      el.trainStatusBadge.textContent = 'ERRO';
      el.trainStatusBadge.className = 'badge badge-danger';
      el.trainTerminalLog.textContent = err.message;
    } finally {
      el.btnStartTrain.disabled = false;
    }
  });
}

// 8. Classify & Evaluate Setup
function setupClassifyAndEval() {
  el.btnRunClassify.addEventListener('click', async () => {
    const model = el.classifySelectModel.value;
    const file = el.classifyImageFile.files[0];
    if (!model || !file) {
      alert('Selecione o modelo e a amostra');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    try {
      const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
      const upData = await upRes.json();
      if (!upData.success) throw new Error('Falha no upload');

      const clsRes = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: model, image_path: upData.path }),
      });
      const clsData = await clsRes.json();
      if (!clsData.success) throw new Error(clsData.error || 'Erro na classificação');

      el.classifyResults.classList.remove('hidden');
      el.classifyPredClass.textContent = clsData.predicted;
      el.classifyPredScore.textContent = `${(clsData.score * 100).toFixed(1)}%`;
      el.classifyPreviewImg.src = `/api/image?path=${encodeURIComponent(upData.path)}`;

      el.classifyProbsContainer.innerHTML = '';
      Object.entries(clsData.probabilities || {}).forEach(([k, v]) => {
        const row = document.createElement('div');
        row.className = 'prob-bar-row';
        row.innerHTML = `
          <div class="prob-label-row">
            <span>${k}</span>
            <span class="mono">${(v * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width:${v * 100}%"></div>
          </div>
        `;
        el.classifyProbsContainer.appendChild(row);
      });
    } catch (err) {
      alert(`Erro: ${err.message}`);
    }
  });

  const runEvaluation = async (split) => {
    const model = el.evalSelectModel.value;
    const dataset = el.evalSelectDataset.value;
    if (!model || !dataset) {
      alert('Selecione modelo e dataset');
      return;
    }

    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: model, dataset_name: dataset, split }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Erro na avaliação');

      el.evalResults.classList.remove('hidden');
      el.evalSplitTag.textContent = split.toUpperCase();
      el.evalAcc.textContent = `${data.accuracy.toFixed(1)}%`;
      el.evalSamples.textContent = data.samples;
      el.evalLoss.textContent = data.loss.toFixed(6);
      el.evalMeanProb.textContent = data.mean_true_probability ? data.mean_true_probability.toFixed(4) : '-';
    } catch (err) {
      alert(`Erro na avaliação: ${err.message}`);
    }
  };

  el.btnEvalDev.addEventListener('click', () => runEvaluation('dev'));
  el.btnEvalProbe.addEventListener('click', () => runEvaluation('probe'));
}

// 9. Dense Map Setup (With Multithreading, GeoTIFF, H3 & O(1) Binary Inspection)
function setupDenseMap() {
  const selectedModelMetadata = () => state.models.find((item) => item.name === el.denseSelectModel.value);
  const selectedSourceIsActiveSentinel = () => Boolean(
    state.sentinelDescriptor &&
    state.sentinelDescriptor.bands &&
    el.denseSelectImage.value === state.imagePath
  );

  el.denseSliderConfidence.addEventListener('input', (e) => {
    el.denseValConfidence.textContent = parseFloat(e.target.value).toFixed(2);
  });
  el.denseSliderMargin.addEventListener('input', (e) => {
    el.denseValMargin.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // 1. Folder (.SAFE / R10m) loading for Tab 3
  const denseFolderInput = document.getElementById('dense-folder-input');
  const btnDenseSelectFolder = document.getElementById('btn-dense-select-folder');

  if (btnDenseSelectFolder && denseFolderInput) {
    btnDenseSelectFolder.addEventListener('click', () => denseFolderInput.click());
    denseFolderInput.addEventListener('change', async (e) => {
      const allFiles = Array.from(e.target.files);
      if (allFiles.length === 0) return;
      btnDenseSelectFolder.textContent = 'Localizando bandas 10m...';
      btnDenseSelectFolder.disabled = true;

      try {
        const isNot20_60 = (f) => !/(20m|60m)/i.test(f.webkitRelativePath || f.name);
        const b2File = allFiles.find((f) => /(^|[_.-])(B02|B2|B02_10m|B2_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b3File = allFiles.find((f) => /(^|[_.-])(B03|B3|B03_10m|B3_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b4File = allFiles.find((f) => /(^|[_.-])(B04|B4|B04_10m|B4_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const b8File = allFiles.find((f) => /(^|[_.-])(B08|B8|B08_10m|B8_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));
        const tciFile = allFiles.find((f) => /(^|[_.-])(TCI|TCI_10m)\.(jp2|tif|tiff)$/i.test(f.name) && isNot20_60(f));

        if (!b2File || !b3File || !b4File || !b8File) {
          throw new Error('Não foram encontradas todas as 4 bandas de 10m (B02, B03, B04, B08) na pasta selecionada.');
        }

        btnDenseSelectFolder.textContent = 'Enviando 4 bandas 10m...';
        const formData = new FormData();
        formData.append('file_b2', b2File, b2File.name);
        formData.append('file_b3', b3File, b3File.name);
        formData.append('file_b4', b4File, b4File.name);
        formData.append('file_b8', b8File, b8File.name);
        if (tciFile) formData.append('file_tci', tciFile, tciFile.name);

        const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const upData = await upRes.json();
        if (!upData.success) throw new Error(upData.error || 'Erro no upload das bandas');

        const savedFiles = upData.files || [upData];
        const findSaved = (re) => {
          const m = savedFiles.find((f) => re.test(f.original_name || f.filename));
          return m ? m.path : null;
        };

        const p2 = findSaved(/(^|[_.-])(B02|B2|B02_10m|B2_10m)\./i);
        const p3 = findSaved(/(^|[_.-])(B03|B3|B03_10m|B3_10m)\./i);
        const p4 = findSaved(/(^|[_.-])(B04|B4|B04_10m|B4_10m)\./i);
        const p8 = findSaved(/(^|[_.-])(B08|B8|B08_10m|B8_10m)\./i);

        if (p2 && p3 && p4 && p8) {
          const sRes = await fetch('/api/sentinel/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bands: { b2: p2, b3: p3, b4: p4, b8: p8 } }),
          });
          const sData = await sRes.json();
          if (sData.success) {
            applySentinelDescriptor(sData);
            refreshWorkspaceStatus();
            alert('Pasta Sentinel carregada com sucesso para classificação!');
            return;
          }
        }
        handleLoadedSource(upData);
        refreshWorkspaceStatus();
      } catch (err) {
        alert(`Erro ao abrir pasta Sentinel: ${err.message}`);
      } finally {
        btnDenseSelectFolder.textContent = '📁 Pasta';
        btnDenseSelectFolder.disabled = false;
      }
    });
  }

  // 2. Upload Arquivo(s) / Bandas para Tab 3
  if (el.btnDenseUpload && el.denseFileInput) {
    el.btnDenseUpload.addEventListener('click', () => el.denseFileInput.click());
    el.denseFileInput.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files);
      if (files.length === 0) return;

      el.btnDenseUpload.textContent = 'Enviando...';
      el.btnDenseUpload.disabled = true;

      try {
        if (files.length > 1) {
          const formData = new FormData();
          files.forEach((f, idx) => formData.append(`file_${idx}`, f, f.name));
          const res = await fetch('/api/upload', { method: 'POST', body: formData });
          const data = await res.json();
          if (!data.success) throw new Error(data.error || 'Erro no upload');

          const savedFiles = data.files || [data];
          const findSaved = (re) => {
            const m = savedFiles.find((f) => re.test(f.original_name || f.filename));
            return m ? m.path : null;
          };

          const p2 = findSaved(/(^|[_.-])(B02|B2|B02_10m|B2_10m)\./i);
          const p3 = findSaved(/(^|[_.-])(B03|B3|B03_10m|B3_10m)\./i);
          const p4 = findSaved(/(^|[_.-])(B04|B4|B04_10m|B4_10m)\./i);
          const p8 = findSaved(/(^|[_.-])(B08|B8|B08_10m|B8_10m)\./i);

          if (p2 && p3 && p4 && p8) {
            const sRes = await fetch('/api/sentinel/open', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ bands: { b2: p2, b3: p3, b4: p4, b8: p8 } }),
            });
            const sData = await sRes.json();
            if (sData.success) {
              applySentinelDescriptor(sData);
              refreshWorkspaceStatus();
              alert('Conjunto de 4 bandas Sentinel 10m carregado para classificação!');
              return;
            }
          }
          handleLoadedSource(savedFiles[0]);
          refreshWorkspaceStatus();
          alert(`${files.length} arquivos enviados com sucesso!`);
        } else {
          const file = files[0];
          const formData = new FormData();
          formData.append('file', file);
          const res = await fetch('/api/upload', { method: 'POST', body: formData });
          const data = await res.json();
          if (!data.success) throw new Error(data.error || 'Erro no upload');
          handleLoadedSource(data);
          refreshWorkspaceStatus();
          alert(`Imagem '${file.name}' carregada para classificação!`);
        }
      } catch (err) {
        alert(`Erro no upload: ${err.message}`);
      } finally {
        el.btnDenseUpload.textContent = '📂 Arquivo';
        el.btnDenseUpload.disabled = false;
      }
    });
  }

  if (el.denseSelectImage) {
    el.denseSelectImage.addEventListener('change', async (e) => {
      const p = e.target.value;
      if (!p) return;
      try {
        const res = await fetch('/api/source/open_local', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: p }),
        });
        const data = await res.json();
        if (data.success) {
          if (data.modality === 'SENTINEL2_MULTIBAND') {
            applySentinelDescriptor(data);
          } else {
            handleLoadedSource(data);
          }
          updateDenseSchemaCompatibility();
        }
      } catch (err) {
        console.error(err);
      }
    });
  }

  if (el.denseSelectModel) {
    el.denseSelectModel.addEventListener('change', updateDenseSchemaCompatibility);
  }

  el.btnRunDenseMap.addEventListener('click', async () => {
    const model = el.denseSelectModel.value;
    const imagePath = el.denseSelectImage.value;
    const stride = parseInt(el.denseSelectStride.value, 10);
    const confidence = parseFloat(el.denseSliderConfidence.value);
    const margin = parseFloat(el.denseSliderMargin.value);
    const threads = parseInt(el.denseSelectThreads.value, 10);
    const tileSize = parseInt(el.denseSelectTileSize.value, 10);
    const isSentinel = el.denseChkSentinel10m.checked;
    const modelMetadata = selectedModelMetadata();
    const sourceIsSentinel = selectedSourceIsActiveSentinel();
    const modelChannels = modelMetadata && modelMetadata.schema ? modelMetadata.schema.channels : 3;

    if (!model || !imagePath) {
      alert('Selecione modelo e imagem');
      return;
    }
    if ((modelChannels === 4) !== sourceIsSentinel) {
      const expected = modelChannels === 4 ? 'uma fonte Sentinel B2/B3/B4/B8' : 'uma imagem RGB';
      alert(`Schema incompatível: o modelo selecionado exige ${expected}.`);
      updateDenseSchemaCompatibility();
      return;
    }
    el.btnRunDenseMap.disabled = true;
    el.denseProgress.classList.remove('hidden');

    try {
      const res = await fetch('/api/dense_map', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: model,
          image_path: imagePath,
          stride,
          confidence,
          margin,
          threads,
          tile_size: tileSize,
          is_sentinel_10m: isSentinel,
          bands: sourceIsSentinel ? state.sentinelDescriptor.bands : null,
        }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Falha na classificação densa');

      renderDenseMapResults(data.run_id, data.metadata);
    } catch (err) {
      alert(`Erro: ${err.message}`);
    } finally {
      el.btnRunDenseMap.disabled = false;
      el.denseProgress.classList.add('hidden');
    }
  });
}

function updateDenseSchemaCompatibility() {
  if (!el.denseSchemaCompatibility) return;
  const model = state.models.find((item) => el.denseSelectModel && item.name === el.denseSelectModel.value);
  const modelSchema = model && model.schema ? model.schema : null;
  const sourceIsSentinel = Boolean(
    state.sentinelDescriptor && state.sentinelDescriptor.bands &&
    el.denseSelectImage && el.denseSelectImage.value === state.imagePath
  );
  const hasSource = Boolean(el.denseSelectImage && el.denseSelectImage.value);
  const sourceChannels = sourceIsSentinel ? 4 : (hasSource ? 3 : null);

  el.denseModelSchema.textContent = modelSchema
    ? `${modelSchema.modality} (${modelSchema.width}×${modelSchema.height}×${modelSchema.channels})`
    : 'não selecionado';
  el.denseSourceSchema.textContent = sourceChannels
    ? `${sourceIsSentinel ? 'SENTINEL2_MULTIBAND' : 'RGB'} (8×8×${sourceChannels})`
    : 'não selecionada';

  if (!modelSchema || !sourceChannels) {
    el.denseSchemaCompatibility.dataset.status = 'unknown';
    el.denseSchemaStatus.textContent = 'AGUARDANDO SELEÇÃO';
    return;
  }

  const compatible = modelSchema.width === 8 && modelSchema.height === 8 && modelSchema.channels === sourceChannels;
  el.denseSchemaCompatibility.dataset.status = compatible ? 'compatible' : 'incompatible';
  el.denseSchemaStatus.textContent = compatible ? 'SCHEMA COMPATÍVEL' : 'SCHEMA INCOMPATÍVEL';
}

function renderDenseMapResults(runId, metadata) {
  el.denseResultsPanel.classList.remove('hidden');
  el.denseTotalDecisions.textContent = `${metadata.decision_count || 0} decisões`;
  el.denseContextInfo.textContent = `Stride ${metadata.stride || 1} • Suporte 8x8`;

  // Download links
  el.linkDownloadCsv.href = `/api/runs/${runId}/classification.csv`;
  el.linkDownloadCsv.classList.toggle('hidden', !(metadata.engineering_stats && metadata.engineering_stats.decision_csv_available));
  el.linkDownloadJson.href = `/api/runs/${runId}/run.json`;
  el.linkDownloadOverlay.href = `/api/runs/${runId}/overlay.png`;
  el.linkDownloadClassmap.href = `/api/runs/${runId}/class_map.png`;
  el.linkDownloadConfidencemap.href = `/api/runs/${runId}/confidence.png`;
  el.linkDownloadMarginmap.href = `/api/runs/${runId}/margin.png`;
  el.linkDownloadGeotiffClass.href = `/api/runs/${runId}/class_map.tif`;
  el.linkDownloadGeotiffConf.href = `/api/runs/${runId}/confidence.tif`;
  el.linkDownloadH3Csv.href = `/api/runs/${runId}/classification_h3.csv`;
  el.linkDownloadProvenance.href = `/api/runs/${runId}/provenance.json`;

  // Dynamic Legend
  el.denseDynamicLegend.innerHTML = '';
  const palette = metadata.palette || {};
  (palette.classes || []).forEach((c) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <span><span class="legend-color-dot" style="background:${c.color}"></span> ${c.name}</span>
      <span class="mono">${c.index}</span>
    `;
    el.denseDynamicLegend.appendChild(item);
  });

  const uncItem = document.createElement('div');
  uncItem.className = 'legend-item';
  uncItem.innerHTML = `
    <span><span class="legend-color-dot" style="background:${(palette.uncertain && palette.uncertain.color) || '#808080'}"></span> UNCERTAIN</span>
    <span class="mono">254</span>
  `;
  el.denseDynamicLegend.appendChild(uncItem);

  // Load and render default view (Overlay)
  const dCanvas = el.denseCanvas;
  const dCtx = dCanvas.getContext('2d');
  const dImg = new Image();
  let activeDenseMode = 'overlay';
  const denseView = {
    scale: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    moved: false,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    inspectedImageX: null,
    inspectedImageY: null,
  };

  const updateDenseCrosshair = () => {
    if (denseView.inspectedImageX === null || !dCanvas.width || !dCanvas.height) {
      el.denseCrosshair.classList.add('hidden');
      return;
    }
    const canvasRect = dCanvas.getBoundingClientRect();
    const containerRect = el.denseCanvasContainer.getBoundingClientRect();
    el.denseCrosshair.style.left = `${canvasRect.left - containerRect.left +
      (denseView.inspectedImageX / dCanvas.width) * canvasRect.width}px`;
    el.denseCrosshair.style.top = `${canvasRect.top - containerRect.top +
      (denseView.inspectedImageY / dCanvas.height) * canvasRect.height}px`;
    el.denseCrosshair.classList.remove('hidden');
  };

  const applyDenseView = () => {
    dCanvas.style.left = `calc(50% + ${denseView.panX}px)`;
    dCanvas.style.top = `calc(50% + ${denseView.panY}px)`;
    dCanvas.style.transform = `translate(-50%, -50%) scale(${denseView.scale})`;
    el.denseZoomLevel.textContent = `${Math.round(denseView.scale * 100)}%`;
    updateDenseCrosshair();
  };

  const denseScaleFor = (fill) => {
    if (!dCanvas.width || !dCanvas.height) return 1;
    const padding = 24;
    const availableWidth = Math.max(1, el.denseCanvasContainer.clientWidth - padding);
    const availableHeight = Math.max(1, el.denseCanvasContainer.clientHeight - padding);
    const scaleX = availableWidth / dCanvas.width;
    const scaleY = availableHeight / dCanvas.height;
    return Math.min(8, Math.max(0.05, fill ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY)));
  };

  const setDenseScale = (nextScale, clientX = null, clientY = null) => {
    const bounded = Math.min(8, Math.max(0.05, nextScale));
    if (clientX !== null && clientY !== null) {
      const rect = el.denseCanvasContainer.getBoundingClientRect();
      const offsetX = clientX - (rect.left + rect.width / 2);
      const offsetY = clientY - (rect.top + rect.height / 2);
      const ratio = bounded / denseView.scale;
      denseView.panX = offsetX - (offsetX - denseView.panX) * ratio;
      denseView.panY = offsetY - (offsetY - denseView.panY) * ratio;
    }
    denseView.scale = bounded;
    applyDenseView();
  };

  const resetDenseView = (fill = false) => {
    denseView.panX = 0;
    denseView.panY = 0;
    denseView.scale = denseScaleFor(fill);
    applyDenseView();
  };

  dImg.onload = () => {
    dCanvas.width = dImg.width;
    dCanvas.height = dImg.height;
    dCtx.drawImage(dImg, 0, 0);
    denseView.inspectedImageX = null;
    denseView.inspectedImageY = null;
    requestAnimationFrame(() => resetDenseView(false));
  };
  dImg.src = `/api/runs/${runId}/overlay.png`;

  el.denseZoomOut.onclick = () => setDenseScale(denseView.scale / 1.25);
  el.denseZoomIn.onclick = () => setDenseScale(denseView.scale * 1.25);
  el.denseZoomFit.onclick = () => resetDenseView(false);
  el.denseZoomFill.onclick = () => resetDenseView(true);
  el.denseZoomActual.onclick = () => {
    denseView.panX = 0;
    denseView.panY = 0;
    setDenseScale(1);
  };
  el.denseToggleExpand.onclick = () => {
    const expanded = el.denseResultsPanel.classList.toggle('dense-expanded');
    el.denseToggleExpand.textContent = expanded ? '✕ Restaurar' : '⛶ Expandir';
    requestAnimationFrame(() => resetDenseView(false));
  };

  el.denseCanvasContainer.onwheel = (event) => {
    event.preventDefault();
    setDenseScale(denseView.scale * (event.deltaY < 0 ? 1.15 : 1 / 1.15), event.clientX, event.clientY);
  };

  el.denseCanvasContainer.onpointerdown = (event) => {
    if (event.button !== 0 || event.target.closest('.dense-navigation-toolbar')) return;
    denseView.dragging = true;
    denseView.moved = false;
    denseView.startX = event.clientX;
    denseView.startY = event.clientY;
    denseView.lastX = event.clientX;
    denseView.lastY = event.clientY;
    el.denseCanvasContainer.classList.add('is-dragging');
    el.denseCanvasContainer.setPointerCapture(event.pointerId);
  };

  el.denseCanvasContainer.onpointermove = (event) => {
    if (!denseView.dragging) return;
    const dx = event.clientX - denseView.lastX;
    const dy = event.clientY - denseView.lastY;
    if (Math.hypot(event.clientX - denseView.startX, event.clientY - denseView.startY) > 3) {
      denseView.moved = true;
    }
    denseView.panX += dx;
    denseView.panY += dy;
    denseView.lastX = event.clientX;
    denseView.lastY = event.clientY;
    applyDenseView();
  };

  // Setup view mode buttons
  const modeBtns = el.denseResultsPanel.querySelectorAll('.btn-mode');
  modeBtns.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === 'overlay');
    if (btn.dataset.mode === 'h3_aggr') {
      btn.disabled = !(metadata.h3 && metadata.h3.available);
      btn.title = btn.disabled ? 'H3 requer fonte georreferenciada e biblioteca H3 oficial' : 'Abrir CSV agregado por célula H3';
    }
    btn.onclick = () => {
      const mode = btn.dataset.mode;
      if (mode === 'h3_aggr') {
        if (metadata.h3 && metadata.h3.available) {
          window.open(`/api/runs/${runId}/classification_h3.csv`, '_blank', 'noopener');
        }
        return;
      }
      modeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      activeDenseMode = mode;
      const srcMap = {
        overlay: `/api/runs/${runId}/overlay.png`,
        class_map: `/api/runs/${runId}/class_map.png`,
        confidence: `/api/runs/${runId}/confidence.png`,
        margin: `/api/runs/${runId}/margin.png`,
        source: `/api/image?path=${encodeURIComponent(metadata.source_image_path || '')}`,
      };
      dImg.src = srcMap[mode] || srcMap.overlay;
    };
  });

  // $O(1)$ Binary Point Inspection
  const inspectDensePoint = async (clientX, clientY) => {
    const rect = dCanvas.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return;
    const scaleX = dCanvas.width / rect.width;
    const scaleY = dCanvas.height / rect.height;
    const canvasX = Math.min(dCanvas.width - 1, Math.max(0, Math.floor((clientX - rect.left) * scaleX)));
    const canvasY = Math.min(dCanvas.height - 1, Math.max(0, Math.floor((clientY - rect.top) * scaleY)));
    denseView.inspectedImageX = canvasX + 0.5;
    denseView.inspectedImageY = canvasY + 0.5;
    updateDenseCrosshair();
    const gridMode = ['class_map', 'confidence', 'margin'].includes(activeDenseMode);
    const clickX = gridMode
      ? Math.floor(canvasX * Number(metadata.grid_width) / dCanvas.width) * Number(metadata.stride)
      : Math.floor(canvasX * Number(metadata.source_width) / dCanvas.width);
    const clickY = gridMode
      ? Math.floor(canvasY * Number(metadata.grid_height) / dCanvas.height) * Number(metadata.stride)
      : Math.floor(canvasY * Number(metadata.source_height) / dCanvas.height);

    try {
      const inspRes = await fetch(`/api/runs/${runId}/inspect?x=${clickX}&y=${clickY}`);
      const inspData = await inspRes.json();
      if (!inspData.success) throw new Error(inspData.error || 'Decisão não encontrada');

      const d = inspData.decision;
      el.inspGrid.textContent = `(${d.grid_x}, ${d.grid_y})`;
      el.inspOrigin.textContent = `(${d.origin_x}, ${d.origin_y})`;
      el.inspCenter.textContent = `(${d.center_x}, ${d.center_y})`;

      const geospatial = metadata.geospatial || {};
      if (geospatial.available && geospatial.geotransform && geospatial.geotransform.length === 6) {
        const gt = geospatial.geotransform;
        const cx = parseFloat(d.center_x);
        const cy = parseFloat(d.center_y);
        const mapX = gt[0] + cx * gt[1] + cy * gt[2];
        const mapY = gt[3] + cx * gt[4] + cy * gt[5];
        el.inspMapCoords.textContent = `X: ${mapX.toFixed(1)} | Y: ${mapY.toFixed(1)}`;
      } else {
        el.inspMapCoords.textContent = 'NOT_AVAILABLE';
      }
      el.inspLatLon.textContent = geospatial.crs ? geospatial.crs.slice(0, 60) : 'NOT_AVAILABLE';
      el.inspH3Index.textContent = metadata.h3 && metadata.h3.available
        ? `Res ${metadata.h3.resolution} — CSV agregado`
        : 'NOT_AVAILABLE';

      el.inspTop1Class.textContent = d.predicted_class || '-';
      el.inspTop1Prob.textContent = d.probability || '-';
      el.inspTop2Class.textContent = d.second_class || '-';
      el.inspTop2Prob.textContent = d.second_probability || '-';
      el.inspMargin.textContent = d.margin || '-';

      if (d.status === 'UNCERTAIN') {
        el.inspStatus.textContent = 'UNCERTAIN';
        el.inspStatus.className = 'badge badge-uncertain';
      } else {
        el.inspStatus.textContent = 'CLASSIFIED';
        el.inspStatus.className = 'badge badge-classified';
      }
    } catch (err) {
      el.inspStatus.textContent = 'ERRO DE LEITURA';
      el.inspStatus.className = 'badge badge-uncertain';
      el.inspTop1Class.textContent = '-';
      el.inspTop1Prob.textContent = '-';
      console.warn('Falha na inspeção espacial:', err);
    }
  };

  el.denseCanvasContainer.onpointerup = (event) => {
    if (!denseView.dragging) return;
    denseView.dragging = false;
    el.denseCanvasContainer.classList.remove('is-dragging');
    if (el.denseCanvasContainer.hasPointerCapture(event.pointerId)) {
      el.denseCanvasContainer.releasePointerCapture(event.pointerId);
    }
    if (!denseView.moved) inspectDensePoint(event.clientX, event.clientY);
  };

  el.denseCanvasContainer.onpointercancel = (event) => {
    denseView.dragging = false;
    el.denseCanvasContainer.classList.remove('is-dragging');
    if (el.denseCanvasContainer.hasPointerCapture(event.pointerId)) {
      el.denseCanvasContainer.releasePointerCapture(event.pointerId);
    }
  };
}

// 10. Async Jobs Monitor
function setupJobsMonitor() {
  if (el.btnRefreshJobs) {
    el.btnRefreshJobs.addEventListener('click', () => fetchAndRenderJobs());
  }

  if (el.btnSubmitQuickJob) {
    el.btnSubmitQuickJob.addEventListener('click', async () => {
      const jType = el.quickJobType.value;
      let payload = { type: jType };
      if (jType === 'train') {
        if (state.datasets.length === 0) {
          alert('Nenhum dataset disponível');
          return;
        }
        payload.dataset_id = state.datasets[0].dataset_id;
        payload.model_name = `quick_model_${Date.now()}`;
      }

      try {
        const res = await fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.success) {
          alert(`Job ${data.job_id} submetido!`);
          fetchAndRenderJobs();
        }
      } catch (err) {
        alert('Erro ao submeter job');
      }
    });
  }
}

async function fetchAndRenderJobs() {
  try {
    const res = await fetch('/api/jobs');
    const data = await res.json();
    if (!data.success) return;

    const jobs = data.jobs || [];
    if (el.jobsCountBadge) el.jobsCountBadge.textContent = `${jobs.length} jobs`;
    if (el.navJobsBadge) {
      const runningCount = jobs.filter((j) => j.state === 'running').length;
      if (runningCount > 0) {
        el.navJobsBadge.textContent = runningCount;
        el.navJobsBadge.classList.remove('hidden');
      } else {
        el.navJobsBadge.classList.add('hidden');
      }
    }

    if (!el.jobsListContainer) return;
    if (jobs.length === 0) {
      el.jobsListContainer.innerHTML = '<div class="empty-msg">Nenhum job em execução no momento.</div>';
      return;
    }

    el.jobsListContainer.innerHTML = '';
    jobs.slice().reverse().forEach((j) => {
      const item = document.createElement('div');
      item.className = 'job-card-item';
      item.innerHTML = `
        <div class="job-card-header">
          <span class="job-title">${j.type === 'train' ? '🏋️ Treinamento' : '🗺️ Mapa Denso'} • <span class="mono text-cyan">${j.job_id}</span></span>
          <span class="badge ${j.state === 'completed' ? 'badge-success' : j.state === 'running' ? 'badge-warning' : 'badge-danger'}">${j.state.toUpperCase()}</span>
        </div>
        <div class="job-meta-row">
          <span>Início: ${new Date(j.started_at).toLocaleTimeString()}</span>
          <span>Exit Code: ${j.exit_code !== null ? j.exit_code : '-'}</span>
        </div>
        <div class="job-progress-bar">
          <div class="job-progress-fill ${j.state}" style="width:${j.progress * 100}%"></div>
        </div>
      `;
      item.addEventListener('click', () => {
        el.jobActiveDetail.classList.remove('hidden');
        el.activeJobId.textContent = j.job_id;
        el.activeJobStatus.textContent = j.state.toUpperCase();
        el.activeJobLog.textContent = j.log || '// Sem logs disponíveis ainda.';
      });
      el.jobsListContainer.appendChild(item);
    });
  } catch (err) {
    // silently catch
  }
}

// 11. Workspace Explorer & RIT Provenance Graph Explorer
async function refreshWorkspaceStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!data.success) return;
    if (el.activeWorkspacePath) {
      el.activeWorkspacePath.textContent = `127.0.0.1 • ${data.runtime_root}`;
      el.activeWorkspacePath.title = data.runtime_root;
    }

    state.datasets = data.datasets || [];
    state.models = data.models || [];

    // Populate dataset selects
    const updateSelect = (selectEl, items, key, textKey) => {
      if (!selectEl) return;
      selectEl.innerHTML = '';
      items.forEach((it) => {
        const opt = document.createElement('option');
        opt.value = it[key];
        opt.textContent = it[textKey] || it[key];
        selectEl.appendChild(opt);
      });
    };

    updateSelect(el.trainSelectDataset, state.datasets, 'dataset_id', 'dataset_id');
    updateSelect(el.evalSelectDataset, state.datasets, 'dataset_id', 'dataset_id');
    updateSelect(el.classifySelectModel, state.models, 'name', 'name');
    updateSelect(el.evalSelectModel, state.models, 'name', 'name');
    updateSelect(el.denseSelectModel, state.models, 'name', 'name');

    // Populate images select for dense mapping
    if (el.denseSelectImage) {
      const currentVal = el.denseSelectImage.value;
      el.denseSelectImage.innerHTML = '';

      if (state.imagePath) {
        const activeOpt = document.createElement('option');
        activeOpt.value = state.imagePath;
        activeOpt.textContent = `⭐ Cena Ativa no Canvas (${(state.imageMeta && state.imageMeta.filename) || 'Sentinel-2 Multibanda'})`;
        el.denseSelectImage.appendChild(activeOpt);
      }

      (data.uploads || []).forEach((u) => {
        if (state.imagePath && u.path === state.imagePath) return;
        const opt = document.createElement('option');
        opt.value = u.path;
        opt.textContent = `${u.name} (${u.width}x${u.height})`;
        el.denseSelectImage.appendChild(opt);
      });

      if (currentVal && Array.from(el.denseSelectImage.options).some(o => o.value === currentVal)) {
        el.denseSelectImage.value = currentVal;
      }
      updateDenseSchemaCompatibility();
    }

    if (el.browserDatasetsCount) el.browserDatasetsCount.textContent = state.datasets.length;
    if (el.browserModelsCount) el.browserModelsCount.textContent = state.models.length;

    renderBrowserItems();
    renderRitProvenance();
  } catch (err) {
    // status fetch failed
  }
}

function renderBrowserItems() {
  if (el.browserDatasetsList) {
    el.browserDatasetsList.innerHTML = '';
    state.datasets.forEach((d) => {
      const item = document.createElement('div');
      item.className = 'browser-item';
      item.innerHTML = `
        <div>
          <strong>${d.dataset_id}</strong>
          <div class="text-dim">${d.total_patches || 0} patches • ${d.is_sentinel_10m ? '10m Sentinel-2' : 'RGB Display'}</div>
        </div>
        <span class="badge badge-cyan">v2</span>
      `;
      el.browserDatasetsList.appendChild(item);
    });
  }

  if (el.browserModelsList) {
    el.browserModelsList.innerHTML = '';
    state.models.forEach((m) => {
      const item = document.createElement('div');
      item.className = 'browser-item';
      item.innerHTML = `
        <div>
          <strong>${m.name}</strong>
          <div class="text-dim">${(m.size_bytes / 1024).toFixed(1)} KB</div>
        </div>
        <span class="badge badge-emerald">.tlv</span>
      `;
      el.browserModelsList.appendChild(item);
    });
  }
}

async function renderRitProvenance() {
  if (!el.ritNodesList) return;
  el.ritNodesList.textContent = 'Carregando proveniência registrada...';
  try {
    const response = await fetch('/api/provenance');
    const graph = await response.json();
    if (!graph.success) throw new Error(graph.error || 'Falha ao carregar proveniência');
    el.ritNodesList.innerHTML = '';
    const nodes = graph.nodes || [];
    const edges = graph.edges || [];
    if (nodes.length === 0) {
      el.ritNodesList.textContent = 'Nenhum provenance.json real foi registrado neste workspace.';
      return;
    }

    const typeOrder = ['Source', 'ROI', 'Patch', 'Dataset', 'TrainingRun', 'Model', 'ClassificationRun', 'Artifact'];
    nodes.sort((a, b) => typeOrder.indexOf(a.type) - typeOrder.indexOf(b.type) || String(a.name).localeCompare(String(b.name)));
    const nodeNames = new Map(nodes.map((node) => [node.id, node.name || node.id]));
    nodes.slice(0, 500).forEach((node) => {
      const row = document.createElement('div');
      row.className = 'rit-node-row';

      const type = document.createElement('span');
      const cssType = ({ TrainingRun: 'run', ClassificationRun: 'run' }[node.type] || String(node.type).toLowerCase());
      type.className = `rit-tag tag-${cssType}`;
      type.textContent = node.type || 'Unknown';
      row.appendChild(type);

      const label = document.createElement('strong');
      label.textContent = node.name || node.id;
      row.appendChild(label);

      if (node.hash_sha256) {
        const hash = document.createElement('span');
        hash.className = 'mono text-dim';
        hash.textContent = `[${String(node.hash_sha256).slice(0, 12)}]`;
        row.appendChild(hash);
      }

      const relationships = edges.filter((edge) => edge.source === node.id).slice(0, 4);
      if (relationships.length) {
        const relation = document.createElement('span');
        relation.className = 'rit-arrow';
        relation.textContent = relationships.map((edge) =>
          `➔ ${edge.type} → ${nodeNames.get(edge.target) || edge.target}`).join(' | ');
        row.appendChild(relation);
      }
      el.ritNodesList.appendChild(row);
    });
    if (nodes.length > 500) {
      const note = document.createElement('div');
      note.className = 'help-text';
      note.textContent = `${nodes.length - 500} nós adicionais permanecem registrados nos arquivos provenance.json.`;
      el.ritNodesList.appendChild(note);
    }
  } catch (error) {
    el.ritNodesList.textContent = `Proveniência indisponível: ${error.message}`;
  }
}
