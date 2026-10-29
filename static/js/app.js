/**
 * DisasterSense-AI: Self-Supervised Deep Learning for Disaster Detection
 * Client-Side Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    currentChipId: 'Spain_7370579',
    currentView: 's1_false_color',
    currentMode: 'overlay',
    viewsData: {},
    lastPrediction: null,
    eventsData: null,
    benchmarkData: null,
    map: null,
    geoJsonLayer: null
  };

  // Safe SVG Placeholders to prevent broken images
  const NO_GT_SVG = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#0f172a"/>
  <text x="256" y="240" fill="#94a3b8" font-family="sans-serif" font-size="18" font-weight="600" text-anchor="middle">No Ground Truth Available</text>
  <text x="256" y="275" fill="#64748b" font-family="sans-serif" font-size="13" text-anchor="middle">Custom Upload Scene (Unlabeled)</text>
</svg>
`);

  const PRED_LOADING_SVG = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#0f172a"/>
  <text x="256" y="240" fill="#00d2ff" font-family="sans-serif" font-size="18" font-weight="600" text-anchor="middle">Generating AI Prediction...</text>
  <text x="256" y="275" fill="#94a3b8" font-family="sans-serif" font-size="13" text-anchor="middle">Running SAR Model Inference</text>
</svg>
`);

  function getErrorSvg(msg) {
    const safeMsg = (msg || 'Inference error').replace(/"/g, '&quot;');
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="#1e1b4b"/>
  <circle cx="256" cy="200" r="48" fill="#ef4444" opacity="0.2"/>
  <text x="256" y="212" fill="#ef4444" font-family="sans-serif" font-size="32" font-weight="bold" text-anchor="middle">!</text>
  <text x="256" y="280" fill="#fca5a5" font-family="sans-serif" font-size="16" font-weight="600" text-anchor="middle">Inference Error</text>
  <text x="256" y="315" fill="#94a3b8" font-family="sans-serif" font-size="12" text-anchor="middle">${safeMsg.slice(0, 50)}</text>
</svg>
    `);
  }

  // DOM Elements
  const elements = {
    // Navigation
    tabs: document.querySelectorAll('.nav-tab'),
    panes: document.querySelectorAll('.tab-pane'),
    deviceText: document.getElementById('deviceText'),
    chipsCountText: document.getElementById('chipsCountText'),

    // Studio Controls
    chipSelect: document.getElementById('chipSelect'),
    methodSelect: document.getElementById('methodSelect'),
    thresholdContainer: document.getElementById('thresholdContainer'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdLabel: document.getElementById('thresholdLabel'),
    thresholdValue: document.getElementById('thresholdValue'),
    opacitySlider: document.getElementById('opacitySlider'),
    opacityValue: document.getElementById('opacityValue'),
    baseImageSelect: document.getElementById('baseImageSelect'),
    runDetectionBtn: document.getElementById('runDetectionBtn'),
    viewButtons: document.querySelectorAll('.btn-chip'),
    modeButtons: document.querySelectorAll('.mode-btn'),
    downloadMaskBtn: document.getElementById('downloadMaskBtn'),
    downloadReportBtn: document.getElementById('downloadReportBtn'),

    // Inspection
    uploadInspectionCard: document.getElementById('uploadInspectionCard'),
    inspectionFormatBadge: document.getElementById('inspectionFormatBadge'),
    inspectionDimensions: document.getElementById('inspectionDimensions'),
    inspectionChannels: document.getElementById('inspectionChannels'),
    inspectionGSD: document.getElementById('inspectionGSD'),
    inspectionNotes: document.getElementById('inspectionNotes'),

    // Viewports & Images
    viewportOverlay: document.getElementById('viewportOverlay'),
    viewportComparison: document.getElementById('viewportComparison'),
    viewportConfidence: document.getElementById('viewportConfidence'),
    viewportError: document.getElementById('viewportError'),
    viewportBinary: document.getElementById('viewportBinary'),
    mainDisplayImg: document.getElementById('mainDisplayImg'),
    compSatelliteImg: document.getElementById('compSatelliteImg'),
    compGtImg: document.getElementById('compGtImg'),
    compPredImg: document.getElementById('compPredImg'),
    confidenceImg: document.getElementById('confidenceImg'),
    errorMapImg: document.getElementById('errorMapImg'),
    binaryMaskImg: document.getElementById('binaryMaskImg'),
    imageLoader: document.getElementById('imageLoader'),
    loaderText: document.getElementById('loaderText'),
    currentViewBadge: document.getElementById('currentViewBadge'),
    currentResBadge: document.getElementById('currentResBadge'),
    algorithmRunInfo: document.getElementById('algorithmRunInfo'),

    // Report Card
    reportSeverityBadge: document.getElementById('reportSeverityBadge'),
    reportTimestamp: document.getElementById('reportTimestamp'),
    reportSummaryText: document.getElementById('reportSummaryText'),

    // Metrics
    confValue: document.getElementById('confValue'),
    confSub: document.getElementById('confSub'),
    iouValue: document.getElementById('iouValue'),
    iouSub: document.getElementById('iouSub'),
    prValue: document.getElementById('prValue'),
    f1Value: document.getElementById('f1Value'),
    areaValue: document.getElementById('areaValue'),
    areaSub: document.getElementById('areaSub'),
    cmTableBody: document.getElementById('cmTableBody'),

    // Upload
    dropZone: document.getElementById('dropZone'),
    fileUpload: document.getElementById('fileUpload'),

    // Benchmark Tab
    benchmarkChartImg: document.getElementById('benchmarkChartImg'),

    // Map & Events
    worldMap: document.getElementById('worldMap'),
    eventsTableBody: document.getElementById('eventsTableBody')
  };

  // ========================================================
  // 1. Initialization
  // ========================================================
  async function init() {
    setupTabNavigation();
    setupEventListeners();
    await fetchSystemStatus();
    await loadSamples();
    await loadEventsData();
    await loadBenchmarkMetrics();
    initLeafletMap();
  }

  // ========================================================
  // 2. Navigation & Tabs
  // ========================================================
  function setupTabNavigation() {
    elements.tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        elements.tabs.forEach(t => t.classList.remove('active'));
        elements.panes.forEach(p => p.classList.remove('active'));

        tab.classList.add('active');
        const targetId = tab.getAttribute('data-tab');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');

        if (targetId === 'eventsTab' && state.map) {
          setTimeout(() => state.map.invalidateSize(), 200);
        }
        if (targetId === 'benchmarkTab') {
          loadBenchmarkMetrics();
        }
      });
    });
  }

  // ========================================================
  // 3. API & Data Loading
  // ========================================================
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.status === 'ready') {
        const dev = data.device.toUpperCase();
        elements.deviceText.textContent = `PyTorch ${data.torch_version} (${dev})`;
        elements.chipsCountText.textContent = `${data.sample_chips_count} Benchmark Chips`;
      }
    } catch (e) {
      console.warn('Status fetch error:', e);
    }
  }

  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const samples = await res.json();
      if (samples && samples.length > 0) {
        elements.chipSelect.innerHTML = '';
        samples.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.id;
          opt.textContent = `${s.country} (${s.id}) - VH: ${s.vh_thresh} dB`;
          elements.chipSelect.appendChild(opt);
        });
        state.currentChipId = samples[0].id;
        await loadChipVisualization(state.currentChipId);
      }
    } catch (e) {
      console.error('Error loading samples:', e);
    }
  }

  async function loadChipVisualization(chipId) {
    showLoader(true, 'Loading Multi-Sensor Arrays & Running Inference...');
    try {
      const res = await fetch(`/api/sample/${chipId}/visualize`);
      if (!res.ok) throw new Error('Failed to load chip visualization');
      const data = await res.json();
      state.viewsData = data.views || {};

      // If S1 is available, default to s1_false_color, else S2 RGB
      if (state.viewsData.s1_false_color) {
        setActiveBandView('s1_false_color');
      } else if (state.viewsData.s2_rgb) {
        setActiveBandView('s2_rgb');
      } else if (state.viewsData.custom_rgb) {
        setActiveBandView('custom_rgb');
      }

      // Update comparison satellite base image
      const baseImg = state.viewsData.s1_false_color || state.viewsData.s2_rgb || state.viewsData.custom_rgb || '';
      elements.compSatelliteImg.src = baseImg;

      if (state.viewsData.ground_truth) {
        elements.compGtImg.src = state.viewsData.ground_truth;
      } else {
        elements.compGtImg.src = NO_GT_SVG;
      }

      // Safe placeholder on prediction image to prevent broken image icon before inference returns
      elements.compPredImg.src = PRED_LOADING_SVG;

      // Update resolution badge if available
      if (data.geo_info && data.geo_info.pixel_size_x_m) {
        elements.currentResBadge.textContent = `${data.geo_info.pixel_size_x_m}m GSD (${data.geo_info.crs})`;
      } else {
        elements.currentResBadge.textContent = '512 × 512 px';
      }

      state.currentChipId = chipId;

      // Automatically execute AI inference for the loaded chip so prediction panel is populated
      await runPrediction();

    } catch (e) {
      alert(`Error loading visualization: ${e.message}`);
    } finally {
      showLoader(false);
    }
  }

  async function loadBenchmarkMetrics() {
    try {
      const res = await fetch('/api/benchmark_metrics');
      const data = await res.json();
      if (data && data.chart_b64) {
        elements.benchmarkChartImg.src = data.chart_b64;
      }
    } catch (e) {
      console.warn('Benchmark metrics error:', e);
    }
  }

  // ========================================================
  // 4. View & Mode Selection
  // ========================================================
  function setActiveBandView(viewKey) {
    state.currentView = viewKey;
    elements.viewButtons.forEach(btn => {
      if (btn.getAttribute('data-view') === viewKey) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    const b64 = state.viewsData[viewKey];
    if (b64) {
      elements.mainDisplayImg.src = b64;
      const titles = {
        s1_false_color: 'Sentinel-1 Dual-Pol False Color (VV/VH/ratio)',
        s1_vv: 'Sentinel-1 VV Backscatter (dB)',
        s1_vh: 'Sentinel-1 VH Backscatter (dB)',
        s2_rgb: 'Sentinel-2 True Color (RGB)',
        s2_cir: 'Sentinel-2 Color Infrared (CIR)',
        s2_swir: 'Sentinel-2 SWIR Agriculture & Water',
        s2_ndwi: 'Sentinel-2 NDWI Spectral Water Index',
        ground_truth: 'Hand Labeled Ground Truth Mask',
        custom_rgb: 'Uploaded RGB Image (Demonstration View)'
      };
      elements.currentViewBadge.textContent = titles[viewKey] || viewKey;
    }
  }

  function setActiveDisplayMode(mode) {
    state.currentMode = mode;
    elements.modeButtons.forEach(btn => {
      if (btn.getAttribute('data-mode') === mode) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    elements.viewportOverlay.style.display = 'none';
    elements.viewportComparison.style.display = 'none';
    elements.viewportConfidence.style.display = 'none';
    elements.viewportError.style.display = 'none';
    elements.viewportBinary.style.display = 'none';

    if (mode === 'overlay') {
      elements.viewportOverlay.style.display = 'flex';
    } else if (mode === 'comparison') {
      elements.viewportComparison.style.display = 'grid';
    } else if (mode === 'confidence') {
      elements.viewportConfidence.style.display = 'flex';
    } else if (mode === 'error') {
      elements.viewportError.style.display = 'flex';
    } else if (mode === 'binary') {
      elements.viewportBinary.style.display = 'flex';
    }
  }

  // ========================================================
  // 5. Prediction Execution
  // ========================================================
  async function runPrediction() {
    const method = elements.methodSelect.value;
    const threshold = parseFloat(elements.thresholdSlider.value);
    const opacity = parseFloat(elements.opacitySlider.value);
    const baseImageType = elements.baseImageSelect.value;

    showLoader(true, `Executing ${method.toUpperCase()} Neural Segmentation...`);
    elements.algorithmRunInfo.textContent = `Running ${method}...`;

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chip_id: state.currentChipId,
          method: method,
          threshold: threshold,
          base_image_type: baseImageType,
          opacity: opacity
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');

      state.lastPrediction = data;

      // Update Images
      if (data.views) {
        if (data.views.overlay) {
          elements.mainDisplayImg.src = data.views.overlay;
        }
        if (data.views.prediction_mask) {
          elements.compPredImg.src = data.views.prediction_mask;
        } else if (data.views.binary_mask) {
          elements.compPredImg.src = data.views.binary_mask;
        } else if (data.views.overlay) {
          elements.compPredImg.src = data.views.overlay;
        }
        if (data.views.confidence_map) {
          elements.confidenceImg.src = data.views.confidence_map;
        }
        if (data.views.binary_mask) {
          elements.binaryMaskImg.src = data.views.binary_mask;
        }
        if (data.views.error_map) {
          elements.errorMapImg.src = data.views.error_map;
        }
      }

      // Update Algorithm Info
      elements.algorithmRunInfo.textContent = `${data.method_name} completed in ${data.execution_time_ms} ms.`;
      elements.currentViewBadge.textContent = `Flood Detection Result (${data.method.toUpperCase()})`;

      // Render Disaster Report
      if (data.report) {
        elements.reportSeverityBadge.textContent = data.report.severity;
        elements.reportTimestamp.textContent = data.report.timestamp;
        elements.reportSummaryText.textContent = data.report.assessment;

        // Color severity badge
        elements.reportSeverityBadge.className = 'severity-badge';
        if (data.report.severity.includes('CRITICAL') || data.report.severity.includes('SEVERE')) {
          elements.reportSeverityBadge.classList.add('critical');
        } else if (data.report.severity.includes('MODERATE')) {
          elements.reportSeverityBadge.classList.add('moderate');
        } else {
          elements.reportSeverityBadge.classList.add('low');
        }
      }

      // Switch to active display mode
      setActiveDisplayMode(state.currentMode);

      // Render Quantitative Metrics
      renderMetrics(data);

    } catch (e) {
      console.error('Analysis error:', e);
      elements.algorithmRunInfo.textContent = `Error: ${e.message}`;
      elements.compPredImg.src = getErrorSvg(e.message);
    } finally {
      showLoader(false);
    }
  }

  function renderMetrics(data) {
    // Confidence
    if (data.confidence && data.confidence.mean_score_pct) {
      elements.confValue.textContent = data.confidence.mean_score_pct;
      elements.confSub.textContent = 'Mean probability score';
    } else {
      elements.confValue.textContent = 'N/A';
    }

    // Area Stats
    if (data.area_stats) {
      const a = data.area_stats;
      if (a.can_calculate_real_area && a.area_hectares !== null) {
        elements.areaValue.textContent = `${a.area_hectares} ha`;
        elements.areaSub.textContent = `${a.area_sq_km} km² (${a.water_percentage}% of scene)`;
      } else {
        elements.areaValue.textContent = `${a.water_percentage}%`;
        elements.areaSub.textContent = 'Relative pixel percentage (no CRS)';
      }
    }

    // Benchmark Metrics against Ground Truth
    if (data.metrics) {
      const m = data.metrics;
      elements.iouValue.textContent = `${(m.iou * 100).toFixed(1)}%`;
      elements.prValue.textContent = `${(m.precision * 100).toFixed(1)}% / ${(m.recall * 100).toFixed(1)}%`;
      elements.f1Value.textContent = `F1 / Dice: ${(m.f1 * 100).toFixed(1)}%`;

      // Update Table
      const totalEvaluated = m.total_evaluated || (m.tp + m.fp + m.tn + m.fn);
      const rowsHtml = `
        <tr>
          <td><strong style="color:#22c55e;">True Positive (TP) Flood Water</strong></td>
          <td>${m.tp.toLocaleString()} px</td>
          <td>${((m.tp / totalEvaluated) * 100).toFixed(2)}%</td>
          <td>${((m.tp * 100) / 10000).toFixed(2)} ha</td>
          <td>${((m.tp * 100) / 1000000).toFixed(4)} km²</td>
        </tr>
        <tr>
          <td><strong style="color:#ef4444;">False Positive (FP) False Alarms</strong></td>
          <td>${m.fp.toLocaleString()} px</td>
          <td>${((m.fp / totalEvaluated) * 100).toFixed(2)}%</td>
          <td>${((m.fp * 100) / 10000).toFixed(2)} ha</td>
          <td>${((m.fp * 100) / 1000000).toFixed(4)} km²</td>
        </tr>
        <tr>
          <td><strong style="color:#f59e0b;">False Negative (FN) Missed Water</strong></td>
          <td>${m.fn.toLocaleString()} px</td>
          <td>${((m.fn / totalEvaluated) * 100).toFixed(2)}%</td>
          <td>${((m.fn * 100) / 10000).toFixed(2)} ha</td>
          <td>${((m.fn * 100) / 1000000).toFixed(4)} km²</td>
        </tr>
        <tr>
          <td><strong style="color:#94a3b8;">True Negative (TN) Correct Land</strong></td>
          <td>${m.tn.toLocaleString()} px</td>
          <td>${((m.tn / totalEvaluated) * 100).toFixed(2)}%</td>
          <td>${((m.tn * 100) / 10000).toFixed(2)} ha</td>
          <td>${((m.tn * 100) / 1000000).toFixed(4)} km²</td>
        </tr>
        <tr style="border-top: 2px solid var(--border-color); background: rgba(0,0,0,0.2);">
          <td><strong>Ignored / Cloud Occlusions (-1)</strong></td>
          <td>${m.ignored.toLocaleString()} px</td>
          <td>--</td>
          <td>--</td>
          <td>--</td>
        </tr>
      `;
      elements.cmTableBody.innerHTML = rowsHtml;
    } else {
      elements.iouValue.textContent = 'N/A';
      elements.iouSub.textContent = 'No Ground Truth mask';
      elements.prValue.textContent = '-- / --';
      elements.f1Value.textContent = 'F1 / Dice: --';
      elements.cmTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">Ground truth labels not present for this custom image</td></tr>`;
    }
  }

  function resetMetrics() {
    elements.confValue.textContent = '--';
    elements.iouValue.textContent = '--';
    elements.iouSub.textContent = 'Ground Truth Overlap';
    elements.prValue.textContent = '-- / --';
    elements.f1Value.textContent = 'F1 / Dice: --';
    elements.areaValue.textContent = '--';
    elements.areaSub.textContent = '10m Ground Resolution';
    elements.reportSeverityBadge.textContent = 'ASSESSMENT READY';
    elements.reportSeverityBadge.className = 'severity-badge';
    elements.reportSummaryText.textContent = 'Select or upload a satellite image and click "Analyze Satellite Image" to generate a disaster extent evaluation and natural language report.';
    elements.cmTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">Run detection to evaluate confusion matrix</td></tr>`;
  }

  function showLoader(visible, text = 'Processing Satellite Array...') {
    elements.loaderText.textContent = text;
    elements.imageLoader.style.display = visible ? 'flex' : 'none';
  }

  // ========================================================
  // 6. Global Events Leaflet Map
  // ========================================================
  async function loadEventsData() {
    try {
      const res = await fetch('/api/events');
      state.eventsData = await res.json();
      renderEventsTable(state.eventsData);
    } catch (e) {
      console.error('Error loading events:', e);
    }
  }

  function renderEventsTable(geoJson) {
    if (!geoJson || !geoJson.features) return;
    const features = geoJson.features;
    let html = '';
    features.forEach(f => {
      const p = f.properties;
      html += `
        <tr>
          <td><span class="sub-badge">#${p.ID}</span></td>
          <td><strong>${p.location}</strong></td>
          <td><span class="mono-badge">${p.ISO_CC}</span></td>
          <td>${p.s1_date}</td>
          <td>${p.s2_date}</td>
          <td><span class="sub-badge">${p.orbit}</span></td>
          <td>${p.rel_orbit_num}</td>
          <td><span class="mono-badge" style="color:var(--accent-amber);">${p.VH_thresh} dB</span></td>
          <td>${p.train_chip}</td>
          <td>${p.val_chip}</td>
          <td>
            <button class="btn-chip" onclick="window.selectEventCountry('${p.location}')">Inspect</button>
          </td>
        </tr>
      `;
    });
    elements.eventsTableBody.innerHTML = html;
  }

  function initLeafletMap() {
    if (typeof L === 'undefined') return;

    state.map = L.map('worldMap', {
      center: [20.0, 0.0],
      zoom: 2,
      minZoom: 2,
      maxZoom: 12
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(state.map);

    if (state.eventsData && state.eventsData.features) {
      addGeoJsonToMap(state.eventsData);
    }
  }

  function addGeoJsonToMap(geoJson) {
    if (!state.map) return;

    state.geoJsonLayer = L.geoJSON(geoJson, {
      style: {
        color: '#00d2ff',
        weight: 2,
        opacity: 0.8,
        fillColor: '#00d2ff',
        fillOpacity: 0.2
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        const popupContent = `
          <div style="font-family: var(--font-sans); color: #0f172a; min-width: 200px;">
            <h4 style="margin: 0 0 4px 0; color: #0284c7; font-size: 1rem;">${p.location} (ID: ${p.ID})</h4>
            <div style="font-size: 0.8rem; line-height: 1.5;">
              <div><strong>Country Code:</strong> ${p.ISO_CC}</div>
              <div><strong>S1 Date:</strong> ${p.s1_date} (${p.orbit})</div>
              <div><strong>S2 Date:</strong> ${p.s2_date}</div>
              <div><strong>Calibrated VH Threshold:</strong> ${p.VH_thresh} dB</div>
              <div><strong>Chips:</strong> ${p.train_chip + p.val_chip}</div>
            </div>
            <button onclick="window.selectEventCountry('${p.location}')" style="margin-top: 8px; width: 100%; background: #0284c7; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-weight: 600;">
              Focus in Disaster Studio
            </button>
          </div>
        `;
        layer.bindPopup(popupContent);
      }
    }).addTo(state.map);
  }

  window.selectEventCountry = function(countryName) {
    const studioTab = document.querySelector('[data-tab="studioTab"]');
    if (studioTab) studioTab.click();

    const options = Array.from(elements.chipSelect.options);
    const match = options.find(o => o.text.toLowerCase().includes(countryName.toLowerCase()));
    if (match) {
      elements.chipSelect.value = match.value;
      state.currentChipId = match.value;
      loadChipVisualization(state.currentChipId);
    }
  };

  // ========================================================
  // 7. Event Listeners Setup
  // ========================================================
  function setupEventListeners() {
    // Chip Selector change
    elements.chipSelect.addEventListener('change', e => {
      state.currentChipId = e.target.value;
      loadChipVisualization(state.currentChipId);
    });

    // Multi-sensor band preview buttons
    elements.viewButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const view = btn.getAttribute('data-view');
        setActiveBandView(view);
      });
    });

    // Display mode buttons
    elements.modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-mode');
        setActiveDisplayMode(mode);
      });
    });

    // Algorithm selection change
    elements.methodSelect.addEventListener('change', e => {
      const val = e.target.value;
      if (val === 'metadata_vh' || val === 'otsu' || val === 'ssl_unet' || val === 'supervised_unet' || val === 'fcn') {
        elements.thresholdContainer.style.display = 'none';
      } else if (val === 'ndwi') {
        elements.thresholdContainer.style.display = 'block';
        elements.thresholdLabel.textContent = 'NDWI Threshold:';
        elements.thresholdSlider.min = -0.5;
        elements.thresholdSlider.max = 0.5;
        elements.thresholdSlider.step = 0.05;
        elements.thresholdSlider.value = 0.0;
        elements.thresholdValue.textContent = '0.00';
      } else if (val === 'rgb_demo') {
        elements.thresholdContainer.style.display = 'none';
      }
    });

    // Opacity Slider
    elements.opacitySlider.addEventListener('input', e => {
      const val = parseFloat(e.target.value);
      elements.opacityValue.textContent = `${Math.round(val * 100)}%`;
      if (state.lastPrediction) {
        runPrediction();
      }
    });

    // Base Image selection change
    elements.baseImageSelect.addEventListener('change', () => {
      if (state.lastPrediction) {
        runPrediction();
      }
    });

    // Run Detection Button
    elements.runDetectionBtn.addEventListener('click', runPrediction);

    // Export Mask Button
    elements.downloadMaskBtn.addEventListener('click', () => {
      if (!state.lastPrediction || !state.lastPrediction.views.binary_mask) {
        alert('Please run disaster analysis first before exporting the mask.');
        return;
      }
      const a = document.createElement('a');
      a.href = state.lastPrediction.views.binary_mask;
      a.download = `${state.currentChipId}_flood_mask.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });

    // Download Report Button
    elements.downloadReportBtn.addEventListener('click', async () => {
      if (!state.lastPrediction || !state.lastPrediction.report) {
        alert('Please analyze an image first to generate the disaster report.');
        return;
      }
      const md = state.lastPrediction.report.markdown;
      const res = await fetch('/api/download_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ markdown: md })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${state.currentChipId}_disaster_assessment_report.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });

    // File Upload Drag and Drop
    elements.dropZone.addEventListener('click', () => elements.fileUpload.click());

    elements.dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      elements.dropZone.style.borderColor = 'var(--accent-cyan)';
    });

    elements.dropZone.addEventListener('dragleave', () => {
      elements.dropZone.style.borderColor = 'var(--border-color)';
    });

    elements.dropZone.addEventListener('drop', e => {
      e.preventDefault();
      elements.dropZone.style.borderColor = 'var(--border-color)';
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    elements.fileUpload.addEventListener('change', e => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    showLoader(true, `Uploading & analyzing ${file.name}...`);

    try {
      const uploadRes = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) throw new Error(uploadData.error || 'Upload failed');

      const info = uploadData.inspection || uploadData.geo_info || {};
      elements.uploadInspectionCard.style.display = 'block';
      if (info.dimensions) {
        elements.inspectionDimensions.textContent = `Dimensions: ${info.dimensions[0]} × ${info.dimensions[1]} px`;
      }
      elements.inspectionChannels.textContent = `Channels: ${info.channels || 3} band(s) | Type: ${info.dtype || 'raster'}`;

      if (info.is_geospatial) {
        elements.inspectionFormatBadge.textContent = `GeoTIFF (${info.crs})`;
        elements.inspectionFormatBadge.className = 'format-badge';
        elements.inspectionGSD.textContent = `Resolution: ${info.pixel_size_x_m}m × ${info.pixel_size_y_m}m GSD (${info.pixel_area_m2} m²/px)`;
      } else {
        elements.inspectionFormatBadge.textContent = 'Standard RGB / Optical Image';
        elements.inspectionFormatBadge.className = 'format-badge rgb';
        elements.inspectionGSD.textContent = 'Resolution: Unreferenced (No Geospatial CRS)';
      }

      elements.inspectionNotes.textContent = info.compatibility_notes || '';
      if (info.is_sar_compatible) {
        elements.inspectionNotes.className = 'inspection-alert good';
        elements.methodSelect.value = 'ssl_unet';
      } else if (info.is_rgb_demo) {
        elements.inspectionNotes.className = 'inspection-alert';
        elements.methodSelect.value = 'rgb_demo';
      }

      const opt = document.createElement('option');
      opt.value = uploadData.chip_id;
      opt.textContent = `Uploaded: ${file.name}`;
      elements.chipSelect.appendChild(opt);
      elements.chipSelect.value = uploadData.chip_id;

      state.currentChipId = uploadData.chip_id;
      await loadChipVisualization(uploadData.chip_id);

    } catch (e) {
      alert(`Upload error: ${e.message}`);
    } finally {
      showLoader(false);
    }
  }

  // Run on startup
  init();
});
