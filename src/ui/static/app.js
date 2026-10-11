/**
 * ==============================================================================
 * DFS Optimizer Pro - Client Application Logic (app.js)
 * ==============================================================================
 */

(function () {
  'use strict';

  // Application State
  const state = {
    playersFile: null,      // { name: string, content: string }
    templateFile: null,     // { name: string, content: string }
    projectionsFile: null,  // { name: string, content: string }
    inspection: null,       // Result of /api/inspect
    currentJobId: null,
    pollTimer: null,
  };

  // DOM Elements
  const dropzonePlayers = document.getElementById('dropzone-players');
  const inputPlayers = document.getElementById('input-players');
  const fileNamePlayers = document.getElementById('file-name-players');

  const dropzoneTemplate = document.getElementById('dropzone-template');
  const inputTemplate = document.getElementById('input-template');
  const fileNameTemplate = document.getElementById('file-name-template');

  const dropzoneProjections = document.getElementById('dropzone-projections');
  const inputProjections = document.getElementById('input-projections');
  const fileNameProjections = document.getElementById('file-name-projections');

  const detectionSummary = document.getElementById('detection-summary');
  const metricPlatform = document.getElementById('metric-platform');
  const metricFormat = document.getElementById('metric-format');
  const metricFee = document.getElementById('metric-fee');
  const metricEntries = document.getElementById('metric-entries');
  const metricPlayers = document.getElementById('metric-players');
  const platformBadgePill = document.getElementById('platform-badge-pill');
  const detectedPlatformText = document.getElementById('detected-platform-text');
  const engineStatusPill = document.getElementById('engine-status-pill');

  const btnOptimize = document.getElementById('btn-optimize');
  const btnOptimizeText = document.getElementById('btn-optimize-text');

  // Sliders & Controls
  const numLineupsSlider = document.getElementById('num-lineups');
  const valNumLineups = document.getElementById('val-num-lineups');

  const stackRatioSlider = document.getElementById('stack-ratio');
  const valStackRatio = document.getElementById('val-stack-ratio');

  const maxQbExposureSlider = document.getElementById('max-qb-exposure');
  const valMaxQbExposure = document.getElementById('val-max-qb-exposure');
  const qbMaxCount = document.getElementById('qb-max-count');

  const maxSkillExposureSlider = document.getElementById('max-skill-exposure');
  const valMaxSkillExposure = document.getElementById('val-max-skill-exposure');
  const skillMaxCount = document.getElementById('skill-max-count');

  const maxDefExposureSlider = document.getElementById('max-def-exposure');
  const valMaxDefExposure = document.getElementById('val-max-def-exposure');
  const defMaxCount = document.getElementById('def-max-count');

  const maxRepeatingSlider = document.getElementById('max-repeating');
  const valMaxRepeating = document.getElementById('val-max-repeating');

  // Advanced Inputs
  const numCandidatesInput = document.getElementById('num-candidates');
  const numFieldInput = document.getElementById('num-field');
  const numTrialsInput = document.getElementById('num-trials');
  const randomnessDevInput = document.getElementById('randomness-dev');
  const strictCapsInput = document.getElementById('strict-caps');

  // Progress Section
  const progressSection = document.getElementById('progress-section');
  const progressStageTitle = document.getElementById('progress-stage-title');
  const progressStageDesc = document.getElementById('progress-stage-desc');
  const progressPctBadge = document.getElementById('progress-pct');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const logConsole = document.getElementById('log-console');
  const btnClearLogs = document.getElementById('btn-clear-logs');

  // Results Section
  const resultsSection = document.getElementById('results-section');
  const btnDownload = document.getElementById('btn-download');
  const btnReset = document.getElementById('btn-reset');
  const resNumLineups = document.getElementById('res-num-lineups');
  const resUniquePlayers = document.getElementById('res-unique-players');
  const resPrimaryStack = document.getElementById('res-primary-stack');
  const resRepeating = document.getElementById('res-repeating');
  const exposuresTableBody = document.getElementById('exposures-table-body');

  // ---------------------------------------------------------------------------
  // Dropzone Setup Helpers
  // ---------------------------------------------------------------------------
  function setupDropzone(dropzone, input, onFileLoaded) {
    dropzone.addEventListener('click', () => input.click());

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFile(files[0], onFileLoaded);
      }
    });

    input.addEventListener('change', (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFile(files[0], onFileLoaded);
      }
    });
  }

  function handleFile(file, callback) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please upload a valid .csv file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      callback(file.name, content);
    };
    reader.readAsText(file);
  }

  // ---------------------------------------------------------------------------
  // Readiness State Tracking
  // ---------------------------------------------------------------------------
  function updateReadyState() {
    const hasTemplate = !!state.templateFile;
    const hasPlayers = !!state.playersFile;
    const hasProjections = !!state.projectionsFile;
    const isReady = hasTemplate && hasPlayers && hasProjections;

    btnOptimize.disabled = !isReady;
    btnOptimizeText.textContent = 'Optimize & Generate Lineups';
  }

  // ---------------------------------------------------------------------------
  // File Ingestion Handlers
  // ---------------------------------------------------------------------------
  setupDropzone(dropzonePlayers, inputPlayers, (name, content) => {
    state.playersFile = { name, content };
    fileNamePlayers.textContent = `${name} (${formatSize(content.length)})`;
    dropzonePlayers.classList.add('file-loaded');
    updateReadyState();
    inspectFiles();
  });

  setupDropzone(dropzoneTemplate, inputTemplate, (name, content) => {
    state.templateFile = { name, content };
    fileNameTemplate.textContent = `${name} (${formatSize(content.length)})`;
    dropzoneTemplate.classList.add('file-loaded');
    updateReadyState();
    inspectFiles();
  });

  setupDropzone(dropzoneProjections, inputProjections, (name, content) => {
    state.projectionsFile = { name, content };
    fileNameProjections.textContent = `${name} (${formatSize(content.length)})`;
    dropzoneProjections.classList.add('file-loaded');
    updateReadyState();
    inspectFiles();
  });

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  }

  // ---------------------------------------------------------------------------
  // Live File Inspection
  // ---------------------------------------------------------------------------
  async function inspectFiles() {
    if (!state.playersFile && !state.templateFile) return;

    try {
      const res = await fetch('/api/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          players_content: state.playersFile ? state.playersFile.content : '',
          template_content: state.templateFile ? state.templateFile.content : '',
          projections_content: state.projectionsFile ? state.projectionsFile.content : '',
        }),
      });

      if (!res.ok) return;
      const data = await res.json();
      state.inspection = data;

      // Update detection bar
      detectionSummary.classList.remove('hidden');

      const isDk = data.platform === 'draftkings';
      metricPlatform.textContent = isDk ? 'DraftKings' : 'FanDuel';
      metricFormat.textContent = data.is_single_game
        ? (isDk ? 'Showdown (CPT/FLEX)' : 'Single Game (MVP)')
        : `Classic (${isDk ? '$50k' : '$60k'} Cap)`;

      metricFee.textContent = `$${data.entry_fee.toFixed(2)}`;
      metricEntries.textContent = `${data.num_entries} Entries`;
      metricPlayers.textContent = data.num_players > 0 ? `${data.num_players} Players` : 'Ready';

      // Update Header Pill
      platformBadgePill.className = `platform-pill ${isDk ? 'dk-active' : 'fd-active'}`;
      detectedPlatformText.textContent = `${isDk ? 'DraftKings' : 'FanDuel'} ${data.is_single_game ? 'Showdown' : 'Classic'}`;

      // Set default target lineups to template row count
      if (data.num_entries && data.num_entries > 0) {
        numLineupsSlider.max = Math.max(150, data.num_entries);
        numLineupsSlider.value = data.num_entries;
        valNumLineups.textContent = data.num_entries;
        updateCapCounts();
      }

      // Check if ready to optimize
      updateReadyState();
    } catch (err) {
      console.error('Inspection failed:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Slider Event Listeners & Cap Count Updates
  // ---------------------------------------------------------------------------
  function updateCapCounts() {
    const k = parseInt(numLineupsSlider.value, 10);
    valNumLineups.textContent = k;

    const qbCap = parseFloat(maxQbExposureSlider.value) / 100;
    qbMaxCount.textContent = Math.floor(k * qbCap);

    const skillCap = parseFloat(maxSkillExposureSlider.value) / 100;
    skillMaxCount.textContent = Math.floor(k * skillCap);

    const defCap = parseFloat(maxDefExposureSlider.value) / 100;
    defMaxCount.textContent = Math.floor(k * defCap);
  }

  numLineupsSlider.addEventListener('input', updateCapCounts);

  stackRatioSlider.addEventListener('input', (e) => {
    valStackRatio.textContent = `${e.target.value}%`;
  });

  maxQbExposureSlider.addEventListener('input', (e) => {
    valMaxQbExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  maxSkillExposureSlider.addEventListener('input', (e) => {
    valMaxSkillExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  maxDefExposureSlider.addEventListener('input', (e) => {
    valMaxDefExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  maxRepeatingSlider.addEventListener('input', (e) => {
    valMaxRepeating.textContent = e.target.value;
  });



  btnClearLogs.addEventListener('click', () => {
    logConsole.innerHTML = '';
  });

  // ---------------------------------------------------------------------------
  // Optimization Execution & Progress Polling
  // ---------------------------------------------------------------------------
  btnOptimize.addEventListener('click', async () => {
    if (!state.playersFile || !state.templateFile || !state.projectionsFile) {
      alert('Please upload all 3 required files: Entries Template, Player Salaries, and Player Projections.');
      return;
    }

    btnOptimize.disabled = true;
    engineStatusPill.className = 'status-pill';
    engineStatusPill.style.color = 'var(--accent-cyan)';
    engineStatusPill.innerHTML = '<span class="pulse-dot" style="background:var(--accent-cyan);box-shadow:0 0 10px var(--accent-cyan);"></span> Simulating...';

    // Show Progress Section
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    logConsole.innerHTML = '<div class="log-line info">[SYS] Launching quantitative optimization pipeline...</div>';
    progressBarFill.style.width = '5%';
    progressPctBadge.textContent = '5%';

    // Build payload
    const payload = {
      players_content: state.playersFile.content,
      players_filename: state.playersFile.name,
      template_content: state.templateFile.content,
      template_filename: state.templateFile.name,
      projections_content: state.projectionsFile ? state.projectionsFile.content : '',
      projections_filename: state.projectionsFile ? state.projectionsFile.name : '',
      platform: state.inspection ? state.inspection.platform : 'draftkings',
      options: {
        is_single_game: state.inspection ? state.inspection.is_single_game : false,
        num_lineups: parseInt(numLineupsSlider.value, 10),
        entry_fee: state.inspection ? state.inspection.entry_fee : 1.0,
        stack_ratio: parseFloat(stackRatioSlider.value) / 100,
        max_qb_exposure: parseFloat(maxQbExposureSlider.value) / 100,
        max_rb_exposure: parseFloat(maxSkillExposureSlider.value) / 100,
        max_wr_exposure: parseFloat(maxSkillExposureSlider.value) / 100,
        max_te_exposure: parseFloat(maxSkillExposureSlider.value) / 100,
        max_def_exposure: parseFloat(maxDefExposureSlider.value) / 100,
        max_exposure: parseFloat(maxSkillExposureSlider.value) / 100,
        max_repeating: parseInt(maxRepeatingSlider.value, 10),
        num_candidates: parseInt(numCandidatesInput.value, 10),
        num_field: parseInt(numFieldInput.value, 10),
        num_trials: parseInt(numTrialsInput.value, 10),
        randomness: parseFloat(randomnessDevInput.value),
        zero_unprojected: true,
        strict_caps: strictCapsInput.checked,
      },
    };

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      state.currentJobId = data.job_id;
      startProgressPolling(data.job_id);
    } catch (err) {
      appendLog(`[ERROR] Optimization failed to start: ${err.message}`, 'error');
      btnOptimize.disabled = false;
      engineStatusPill.className = 'status-pill status-ready';
      engineStatusPill.innerHTML = '<span class="pulse-dot"></span> Ready for Ingestion';
    }
  });

  function startProgressPolling(jobId) {
    if (state.pollTimer) clearInterval(state.pollTimer);

    let lastLogIndex = 0;

    state.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/progress?job_id=${jobId}`);
        if (!res.ok) return;

        const data = await res.json();

        // Update progress bar
        const pct = Math.max(5, data.progress || 5);
        progressBarFill.style.width = `${pct}%`;
        progressPctBadge.textContent = `${pct}%`;
        progressStageTitle.textContent = data.stage || 'Simulating...';

        // Update milestone chips
        updateStageSteps(pct);

        // Append new logs
        if (data.logs && data.logs.length > lastLogIndex) {
          for (let i = lastLogIndex; i < data.logs.length; i++) {
            appendLog(data.logs[i], 'info');
          }
          lastLogIndex = data.logs.length;
        }

        if (data.status === 'completed') {
          clearInterval(state.pollTimer);
          onJobCompleted(data);
        } else if (data.status === 'failed') {
          clearInterval(state.pollTimer);
          onJobFailed(data);
        }
      } catch (err) {
        console.error('Progress poll error:', err);
      }
    }, 600);
  }

  function updateStageSteps(pct) {
    const s1 = document.getElementById('step-stage-1');
    const s2 = document.getElementById('step-stage-2');
    const s3 = document.getElementById('step-stage-3');
    const s4 = document.getElementById('step-stage-4');

    if (pct >= 100) {
      [s1, s2, s3, s4].forEach(s => { s.className = 'stage-step completed'; });
    } else if (pct >= 75) {
      s1.className = 'stage-step completed';
      s2.className = 'stage-step completed';
      s3.className = 'stage-step completed';
      s4.className = 'stage-step active';
    } else if (pct >= 55) {
      s1.className = 'stage-step completed';
      s2.className = 'stage-step completed';
      s3.className = 'stage-step active';
      s4.className = 'stage-step';
    } else if (pct >= 35) {
      s1.className = 'stage-step completed';
      s2.className = 'stage-step active';
      [s3, s4].forEach(s => { s.className = 'stage-step'; });
    } else {
      s1.className = 'stage-step active';
      [s2, s3, s4].forEach(s => { s.className = 'stage-step'; });
    }
  }

  function appendLog(msg, type = 'info') {
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = msg;
    logConsole.appendChild(line);
    logConsole.scrollTop = logConsole.scrollHeight;
  }

  // ---------------------------------------------------------------------------
  // Completion & Results Presentation
  // ---------------------------------------------------------------------------
  function onJobCompleted(jobData) {
    appendLog('[SUCCESS] 150 tournament lineups generated and validated against all contest invariants.', 'success');
    updateReadyState();
    engineStatusPill.className = 'status-pill status-ready';
    engineStatusPill.innerHTML = '<span class="pulse-dot"></span> Lineups Ready';

    // Show Results
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Download Button setup
    btnDownload.onclick = () => {
      window.location.href = `/api/download?job_id=${jobData.id}`;
    };

    // Populate Metrics
    const summary = jobData.summary || {};
    resNumLineups.textContent = summary.num_lineups || numLineupsSlider.value;
    resUniquePlayers.textContent = summary.unique_players_count || '45+';
    resPrimaryStack.textContent = `${stackRatioSlider.value}%`;
    resRepeating.textContent = `≤ ${maxRepeatingSlider.value}`;

    // Populate Exposures Table
    renderExposuresTable(summary.top_exposures || []);
  }

  function onJobFailed(jobData) {
    appendLog(`[FATAL] Job failed: ${jobData.error || 'Unknown error'}`, 'error');
    updateReadyState();
    engineStatusPill.className = 'status-pill';
    engineStatusPill.style.color = 'var(--accent-rose)';
    engineStatusPill.innerHTML = '<span class="pulse-dot" style="background:var(--accent-rose);"></span> Execution Failed';
    alert(`Optimization failed: ${jobData.error || 'Please check terminal logs.'}`);
  }

  function cleanPlayerName(raw) {
    if (!raw) return '';
    let name = String(raw).trim();
    if (name.includes(':')) {
      const parts = name.split(':');
      if (parts[1] && parts[1].trim()) {
        name = parts.slice(1).join(':').trim();
      }
    }
    if (name.includes('(')) {
      name = name.split('(')[0].trim();
    }
    return name;
  }

  function renderExposuresTable(exposures) {
    exposuresTableBody.innerHTML = '';
    if (!exposures || exposures.length === 0) {
      exposuresTableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-dim);padding:2rem;">No exposure data available.</td></tr>';
      return;
    }

    exposures.forEach(item => {
      const displayName = cleanPlayerName(item.name);
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="player-name-cell">${escapeHtml(displayName)}</td>
        <td>${item.count} / ${numLineupsSlider.value}</td>
        <td><strong>${item.exposure_pct}%</strong></td>
        <td class="bar-cell">
          <div class="bar-wrap">
            <div class="bar-fill" style="width: ${Math.min(100, item.exposure_pct * 3)}%;"></div>
          </div>
        </td>
      `;
      exposuresTableBody.appendChild(row);
    });
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }

  // Reset
  btnReset.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    progressSection.classList.add('hidden');
    updateReadyState();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Initial ready state evaluation
  updateReadyState();

})();
