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
    hasCompletedRun: false,
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

  const gameStackRatioSlider = document.getElementById('game-stack-ratio');
  const valGameStackRatio = document.getElementById('val-game-stack-ratio');

  const selectionMetricSelect = document.getElementById('selection-metric');
  const valSelectionMetric = document.getElementById('val-selection-metric');

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

  // Strategy Presets
  const strategyPresetSelect = document.getElementById('strategy-preset');

  // Advanced Inputs
  const numCandidatesInput = document.getElementById('num-candidates');
  const numFieldInput = document.getElementById('num-field');
  const numTrialsInput = document.getElementById('num-trials');
  const randomnessDevInput = document.getElementById('randomness-dev');
  const maxTotalOwnershipInput = document.getElementById('max-total-ownership');
  const primaryStackSizeSelect = document.getElementById('primary-stack-size');
  const bringBackSizeSelect = document.getElementById('bring-back-size');
  const teamDefenseStackCheckbox = document.getElementById('team-defense-stack');
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
  const exposuresTableBody = document.getElementById('exposures-table-body');
  const qbExposuresTableBody = document.getElementById('qb-exposures-table-body');
  const nonQbExposuresTableBody = document.getElementById('non-qb-exposures-table-body');

  // Results Tabs & Lineup Inspector Elements
  const resultsReadyBanner = document.getElementById('results-ready-banner');
  const tabBtnExposures = document.getElementById('tab-btn-exposures');
  const tabBtnLineups = document.getElementById('tab-btn-lineups');
  const tabPaneExposures = document.getElementById('tab-pane-exposures');
  const tabPaneLineups = document.getElementById('tab-pane-lineups');
  const tabExposuresCount = document.getElementById('tab-exposures-count');
  const tabLineupsCount = document.getElementById('tab-lineups-count');

  const lineupEmptyState = document.getElementById('lineup-empty-state');
  const lineupInspectorContainer = document.getElementById('lineup-inspector-container');
  const btnPrevLineup = document.getElementById('btn-prev-lineup');
  const btnNextLineup = document.getElementById('btn-next-lineup');
  const lineupSelect = document.getElementById('lineup-select');
  const lineupCounterText = document.getElementById('lineup-counter-text');
  const lineupSalaryBadge = document.getElementById('lineup-salary-badge');
  const lineupFppgBadge = document.getElementById('lineup-fppg-badge');
  const lineupStackBadge = document.getElementById('lineup-stack-badge');
  const lineupSlotsBody = document.getElementById('lineup-slots-body');

  let currentLineups = [];
  let selectedLineupIdx = 0;

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

      metricFee.textContent = data.entry_fee_display || `$${data.entry_fee.toFixed(2)}`;

      const metricFeeSub = document.getElementById('metric-fee-sub');
      if (metricFeeSub) {
        if (data.entry_fee_sub) {
          metricFeeSub.textContent = data.entry_fee_sub;
          metricFeeSub.classList.remove('hidden');
        } else {
          metricFeeSub.textContent = '';
          metricFeeSub.classList.add('hidden');
        }
      }

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
        updateRepeatingHint();
      }

      // Auto-adapt Contest Profile based on entry count if available, or maintain selection
      if (data.num_entries) {
        if (!state.userSelectedPreset) {
          if (data.num_entries > 20) {
            applyStrategyPreset('large_gpp', true);
          } else if (data.num_entries > 3) {
            applyStrategyPreset('mid_gpp', true);
          } else {
            applyStrategyPreset('small_field', true);
          }
        } else {
          // Re-apply existing profile to respect Single Game vs Classic format changes
          applyStrategyPreset(strategyPresetSelect.value, true);
        }
      } else {
        applyStrategyPreset(strategyPresetSelect.value || 'large_gpp', true);
      }

      // Check if ready to optimize
      updateReadyState();
    } catch (err) {
      console.error('Inspection failed:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Contest Profiles (SaberSim-style Target Tournament Profiles)
  // ---------------------------------------------------------------------------
  const STRATEGY_PRESETS = {
    large_gpp: {
      name: 'Large-Field GPP (150-Max)',
      defaultEntries: 150,
      selectionMetric: 'ceiling',
      stackRatio: 80,
      gameStackRatio: 40,
      maxQbExposure: (isSg) => isSg ? 65 : 25,
      maxSkillExposure: (isSg) => isSg ? 65 : 25,
      maxDefExposure: (isSg) => isSg ? 30 : 20,
      maxRepeating: (isSg) => isSg ? 4 : 6,
      randomnessDev: 0.25,
    },
    mid_gpp: {
      name: 'Mid-Size GPP (20-Max)',
      defaultEntries: 20,
      selectionMetric: 'ceiling',
      stackRatio: 75,
      gameStackRatio: 35,
      maxQbExposure: (isSg) => isSg ? 65 : 30,
      maxSkillExposure: (isSg) => isSg ? 65 : 30,
      maxDefExposure: (isSg) => isSg ? 30 : 20,
      maxRepeating: (isSg) => isSg ? 4 : 6,
      randomnessDev: 0.25,
    },
    small_field: {
      name: 'Small-Field / Single Entry',
      defaultEntries: 1,
      selectionMetric: 'win_rate',
      stackRatio: 85,
      gameStackRatio: 45,
      maxQbExposure: (isSg) => isSg ? 60 : 25,
      maxSkillExposure: (isSg) => isSg ? 60 : 25,
      maxDefExposure: (isSg) => isSg ? 25 : 20,
      maxRepeating: (isSg) => isSg ? 3 : 5,
      randomnessDev: 0.30,
    },
    cash: {
      name: 'Cash Games (50/50s & Double-Ups)',
      defaultEntries: 1,
      selectionMetric: 'roi',
      stackRatio: 60,
      gameStackRatio: 20,
      maxQbExposure: (isSg) => isSg ? 50 : 35,
      maxSkillExposure: (isSg) => isSg ? 50 : 35,
      maxDefExposure: () => 20,
      maxRepeating: (isSg) => isSg ? 4 : 7,
      randomnessDev: 0.15,
    }
  };

  let isApplyingPreset = false;

  function applyStrategyPreset(presetKey, isAuto = false) {
    if (!presetKey || presetKey === 'custom') {
      return;
    }
    const preset = STRATEGY_PRESETS[presetKey];
    if (!preset) return;

    isApplyingPreset = true;
    const isSg = !!(state.inspection && state.inspection.is_single_game);

    if (numLineupsSlider && preset.defaultEntries) {
      let targetK = preset.defaultEntries;
      if (state.inspection && state.inspection.num_entries) {
        targetK = Math.min(targetK, state.inspection.num_entries);
      }
      if (targetK > parseInt(numLineupsSlider.max, 10)) {
        numLineupsSlider.max = targetK;
      }
      numLineupsSlider.value = targetK;
      if (valNumLineups) {
        valNumLineups.textContent = targetK;
      }
    }

    if (selectionMetricSelect) {
      selectionMetricSelect.value = preset.selectionMetric;
      const labels = { ceiling: 'Ceiling', roi: 'ROI', win_rate: 'Win Rate' };
      if (valSelectionMetric) valSelectionMetric.textContent = labels[preset.selectionMetric] || 'Ceiling';
    }

    if (stackRatioSlider && valStackRatio) {
      stackRatioSlider.value = preset.stackRatio;
      valStackRatio.textContent = `${preset.stackRatio}%`;
    }

    if (gameStackRatioSlider && valGameStackRatio) {
      gameStackRatioSlider.value = preset.gameStackRatio;
      valGameStackRatio.textContent = `${preset.gameStackRatio}%`;
    }

    const qbVal = preset.maxQbExposure(isSg);
    if (maxQbExposureSlider && valMaxQbExposure) {
      maxQbExposureSlider.value = qbVal;
      valMaxQbExposure.textContent = `${qbVal}%`;
    }

    const skillVal = preset.maxSkillExposure(isSg);
    if (maxSkillExposureSlider && valMaxSkillExposure) {
      maxSkillExposureSlider.value = skillVal;
      valMaxSkillExposure.textContent = `${skillVal}%`;
    }

    const defVal = preset.maxDefExposure(isSg);
    if (maxDefExposureSlider && valMaxDefExposure) {
      maxDefExposureSlider.value = defVal;
      valMaxDefExposure.textContent = `${defVal}%`;
    }

    const repVal = typeof preset.maxRepeating === 'function' ? preset.maxRepeating(isSg) : preset.maxRepeating;
    if (maxRepeatingSlider && valMaxRepeating) {
      maxRepeatingSlider.value = repVal;
      valMaxRepeating.textContent = repVal;
    }

    if (randomnessDevInput) {
      randomnessDevInput.value = preset.randomnessDev;
    }

    if (strategyPresetSelect) {
      strategyPresetSelect.value = presetKey;
    }

    updateCapCounts();
    updateRepeatingHint();
    isApplyingPreset = false;
  }

  function markCustomPreset() {
    if (isApplyingPreset) return;
    if (strategyPresetSelect && strategyPresetSelect.value !== 'custom') {
      strategyPresetSelect.value = 'custom';
    }
  }

  if (strategyPresetSelect) {
    strategyPresetSelect.addEventListener('change', (e) => {
      state.userSelectedPreset = true;
      applyStrategyPreset(e.target.value);
    });
  }

  // ---------------------------------------------------------------------------
  // Slider Event Listeners & Cap Count Updates
  // ---------------------------------------------------------------------------
  function updateCapCounts() {
    const k = parseInt(numLineupsSlider.value, 10);
    valNumLineups.textContent = k;
    document.querySelectorAll('.total-k-label').forEach((el) => {
      el.textContent = k;
    });

    const qbCap = parseFloat(maxQbExposureSlider.value) / 100;
    qbMaxCount.textContent = Math.floor(k * qbCap);

    const skillCap = parseFloat(maxSkillExposureSlider.value) / 100;
    skillMaxCount.textContent = Math.floor(k * skillCap);

    const defCap = parseFloat(maxDefExposureSlider.value) / 100;
    defMaxCount.textContent = Math.floor(k * defCap);
  }

  numLineupsSlider.addEventListener('input', () => {
    updateCapCounts();
  });

  stackRatioSlider.addEventListener('input', (e) => {
    markCustomPreset();
    valStackRatio.textContent = `${e.target.value}%`;
    const pVal = parseInt(e.target.value, 10);
    const gVal = parseInt(gameStackRatioSlider.value, 10);
    if (gVal > pVal) {
      gameStackRatioSlider.value = pVal;
      valGameStackRatio.textContent = `${pVal}%`;
    }
  });

  gameStackRatioSlider.addEventListener('input', (e) => {
    markCustomPreset();
    let gVal = parseInt(e.target.value, 10);
    const pVal = parseInt(stackRatioSlider.value, 10);
    if (gVal > pVal) {
      gVal = pVal;
      gameStackRatioSlider.value = gVal;
    }
    valGameStackRatio.textContent = `${gVal}%`;
  });

  selectionMetricSelect.addEventListener('change', (e) => {
    markCustomPreset();
    const labels = {
      ceiling: 'Ceiling',
      roi: 'ROI',
      win_rate: 'Win Rate',
    };
    valSelectionMetric.textContent = labels[e.target.value] || 'Ceiling';
  });

  maxQbExposureSlider.addEventListener('input', (e) => {
    markCustomPreset();
    valMaxQbExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  maxSkillExposureSlider.addEventListener('input', (e) => {
    markCustomPreset();
    valMaxSkillExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  maxDefExposureSlider.addEventListener('input', (e) => {
    markCustomPreset();
    valMaxDefExposure.textContent = `${e.target.value}%`;
    updateCapCounts();
  });

  function updateRepeatingHint() {
    const val = parseInt(maxRepeatingSlider.value, 10);
    valMaxRepeating.textContent = val;
    const diffEl = document.getElementById('repeating-diff-count');
    if (diffEl) {
      const rSize = (state.inspection && state.inspection.is_single_game)
        ? (state.inspection.platform === 'draftkings' ? 6 : 5)
        : 9;
      diffEl.textContent = Math.max(1, rSize - val);
    }
  }

  maxRepeatingSlider.addEventListener('input', () => {
    markCustomPreset();
    updateRepeatingHint();
  });

  if (randomnessDevInput) {
    randomnessDevInput.addEventListener('input', () => {
      markCustomPreset();
    });
  }



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
    if (resultsReadyBanner) resultsReadyBanner.classList.add('hidden');
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
        game_stack_ratio: parseFloat(gameStackRatioSlider.value) / 100,
        selection_metric: selectionMetricSelect.value,
        primary_stack_size: parseInt(primaryStackSizeSelect.value, 10),
        bring_back_size: parseInt(bringBackSizeSelect.value, 10),
        team_defense_stack: teamDefenseStackCheckbox.checked,
        include_team_defense_stack: teamDefenseStackCheckbox.checked,
        max_total_ownership: maxTotalOwnershipInput.value.trim() !== '' ? parseFloat(maxTotalOwnershipInput.value) : null,
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
    state.hasCompletedRun = true;
    const count = (jobData.summary && jobData.summary.num_lineups) || numLineupsSlider.value;
    appendLog(`[SUCCESS] ${count} tournament lineups generated and validated against all contest invariants.`, 'success');
    updateReadyState();
    engineStatusPill.className = 'status-pill status-ready';
    engineStatusPill.innerHTML = '<span class="pulse-dot"></span> Lineups Ready';

    // Show Results
    resultsSection.classList.remove('hidden');
    if (resultsReadyBanner) resultsReadyBanner.classList.remove('hidden');
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
    const totalCount = summary.num_lineups || numLineupsSlider.value;
    renderExposuresTable(summary, totalCount);

    // Populate Lineup Inspector
    renderLineupInspector(summary.lineups || [], totalCount);
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

  function renderExposuresTable(summaryData, totalCount) {
    const denom = totalCount || numLineupsSlider.value;

    const qbExposures = (summaryData && summaryData.top_qb_exposures) || [];
    const nonQbExposures = (summaryData && summaryData.top_non_qb_exposures) || [];
    const fallbackExposures = Array.isArray(summaryData) ? summaryData : ((summaryData && summaryData.top_exposures) || []);

    if (tabExposuresCount) {
      if (qbExposures.length > 0 || nonQbExposures.length > 0) {
        tabExposuresCount.textContent = `Top 10 QBs + Top 10 Core`;
      } else if (fallbackExposures.length > 0) {
        tabExposuresCount.textContent = `Top ${fallbackExposures.length}`;
      } else {
        tabExposuresCount.textContent = '-';
      }
    }

    // 1. Render Top 10 Quarterbacks Sub-Table
    if (qbExposuresTableBody) {
      qbExposuresTableBody.innerHTML = '';
      if (qbExposures.length === 0) {
        qbExposuresTableBody.innerHTML = '<tr><td colspan="5" class="empty-state-cell">Generate lineups to see results.</td></tr>';
      } else {
        qbExposures.forEach(item => {
          const row = document.createElement('tr');

          const nameCell = document.createElement('td');
          nameCell.className = 'player-name-cell';
          nameCell.textContent = cleanPlayerName(item.name);

          const teamCell = document.createElement('td');
          teamCell.textContent = item.team || '-';

          const countCell = document.createElement('td');
          countCell.textContent = `${item.count} / ${denom}`;

          const pctCell = document.createElement('td');
          const strong = document.createElement('strong');
          strong.textContent = `${item.exposure_pct}%`;
          pctCell.appendChild(strong);

          const barCell = document.createElement('td');
          barCell.className = 'bar-cell';
          const barWrap = document.createElement('div');
          barWrap.className = 'bar-wrap';
          const barFill = document.createElement('div');
          barFill.className = 'bar-fill';
          barFill.style.width = `${Math.min(100, item.exposure_pct * 3)}%`;
          barWrap.appendChild(barFill);
          barCell.appendChild(barWrap);

          row.appendChild(nameCell);
          row.appendChild(teamCell);
          row.appendChild(countCell);
          row.appendChild(pctCell);
          row.appendChild(barCell);
          qbExposuresTableBody.appendChild(row);
        });
      }
    }

    // 2. Render Top 10 Non-Quarterbacks Sub-Table
    if (nonQbExposuresTableBody) {
      nonQbExposuresTableBody.innerHTML = '';
      if (nonQbExposures.length === 0) {
        nonQbExposuresTableBody.innerHTML = '<tr><td colspan="6" class="empty-state-cell">Generate lineups to see results.</td></tr>';
      } else {
        nonQbExposures.forEach(item => {
          const row = document.createElement('tr');

          const nameCell = document.createElement('td');
          nameCell.className = 'player-name-cell';
          nameCell.textContent = cleanPlayerName(item.name);

          const posCell = document.createElement('td');
          const posTag = document.createElement('span');
          posTag.className = 'pos-tag-small';
          posTag.textContent = item.position || 'FLEX';
          posCell.appendChild(posTag);

          const teamCell = document.createElement('td');
          teamCell.textContent = item.team || '-';

          const countCell = document.createElement('td');
          countCell.textContent = `${item.count} / ${denom}`;

          const pctCell = document.createElement('td');
          const strong = document.createElement('strong');
          strong.textContent = `${item.exposure_pct}%`;
          pctCell.appendChild(strong);

          const barCell = document.createElement('td');
          barCell.className = 'bar-cell';
          const barWrap = document.createElement('div');
          barWrap.className = 'bar-wrap';
          const barFill = document.createElement('div');
          barFill.className = 'bar-fill';
          barFill.style.width = `${Math.min(100, item.exposure_pct * 3)}%`;
          barWrap.appendChild(barFill);
          barCell.appendChild(barWrap);

          row.appendChild(nameCell);
          row.appendChild(posCell);
          row.appendChild(teamCell);
          row.appendChild(countCell);
          row.appendChild(pctCell);
          row.appendChild(barCell);
          nonQbExposuresTableBody.appendChild(row);
        });
      }
    }

    // 3. Fallback Legacy Table Support
    if (exposuresTableBody) {
      exposuresTableBody.innerHTML = '';
      if (fallbackExposures.length === 0) {
        exposuresTableBody.innerHTML = '<tr><td colspan="4" class="empty-state-cell">Generate lineups to see results.</td></tr>';
      } else {
        fallbackExposures.forEach(item => {
          const row = document.createElement('tr');
          const nameCell = document.createElement('td');
          nameCell.className = 'player-name-cell';
          nameCell.textContent = cleanPlayerName(item.name);
          const countCell = document.createElement('td');
          countCell.textContent = `${item.count} / ${denom}`;
          const pctCell = document.createElement('td');
          const strong = document.createElement('strong');
          strong.textContent = `${item.exposure_pct}%`;
          pctCell.appendChild(strong);
          const barCell = document.createElement('td');
          barCell.className = 'bar-cell';
          const barWrap = document.createElement('div');
          barWrap.className = 'bar-wrap';
          const barFill = document.createElement('div');
          barFill.className = 'bar-fill';
          barFill.style.width = `${Math.min(100, item.exposure_pct * 3)}%`;
          barWrap.appendChild(barFill);
          barCell.appendChild(barWrap);
          row.appendChild(nameCell);
          row.appendChild(countCell);
          row.appendChild(pctCell);
          row.appendChild(barCell);
          exposuresTableBody.appendChild(row);
        });
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Tab Navigation & Lineup Inspector Implementation
  // ---------------------------------------------------------------------------
  if (tabBtnExposures && tabBtnLineups) {
    tabBtnExposures.addEventListener('click', () => {
      tabBtnExposures.classList.add('active');
      tabBtnLineups.classList.remove('active');
      tabPaneExposures.classList.add('active');
      tabPaneLineups.classList.remove('active');
    });

    tabBtnLineups.addEventListener('click', () => {
      tabBtnLineups.classList.add('active');
      tabBtnExposures.classList.remove('active');
      tabPaneLineups.classList.add('active');
      tabPaneExposures.classList.remove('active');
    });
  }

  if (btnPrevLineup) {
    btnPrevLineup.addEventListener('click', () => {
      if (selectedLineupIdx > 0) {
        displayLineup(selectedLineupIdx - 1);
      }
    });
  }

  if (btnNextLineup) {
    btnNextLineup.addEventListener('click', () => {
      if (selectedLineupIdx < currentLineups.length - 1) {
        displayLineup(selectedLineupIdx + 1);
      }
    });
  }

  if (lineupSelect) {
    lineupSelect.addEventListener('change', (e) => {
      displayLineup(parseInt(e.target.value, 10));
    });
  }

  function renderLineupInspector(lineups, targetK) {
    currentLineups = lineups || [];
    selectedLineupIdx = 0;

    if (currentLineups.length === 0) {
      if (tabLineupsCount) tabLineupsCount.textContent = '-';
      if (lineupEmptyState) {
        lineupEmptyState.classList.remove('hidden');
        const emptyText = lineupEmptyState.querySelector('.empty-state-text');
        if (emptyText) {
          emptyText.textContent = state.hasCompletedRun
            ? 'No lineup details received from server. Please restart ./run_optimizer --ui in your terminal and generate again.'
            : 'Generate lineups to see results.';
        }
      }
      if (lineupInspectorContainer) lineupInspectorContainer.classList.add('hidden');
      return;
    }

    if (lineupEmptyState) lineupEmptyState.classList.add('hidden');
    if (lineupInspectorContainer) lineupInspectorContainer.classList.remove('hidden');

    const total = currentLineups.length;
    if (tabLineupsCount) tabLineupsCount.textContent = total;

    if (!lineupSelect) return;
    lineupSelect.innerHTML = '';

    currentLineups.forEach((lu, idx) => {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = `Lineup #${lu.index || idx + 1}`;
      lineupSelect.appendChild(opt);
    });

    displayLineup(0);
  }

  function displayLineup(idx) {
    if (!currentLineups || currentLineups.length === 0) return;
    selectedLineupIdx = Math.max(0, Math.min(idx, currentLineups.length - 1));
    const lu = currentLineups[selectedLineupIdx];

    if (lineupSelect) lineupSelect.value = selectedLineupIdx;
    if (btnPrevLineup) btnPrevLineup.disabled = (selectedLineupIdx === 0);
    if (btnNextLineup) btnNextLineup.disabled = (selectedLineupIdx === currentLineups.length - 1);
    if (lineupCounterText) {
      lineupCounterText.textContent = `Lineup ${selectedLineupIdx + 1} of ${currentLineups.length}`;
    }

    // Update Badges
    if (lineupSalaryBadge) {
      const capStr = lu.salary_cap ? `$${lu.salary_cap.toLocaleString()}` : '$60,000';
      const salStr = lu.total_salary ? `$${lu.total_salary.toLocaleString()}` : '$0';
      lineupSalaryBadge.textContent = `Salary: ${salStr} / ${capStr}`;
    }

    if (lineupFppgBadge) {
      lineupFppgBadge.textContent = `${lu.total_fppg || 0} Proj FPPG`;
    }

    if (lineupStackBadge) {
      lineupStackBadge.textContent = lu.stack_badge || 'Balanced';
    }

    // Populate Slots Table
    if (!lineupSlotsBody) return;
    lineupSlotsBody.innerHTML = '';

    (lu.slots || []).forEach(slot => {
      const row = document.createElement('tr');

      // Slot Tag
      const slotTd = document.createElement('td');
      const slotTag = document.createElement('span');
      const slotUpper = (slot.slot || '').toUpperCase();
      slotTag.className = `slot-tag slot-${slotUpper.toLowerCase()}`;
      slotTag.textContent = slot.slot;
      slotTd.appendChild(slotTag);

      // Player Name
      const nameTd = document.createElement('td');
      nameTd.className = 'player-name-cell';
      nameTd.textContent = cleanPlayerName(slot.name);

      // Team
      const teamTd = document.createElement('td');
      teamTd.textContent = slot.team || '-';

      // Opponent
      const oppTd = document.createElement('td');
      oppTd.textContent = slot.opponent || '-';

      // Salary
      const salTd = document.createElement('td');
      salTd.className = 'salary-num';
      salTd.textContent = slot.salary ? `$${slot.salary.toLocaleString()}` : '-';

      // FPPG
      const fppgTd = document.createElement('td');
      fppgTd.className = 'fppg-num';
      fppgTd.textContent = slot.fppg ? slot.fppg.toFixed(1) : '-';

      // Correlation / Role
      const roleTd = document.createElement('td');
      if (slot.stack_type) {
        const pill = document.createElement('span');
        const stLower = slot.stack_type.toLowerCase();
        if (stLower.includes('qb')) {
          pill.className = 'role-pill role-qb';
          pill.textContent = '★ QB';
        } else if (stLower.includes('pass') || stLower.includes('stack')) {
          pill.className = 'role-pill role-stack';
          pill.textContent = '⚡ Pass-Catcher';
        } else if (stLower.includes('bring')) {
          pill.className = 'role-pill role-bringback';
          pill.textContent = '⮑ Bring-Back';
        } else {
          pill.className = 'role-pill';
          pill.textContent = slot.stack_type;
        }
        roleTd.appendChild(pill);
      } else {
        const standalonePill = document.createElement('span');
        standalonePill.className = 'role-pill role-standalone';
        standalonePill.textContent = '-';
        standalonePill.title = 'Uncorrelated Standalone';
        roleTd.appendChild(standalonePill);
      }

      row.appendChild(slotTd);
      row.appendChild(nameTd);
      row.appendChild(teamTd);
      row.appendChild(oppTd);
      row.appendChild(salTd);
      row.appendChild(fppgTd);
      row.appendChild(roleTd);
      lineupSlotsBody.appendChild(row);
    });
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }

  // Reset
  btnReset.addEventListener('click', () => {
    state.hasCompletedRun = false;
    if (resultsReadyBanner) resultsReadyBanner.classList.add('hidden');
    progressSection.classList.add('hidden');
    state.userSelectedPreset = false;
    applyStrategyPreset('large_gpp');
    updateReadyState();

    // Reset tabs to default active
    if (tabBtnExposures && tabBtnLineups) {
      tabBtnExposures.classList.add('active');
      tabBtnLineups.classList.remove('active');
      tabPaneExposures.classList.add('active');
      tabPaneLineups.classList.remove('active');
    }

    if (tabExposuresCount) tabExposuresCount.textContent = '-';
    if (tabLineupsCount) tabLineupsCount.textContent = '-';

    if (exposuresTableBody) {
      exposuresTableBody.innerHTML = '<tr><td colspan="4" class="empty-state-cell">Generate lineups to see results.</td></tr>';
    }
    if (qbExposuresTableBody) {
      qbExposuresTableBody.innerHTML = '<tr><td colspan="5" class="empty-state-cell">Generate lineups to see results.</td></tr>';
    }
    if (nonQbExposuresTableBody) {
      nonQbExposuresTableBody.innerHTML = '<tr><td colspan="6" class="empty-state-cell">Generate lineups to see results.</td></tr>';
    }

    renderLineupInspector([], 0);

    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Initial ready state evaluation & default preset load
  applyStrategyPreset('large_gpp');
  updateReadyState();
  renderLineupInspector([], 0);

})();

