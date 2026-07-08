'use strict';

/**
 * offline.js — Offline Page & Lock Screen Controller
 * Feature: 014-offline-resilience
 *
 * Responsibilities:
 *   - Receive offline:state-changed IPC events via bridge.onOfflineStateChanged
 *   - Render countdown, answered count, disconnection count
 *   - Apply visual escalation (neutral → warning → critical)
 *   - Transition to lock screen on LOCKED state
 *   - Navigate back to exam page on ONLINE state (with resume data via sessionStorage)
 *   - Respond to snapshot heartbeat requests
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let countdownInterval = null;
let budgetRemainingMs = 0;
let budgetTotalMs     = 0;
let lastSnapshotData  = null; // { currentQuestionIndex, answers, frozenTimerSeconds }

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const offlinePage      = document.getElementById('offline-page');
const lockScreen       = document.getElementById('lock-screen');
const budgetCountdown  = document.getElementById('budget-countdown');
const answeredCount    = document.getElementById('answered-count');
const disconnCount     = document.getElementById('disconnection-count');
const lockReasonText   = document.getElementById('lock-reason-text');
const reconnectLabel   = document.getElementById('reconnect-label');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format milliseconds into MM:SS display string.
 * @param {number} ms
 * @returns {string}
 */
function formatMs(ms) {
  if (ms <= 0) return '00:00';
  const totalSec = Math.ceil(ms / 1000);
  const minutes  = Math.floor(totalSec / 60);
  const seconds  = totalSec % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Apply the correct escalation CSS class based on budget percentage remaining.
 * Thresholds: > 50% → neutral, ≤ 50% → warning, ≤ 20% → critical
 * @param {number} remainingMs
 * @param {number} totalMs
 */
function applyEscalationClass(remainingMs, totalMs) {
  if (!offlinePage) return;
  const pct = totalMs > 0 ? remainingMs / totalMs : 1;

  offlinePage.classList.remove('offline--neutral', 'offline--warning', 'offline--critical');

  if (pct <= 0.2) {
    offlinePage.classList.add('offline--critical');
  } else if (pct <= 0.5) {
    offlinePage.classList.add('offline--warning');
  } else {
    offlinePage.classList.add('offline--neutral');
  }
}

/**
 * Update the budget countdown display and escalation class.
 */
function renderCountdown() {
  if (budgetCountdown) {
    budgetCountdown.textContent = formatMs(budgetRemainingMs);
  }
  applyEscalationClass(budgetRemainingMs, budgetTotalMs);
}

/**
 * Start the local 1-second countdown tick.
 * Each tick decrements budgetRemainingMs and re-renders.
 */
function startLocalCountdown() {
  clearInterval(countdownInterval);
  countdownInterval = setInterval(() => {
    budgetRemainingMs = Math.max(0, budgetRemainingMs - 1000);
    renderCountdown();
    if (budgetRemainingMs <= 0) {
      clearInterval(countdownInterval);
      countdownInterval = null;
    }
  }, 1_000);
}

// ---------------------------------------------------------------------------
// State change handler
// ---------------------------------------------------------------------------

/**
 * Handle an OfflineStateEvent from the main process.
 * @param {object} payload
 */
function handleStateChange(payload) {
  const { state, budgetRemainingMs: rem, budgetTotalMs: total,
          disconnectionCount, maxDisconnections, answeredCount: answered,
          lockReason } = payload;

  // Update module-level budget values
  budgetRemainingMs = rem ?? 0;
  budgetTotalMs     = total ?? 1;

  if (state === 'OFFLINE') {
    // Show the offline page (already showing — just refresh data)
    if (answeredCount) answeredCount.textContent = answered ?? 0;
    if (disconnCount) {
      disconnCount.textContent = `${disconnectionCount ?? 0} / ${maxDisconnections ?? '?'}`;
    }
    renderCountdown();
    startLocalCountdown();

  } else if (state === 'ONLINE') {
    // Reconnection — stop countdown, fade out, navigate back to exam
    clearInterval(countdownInterval);
    countdownInterval = null;

    // Persist resume data so exam.js can restore state
    if (lastSnapshotData) {
      try {
        sessionStorage.setItem('offlineResumeData', JSON.stringify({
          currentQuestionIndex: lastSnapshotData.currentQuestionIndex,
          answers:              lastSnapshotData.answers,
          frozenTimerSeconds:   lastSnapshotData.frozenTimerSeconds,
          flagged:              lastSnapshotData.flagged,
        }));
        sessionStorage.setItem('offlineResume', '1');
      } catch (_e) { /* best-effort */ }
    }

    // Animate fade-out then navigate
    if (offlinePage) {
      offlinePage.classList.add('fading-out');
      setTimeout(() => {
        window.location.href = '../exam/index.html';
      }, 420);
    } else {
      window.location.href = '../exam/index.html';
    }

  } else if (state === 'LOCKED') {
    // Transform to lock screen
    clearInterval(countdownInterval);
    countdownInterval = null;
    showLockScreen(lockReason);

  } else if (state === 'SUBMITTED') {
    // Auto-submit succeeded — update lock screen
    if (lockScreen) {
      lockScreen.classList.add('submitted');
      if (reconnectLabel) {
        reconnectLabel.textContent = 'Your answers have been submitted. You may close this window.';
      }
    }
  }
}

/**
 * Show the lock screen, hiding the offline page.
 * @param {'budget_exhausted'|'disconnection_limit'|'auto_submitted'|null} reason
 */
function showLockScreen(reason) {
  if (offlinePage) offlinePage.style.display = 'none';
  if (!lockScreen) return;

  lockScreen.classList.remove('hidden');

  const reasonMessages = {
    budget_exhausted:    'Your exam has been locked because your offline time allowance has been exhausted.',
    disconnection_limit: 'Your exam has been locked because the maximum number of disconnections has been reached.',
    auto_submitted:      'Your answers have been automatically submitted.',
  };

  if (lockReasonText) {
    lockReasonText.textContent = reasonMessages[reason] ?? reasonMessages.budget_exhausted;
  }
  if (reconnectLabel) {
    reconnectLabel.textContent = 'Waiting for connection to submit your answers…';
  }
}

// ---------------------------------------------------------------------------
// Snapshot heartbeat response
// ---------------------------------------------------------------------------

/**
 * Respond to a snapshot heartbeat request from the main process.
 * Reads last known snapshot data from sessionStorage (set by exam.js on init).
 */
function handleSnapshotRequest() {
  try {
    const raw = sessionStorage.getItem('offlineResumeData');
    if (raw) {
      lastSnapshotData = JSON.parse(raw);
    }
  } catch (_e) { /* best-effort */ }

  if (lastSnapshotData && window.bridge) {
    window.bridge.sendSnapshotData(lastSnapshotData);
  }
}

// ---------------------------------------------------------------------------
// Initialise — check if arriving in LOCKED state (crash recovery)
// ---------------------------------------------------------------------------

function init() {
  // Check if we landed here due to a crash recovery with a locked session
  const crashLocked = sessionStorage.getItem('offlineCrashLocked');
  if (crashLocked === '1') {
    sessionStorage.removeItem('offlineCrashLocked');
    showLockScreen(sessionStorage.getItem('offlineLockReason') ?? 'budget_exhausted');
    return;
  }

  // Restore last known snapshot data for heartbeat responses
  try {
    const raw = sessionStorage.getItem('offlineResumeData');
    if (raw) lastSnapshotData = JSON.parse(raw);
  } catch (_e) { /* best-effort */ }

  // Set initial display from sessionStorage (written by exam.js before navigation)
  try {
    const initPayload = sessionStorage.getItem('offlineInitPayload');
    if (initPayload) {
      const p = JSON.parse(initPayload);
      sessionStorage.removeItem('offlineInitPayload');
      handleStateChange({ state: 'OFFLINE', ...p });
    }
  } catch (_e) { /* use defaults */ }

  // Subscribe to live state changes
  if (window.bridge?.onOfflineStateChanged) {
    window.bridge.onOfflineStateChanged(handleStateChange);
  }

  // Subscribe to heartbeat requests
  if (window.bridge?.onSnapshotRequest) {
    window.bridge.onSnapshotRequest(handleSnapshotRequest);
  }
}

// Run after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
