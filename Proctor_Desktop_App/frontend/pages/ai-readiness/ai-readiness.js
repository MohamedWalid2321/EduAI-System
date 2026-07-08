'use strict';

const EYE_CALIBRATION_INTERVAL_MS = 100;
const SPEECH_POLL_INTERVAL_MS = 1000;
const STATUS_POLL_INTERVAL_MS = 1000;
const COUNTDOWN_SECONDS = 3;

let stream = null;
let canvas = null;
let ctx = null;
let eyeTimer = null;
let speechTimer = null;
let statusTimer = null;
let eyeReady = false;
let speechReady = false;
let inFlight = false;
let calibrationStarted = false;  // true once user presses Start Calibration
let calibrationEverStarted = false; // true once at least one calibration cycle ran
let isRecalibrating = false;     // true during an active recalibration cycle
let countdownActive = false;     // true while the 3-second countdown is ticking
let faceWasLost = false;         // true if face disappeared mid-calibration; cleared on Recalibrate
let unsubscribeAi = null;

const eyeStatusEl    = document.getElementById('eyeStatus');
const speechStatusEl = document.getElementById('speechStatus');
const hintTextEl     = document.getElementById('hintText');
const continueBtn    = document.getElementById('continueBtn');
const startCalibBtn  = document.getElementById('startCalibBtn');
const recalibrateBtn = document.getElementById('recalibrateBtn');
const webcamFeed     = document.getElementById('webcamFeed');
const cameraUnavailable = document.getElementById('cameraUnavailable');

document.addEventListener('DOMContentLoaded', async () => {
  const result = await window.bridge.getExamSession();
  if (!result?.ok) {
    window.location.replace('../exam-code/index.html');
    return;
  }

  continueBtn.addEventListener('click', () => {
    window.location.href = '../exam-instructions/index.html';
  });

  // ── Start Calibration button ────────────────────────────────────────
  startCalibBtn.addEventListener('click', async () => {
    if (calibrationStarted || countdownActive) return;

    // Disable button during countdown
    startCalibBtn.disabled = true;
    await runCountdown(startCalibBtn, 'Start Calibration');
    startCalibBtn.disabled = false;

    calibrationStarted = true;
    calibrationEverStarted = true;

    startCalibBtn.classList.add('hidden');
    recalibrateBtn.classList.remove('hidden');

    setPill(eyeStatusEl, 'pending', 'Calibrating');
    hintTextEl.textContent = 'Look at the screen and keep your face visible.';

    startCalibrationLoops();
  });

  // ── Recalibrate button ─────────────────────────────────────────────
  recalibrateBtn.addEventListener('click', async () => {
    if (countdownActive) return;

    // Reset UI state immediately
    eyeReady = false;
    faceWasLost = false;  // Clear the face-lost interrupt — user acknowledged
    continueBtn.disabled = true;
    isRecalibrating = true;
    setPill(eyeStatusEl, 'pending', 'Recalibrating');
    hintTextEl.textContent = 'Recalibrating — look at the screen and keep your face visible.';
    recalibrateBtn.disabled = true;

    // Stop running loops for a clean restart
    stopCalibrationLoops();

    // Show 3-second countdown on the Recalibrate button
    await runCountdown(recalibrateBtn, 'Recalibrate');

    try {
      await window.bridge.aiRpc('recalibrateGaze', {}, { timeoutMs: 5000 });
    } catch {
      // Best-effort; the backend will recalibrate on the next predict call anyway
    }

    isRecalibrating = false;
    recalibrateBtn.disabled = false;
    calibrationStarted = true;
    startCalibrationLoops();
  });

  unsubscribeAi = window.bridge.onAiEvent((event) => {
    const data = event?.params || event;
    const msgType = event?.type || event?.method;

    // Handle serviceError notifications for eye-gaze.
    if (msgType === 'serviceError' && data?.service === 'eye-gaze') {
      eyeReady = false;
      isRecalibrating = false;
      setPill(eyeStatusEl, 'error', 'Unavailable');
      updateGate();
      return;
    }

    if (msgType !== 'detection') return;
    if (data?.service !== 'eye-gaze') return;

    const status = data?.payload?.status || 'initializing';

    if (status === 'no-face') {
      // Face not visible — show red warning; do NOT mark as ready
      eyeReady = false;
      // Mark that face was lost during an active calibration so we can pause
      // calibration when it reappears and prompt the user to recalibrate.
      if (calibrationStarted && calibrationEverStarted) {
        faceWasLost = true;
      }
      setPill(eyeStatusEl, 'error', 'No Face Detected');
    } else if (faceWasLost) {
      // Face has reappeared after being lost mid-calibration.
      // Stop calibration loops and wait for the user to explicitly press Recalibrate.
      faceWasLost = false;
      eyeReady = false;
      stopCalibrationLoops();
      setPill(eyeStatusEl, 'pending', 'Face Returned');
      hintTextEl.textContent = 'Your face reappeared — press Recalibrate to restart calibration.';
    } else if (status === 'initializing') {
      // Only update the pill when calibration is actually running
      if (calibrationStarted && !countdownActive) {
        if (isRecalibrating) {
          setPill(eyeStatusEl, 'pending', 'Recalibrating');
        } else {
          setPill(eyeStatusEl, 'pending', 'Calibrating');
        }
      }
    } else {
      // on-screen / away → calibration succeeded
      eyeReady = true;
      isRecalibrating = false;
      setPill(eyeStatusEl, 'ready', 'Ready');
    }

    updateGate();
  });

  await initCamera();

  // Start speech + status loops immediately; eye calibration waits for button.
  startSupportLoops();
});

window.addEventListener('beforeunload', () => {
  stopCalibrationLoops();
  stopSupportLoops();
  stream?.getTracks().forEach((t) => t.stop());
  if (unsubscribeAi) unsubscribeAi();
});

// ── 3-second countdown helper ────────────────────────────────────────────────
/**
 * Counts down from COUNTDOWN_SECONDS to 1 on a button, then restores its label.
 * Resolves when the countdown finishes.
 * @param {HTMLButtonElement} btn  Button to display countdown on
 * @param {string} originalLabel  Label to restore after countdown
 */
function runCountdown(btn, originalLabel) {
  return new Promise((resolve) => {
    countdownActive = true;
    let remaining = COUNTDOWN_SECONDS;

    btn.textContent = `Starting in ${remaining}…`;

    const tick = setInterval(() => {
      remaining -= 1;
      if (remaining > 0) {
        btn.textContent = `Starting in ${remaining}…`;
      } else {
        clearInterval(tick);
        btn.textContent = originalLabel;
        countdownActive = false;
        resolve();
      }
    }, 1000);
  });
}

async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamFeed.srcObject = stream;
    cameraUnavailable.classList.add('hidden');
  } catch {
    webcamFeed.classList.add('hidden');
    cameraUnavailable.classList.remove('hidden');
    setPill(eyeStatusEl, 'error', 'Camera Error');
    hintTextEl.textContent = 'Camera access is required to calibrate eye gaze.';
    startCalibBtn.disabled = true;
  }
}

// ── Eye calibration loop ─────────────────────────────────────────────────────
function startCalibrationLoops() {
  if (eyeTimer) return; // already running
  canvas = canvas || document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 240;
  ctx = ctx || canvas.getContext('2d');
  eyeTimer = setInterval(runEyeCalibrationTick, EYE_CALIBRATION_INTERVAL_MS);
}

function stopCalibrationLoops() {
  if (eyeTimer) { clearInterval(eyeTimer); eyeTimer = null; }
  calibrationStarted = false;
}

// ── Speech + status loops (run independently of calibration) ─────────────────
function startSupportLoops() {
  speechTimer = setInterval(runSpeechTick, SPEECH_POLL_INTERVAL_MS);
  statusTimer = setInterval(syncStatusTick, STATUS_POLL_INTERVAL_MS);
}

function stopSupportLoops() {
  if (speechTimer) { clearInterval(speechTimer); speechTimer = null; }
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}

async function runEyeCalibrationTick() {
  if (inFlight || !ctx || !webcamFeed || webcamFeed.readyState < 2) return;
  inFlight = true;
  try {
    ctx.drawImage(webcamFeed, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL('image/jpeg', 0.6);
    await window.bridge.aiRpc('predict', { service: 'eye-gaze', frame }, { timeoutMs: 5000 });
  } catch {
    if (calibrationStarted && !countdownActive) {
      setPill(eyeStatusEl, 'pending', isRecalibrating ? 'Recalibrating' : 'Calibrating');
    }
  } finally {
    inFlight = false;
    updateGate();
  }
}

async function runSpeechTick() {
  try {
    const res = await window.bridge.aiRpc(
      'predict',
      { service: 'speech-detection', frame: 'MIC_POLL' },
      { timeoutMs: 5000 }
    );
    if (res?.ok) {
      speechReady = true;
      setPill(speechStatusEl, 'ready', 'Ready');
    } else {
      speechReady = false;
      setPill(speechStatusEl, 'error', 'Not Running');
    }
  } catch {
    setPill(speechStatusEl, 'pending', 'Checking');
  } finally {
    updateGate();
  }
}

async function syncStatusTick() {
  try {
    const res = await window.bridge.aiRpc('queryStatus', {}, { timeoutMs: 5000 });
    if (!res?.ok) {
      setPill(eyeStatusEl, 'error', 'Not Running');
      setPill(speechStatusEl, 'error', 'Not Running');
      eyeReady = false;
      speechReady = false;
      return;
    }
    const statuses = res.result || {};
    const eyeStatus = statuses['eye-gaze'];
    const speechStatus = statuses['speech-detection'];

    if (eyeStatus !== 'running') {
      if (!eyeReady) {
        const label = eyeStatus === 'unavailable' ? 'Unavailable' : 'Not Running';
        setPill(eyeStatusEl, 'error', label);
      }
      eyeReady = false;
    }
    if (speechStatus === 'running' && !speechReady) {
      setPill(speechStatusEl, 'pending', 'Checking');
    } else if (speechStatus !== 'running') {
      const label = speechStatus === 'unavailable' ? 'Unavailable' : 'Not Running';
      setPill(speechStatusEl, 'error', label);
      speechReady = false;
    }
  } catch {
    // Ignore transient router errors.
  } finally {
    updateGate();
  }
}

function updateGate() {
  const allReady = eyeReady && speechReady;
  continueBtn.disabled = !allReady;

  if (allReady) {
    hintTextEl.textContent = 'All checks complete. You can continue to the exam.';
  } else if (!calibrationEverStarted) {
    hintTextEl.textContent = 'Press Start Calibration when you are ready, then look at the screen.';
  } else if (calibrationStarted) {
    hintTextEl.textContent = 'Look at the screen and keep your face visible.';
  }
  // Note: auto-navigate is intentionally removed — student clicks Continue manually.
}

function setPill(el, kind, text) {
  el.className = `pill ${kind}`;
  el.textContent = text;
}
