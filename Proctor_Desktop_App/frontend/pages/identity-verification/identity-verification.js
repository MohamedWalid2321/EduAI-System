'use strict';

// ---------------------------------------------------------------------------
// Module-scope state
// ---------------------------------------------------------------------------

/** @type {MediaStream | null} */
let stream = null;

/** @type {string | null} Captured JPEG data URL (640×480 @ 0.85 quality) */
let capturedDataUrl = null;

/** @type {number | null} Quality loop interval handle */
let qualityIntervalId = null;

/** @type {HTMLCanvasElement | null} Offscreen canvas for quality sampling */
let qualityCanvas = null;

/** @type {CanvasRenderingContext2D | null} */
let qualityCtx = null;

// ---------------------------------------------------------------------------
// DOM references (resolved after DOMContentLoaded)
// ---------------------------------------------------------------------------

let webcamFeed      = null;
let capturePreview  = null;
let captureCanvas   = null;
let captureCtx      = null;
let captureBtn      = null;
let retakeBtn       = null;
let confirmBtn      = null;
let verifySpinner   = null;
let errorPill       = null;
let cameraDenied    = null;
let livePill        = null;
let lightingEl      = null;
let focusEl         = null;
let faceEl          = null;
let lightingValue   = null;
let focusValue      = null;
let faceValue       = null;

// ---------------------------------------------------------------------------
// Error message map
// ---------------------------------------------------------------------------

const ERROR_MESSAGES = {
  NO_FACE_DETECTED:      'No face detected \u2014 adjust your position and try again.',
  MULTIPLE_FACES:        'Multiple faces detected \u2014 make sure only you are visible.',
  SPOOF_DETECTED:        'Liveness check failed \u2014 please use a live webcam.',
  IDENTITY_MISMATCH:     'Your face does not match your official student record \u2014 please try again.',
  REFERENCE_FETCH_FAILED:'Could not load your official profile image. Please check your connection.',
  FACE_FRAME_ERROR:      'Identity comparison failed \u2014 please retake your photo.',
  ENROLLMENT_FAILED:     'Verification failed \u2014 please retake your photo.',
  SERVICE_NOT_RUNNING:   'Face recognition service is not available. Please try again shortly.',
  TIMEOUT:               'Verification timed out. Please try again.',
};

// ---------------------------------------------------------------------------
// DOMContentLoaded — entry point
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  // Resolve DOM references
  webcamFeed     = document.getElementById('webcamFeed');
  capturePreview = document.getElementById('capturePreview');
  captureCanvas  = document.getElementById('captureCanvas');
  captureCtx     = captureCanvas.getContext('2d');
  captureBtn     = document.getElementById('captureBtn');
  retakeBtn      = document.getElementById('retakeBtn');
  confirmBtn     = document.getElementById('confirmBtn');
  verifySpinner  = document.getElementById('verifySpinner');
  errorPill      = document.getElementById('errorPill');
  cameraDenied   = document.getElementById('cameraDenied');
  livePill       = document.getElementById('livePill');
  lightingEl     = document.getElementById('lightingIndicator');
  focusEl        = document.getElementById('focusIndicator');
  faceEl         = document.getElementById('faceIndicator');
  lightingValue  = document.getElementById('lightingValue');
  focusValue     = document.getElementById('focusValue');
  faceValue      = document.getElementById('faceValue');

  // T020 — guard: require active exam session
  const sessionResult = await window.bridge.getExamSession();
  if (!sessionResult?.ok) {
    window.location.replace('../exam-code/index.html');
    return;
  }

  // Attach button handlers
  captureBtn.addEventListener('click', handleCaptureClick);
  retakeBtn.addEventListener('click', handleRetakeClick);
  confirmBtn.addEventListener('click', handleConfirmClick);

  // Initialise camera then quality loop
  await initCamera();
  startQualityLoop();
});

// ---------------------------------------------------------------------------
// T021 — beforeunload: release resources
// ---------------------------------------------------------------------------

window.addEventListener('beforeunload', () => {
  stopQualityLoop();
  stream?.getTracks().forEach((t) => t.stop());
});

// ---------------------------------------------------------------------------
// T011 — initCamera
// ---------------------------------------------------------------------------

async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamFeed.srcObject = stream;
    cameraDenied.classList.add('hidden');
  } catch {
    webcamFeed.classList.add('hidden');
    cameraDenied.classList.remove('hidden');
    captureBtn.disabled = true;
    livePill.classList.add('hidden');
  }
}

// ---------------------------------------------------------------------------
// T012 — handleCaptureClick
// ---------------------------------------------------------------------------

function handleCaptureClick() {
  if (!stream) return;

  // Draw current frame to 640×480 canvas at JPEG quality 0.85
  captureCanvas.width  = 640;
  captureCanvas.height = 480;
  captureCtx.drawImage(webcamFeed, 0, 0, 640, 480);
  capturedDataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);

  // Freeze preview: show captured img, hide live video
  capturePreview.src = capturedDataUrl;
  capturePreview.classList.remove('hidden');
  webcamFeed.classList.add('hidden');
  livePill.classList.add('hidden');

  // UI state: CAPTURED
  captureBtn.classList.add('hidden');
  retakeBtn.classList.remove('hidden');
  confirmBtn.classList.remove('hidden');

  // Stop quality loop during review
  stopQualityLoop();
  clearErrorPill();
}

// ---------------------------------------------------------------------------
// T022 — handleRetakeClick
// ---------------------------------------------------------------------------

function handleRetakeClick() {
  capturedDataUrl = null;

  // Resume live feed
  capturePreview.classList.add('hidden');
  capturePreview.src = '';
  webcamFeed.classList.remove('hidden');
  webcamFeed.srcObject = stream;
  livePill.classList.remove('hidden');

  // UI state: LIVE
  captureBtn.classList.remove('hidden');
  retakeBtn.classList.add('hidden');
  confirmBtn.classList.add('hidden');

  clearErrorPill();
  startQualityLoop();
}

// ---------------------------------------------------------------------------
// T013 — handleConfirmClick
// ---------------------------------------------------------------------------

async function handleConfirmClick() {
  if (!capturedDataUrl) return;

  // Enter VERIFYING state
  setButtonsDisabled(true);
  verifySpinner.classList.remove('hidden');
  clearErrorPill();

  let result;
  try {
    result = await window.bridge.enrollReference(capturedDataUrl);
  } catch {
    result = { ok: false, error: { code: 'BRIDGE_ERROR', message: 'Unexpected error.' } };
  }

  if (result?.ok) {
    window.location.href = '../ai-readiness/index.html';
    return;
  }

  // Enrollment failed — return to LIVE state and show error
  verifySpinner.classList.add('hidden');
  setButtonsDisabled(false);
  showError(result?.error?.code);

  // Resume live feed for retake
  handleRetakeClick();
}

// ---------------------------------------------------------------------------
// T014 — showError
// ---------------------------------------------------------------------------

function showError(code) {
  const message = ERROR_MESSAGES[code] ?? 'An error occurred. Please try again.';
  errorPill.textContent = message;
  errorPill.classList.remove('hidden');
}

function clearErrorPill() {
  errorPill.textContent = '';
  errorPill.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setButtonsDisabled(disabled) {
  captureBtn.disabled  = disabled;
  retakeBtn.disabled   = disabled;
  confirmBtn.disabled  = disabled;
}

// ---------------------------------------------------------------------------
// T023 — analyzeQuality: pixel analysis for lighting, focus, face
// ---------------------------------------------------------------------------

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} width
 * @param {number} height
 * @returns {{ lighting: 'adequate'|'low', focus: 'sharp'|'blurry', face: 'detected'|'unknown' }}
 */
function analyzeQuality(ctx, width, height) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const pixelCount = width * height;

  let totalLuminance = 0;
  const grey = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    grey[i] = lum;
    totalLuminance += lum;
  }

  const avgLuminance = totalLuminance / pixelCount;
  const lighting = avgLuminance > 40 ? 'adequate' : 'low';

  // Laplacian-variance estimate for focus
  let laplacianSum = 0;
  let laplacianCount = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const lap =
        -grey[idx - width - 1] - grey[idx - width] - grey[idx - width + 1]
        - grey[idx - 1]        + 8 * grey[idx]     - grey[idx + 1]
        - grey[idx + width - 1] - grey[idx + width] - grey[idx + width + 1];
      laplacianSum += lap * lap;
      laplacianCount++;
    }
  }
  const laplacianVariance = laplacianCount > 0 ? laplacianSum / laplacianCount : 0;
  const focus = laplacianVariance > 30 ? 'sharp' : 'blurry';

  // Face heuristic: check for brightness contrast in upper-centre third of frame
  const faceRegionTop    = Math.floor(height * 0.1);
  const faceRegionBottom = Math.floor(height * 0.5);
  const faceRegionLeft   = Math.floor(width * 0.25);
  const faceRegionRight  = Math.floor(width * 0.75);
  let faceRegionLum = 0;
  let faceRegionCount = 0;
  for (let y = faceRegionTop; y < faceRegionBottom; y++) {
    for (let x = faceRegionLeft; x < faceRegionRight; x++) {
      faceRegionLum += grey[y * width + x];
      faceRegionCount++;
    }
  }
  const faceAvg = faceRegionCount > 0 ? faceRegionLum / faceRegionCount : 0;
  // Face is likely present if the face region is meaningfully brighter than overall average
  const face = (faceAvg > avgLuminance * 1.05 && faceAvg > 50) ? 'detected' : 'unknown';

  return { lighting, focus, face };
}

// ---------------------------------------------------------------------------
// T024 — startQualityLoop / stopQualityLoop
// ---------------------------------------------------------------------------

function startQualityLoop() {
  if (qualityIntervalId !== null) return; // already running

  qualityCanvas = document.createElement('canvas');
  qualityCanvas.width  = 160;
  qualityCanvas.height = 120;
  qualityCtx = qualityCanvas.getContext('2d');

  qualityIntervalId = setInterval(() => {
    if (!stream || !webcamFeed || webcamFeed.readyState < 2) return;

    qualityCtx.drawImage(webcamFeed, 0, 0, 160, 120);
    const result = analyzeQuality(qualityCtx, 160, 120);

    // Update lighting indicator
    setIndicator(lightingEl, lightingValue,
      result.lighting === 'adequate',
      result.lighting === 'adequate' ? 'Adequate' : 'Low');

    // Update focus indicator
    setIndicator(focusEl, focusValue,
      result.focus === 'sharp',
      result.focus === 'sharp' ? 'Sharp' : 'Blurry');

    // Update face indicator
    setIndicator(faceEl, faceValue,
      result.face === 'detected',
      result.face === 'detected' ? 'Detected' : 'Not Found');
  }, 500);
}

function stopQualityLoop() {
  if (qualityIntervalId !== null) {
    clearInterval(qualityIntervalId);
    qualityIntervalId = null;
  }
}

/**
 * Apply ok/warn class and label to a quality indicator element.
 * @param {HTMLElement} indicatorEl  The `.iv-indicator` element
 * @param {HTMLElement} valueEl      The `.iv-indicator-value` span
 * @param {boolean} isOk
 * @param {string} label
 */
function setIndicator(indicatorEl, valueEl, isOk, label) {
  indicatorEl.classList.toggle('indicator--ok',   isOk);
  indicatorEl.classList.toggle('indicator--warn', !isOk);
  valueEl.textContent = label;
}
