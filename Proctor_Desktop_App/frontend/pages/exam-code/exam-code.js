/**
 * exam-code.js — Form controller for the Lumina AI Exam Access page.
 *
 * Responsibilities:
 *   - Client-side validation (non-empty exam code, whitespace trim)
 *   - Loading state management (disable form, show spinner, change label)
 *   - 15-second client-side timeout via Promise.race (FR-012)
 *   - IPC call to window.bridge.startExam() and navigation on success
 *   - Safe error display via ERROR_MESSAGES lookup (no raw server text)
 *   - UNAUTHORIZED redirect to Login page (session cleared in main.js)
 *   - Logout handler (FR-010)
 *   - Async student name reveal on page init (FR-009)
 *
 * Security:
 *   - This module never calls fetch() or XMLHttpRequest directly.
 *   - All network communication goes through window.bridge.startExam() (IPC).
 *   - No token, exam ID, or raw server error is ever written to the DOM
 *     or logged to the console.
 */

// ---------------------------------------------------------------------------
// Error message map
// BridgeExamError codes → safe, sanitised display strings.
// These exact values match the contract in contracts/bridge-exam-access.md.
// ---------------------------------------------------------------------------

const ERROR_MESSAGES = {
  EXAM_NOT_FOUND:    'Exam code not found. Please check the code and try again.',
  ALREADY_ATTEMPTED: 'You have already attempted this exam.',
  BRIDGE_ERROR:      'Unable to reach the server. Please check your connection and try again.',
};

// ---------------------------------------------------------------------------
// DOM References (resolved after DOMContentLoaded)
// ---------------------------------------------------------------------------

let examForm;
let examCodeInput;
let submitBtn;
let btnText;
let btnArrow;
let btnSpinner;
let errorMessage;
let logoutBtn;
let studentNameEl;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  examForm      = document.getElementById('examForm');
  examCodeInput = document.getElementById('examCode');
  submitBtn     = document.getElementById('submitBtn');
  btnText       = document.getElementById('btnText');
  btnArrow      = document.getElementById('btnArrow');
  btnSpinner    = document.getElementById('btnSpinner');
  errorMessage  = document.getElementById('errorMessage');
  logoutBtn     = document.getElementById('logoutBtn');
  studentNameEl = document.getElementById('studentName');

  // FR-009: reveal student name asynchronously
  window.bridge.getSavedSession()
    .then((result) => {
      if (result?.ok && result.session?.userProfile?.firstName) {
        studentNameEl.textContent = result.session.userProfile.firstName;
      } else {
        studentNameEl.textContent = 'Student';
      }
    })
    .catch(() => { studentNameEl.textContent = 'Student'; });

  // FR-010: logout clears session and returns to Login page
  logoutBtn.addEventListener('click', async () => {
    await window.bridge.clearSession();
    window.location.href = '../login/index.html';
  });

  // Auto-uppercase every character as the user types (FR-002)
  examCodeInput.addEventListener('input', () => {
    const pos = examCodeInput.selectionStart;
    examCodeInput.value = examCodeInput.value.toUpperCase();
    examCodeInput.setSelectionRange(pos, pos);
    clearError();
  });

  // Form submission
  examForm.addEventListener('submit', handleSubmit);

  // Auto-focus the field on load
  examCodeInput.focus();
});

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

/**
 * Toggle the form between IDLE and LOADING states.
 *
 * LOADING: disables input and button, replaces arrow with spinner,
 *          changes label to "Checking…".
 * IDLE:    re-enables input and button, restores arrow and label.
 *
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
  submitBtn.disabled     = isLoading;
  examCodeInput.disabled = isLoading;
  btnText.textContent    = isLoading ? 'Checking\u2026' : 'JOIN EXAM';
  btnArrow.hidden        = isLoading;
  // Toggle CSP-safe hidden class — the `hidden` attribute is blocked by style-src 'self'
  btnSpinner.classList.toggle('btn-spinner--hidden', !isLoading);
  btnSpinner.setAttribute('aria-hidden', String(!isLoading));
}

function clearError() {
  errorMessage.textContent = '';
}

/**
 * Display a sanitised error from a BridgeExamError object.
 * Falls back to BRIDGE_ERROR if the code is unrecognised.
 *
 * @param {{ code?: string } | null | undefined} err
 */
function showError(err) {
  errorMessage.textContent =
    ERROR_MESSAGES[err?.code] ?? ERROR_MESSAGES.BRIDGE_ERROR;
}

// ---------------------------------------------------------------------------
// Form submission handler
// ---------------------------------------------------------------------------

/**
 * Handle the exam form submit event.
 * Flow: validate → LOADING → Promise.race(IPC, 15s timeout) → navigate or error.
 *
 * @param {SubmitEvent} event
 */
async function handleSubmit(event) {
  event.preventDefault();
  clearError();

  // FR-002: strip spaces/dashes, trim, uppercase
  const quizCode = examCodeInput.value.trim().replace(/[\s-]/g, '').toUpperCase();

  // FR-003: reject submissions with fewer than 8 characters
  if (quizCode.length < 8) {
    errorMessage.textContent = 'Please enter all 8 characters of your exam code.';
    examCodeInput.focus();
    return;
  }

  // FR-004: enter loading state (button disabled immediately — prevents FR-005 duplicate call)
  setLoading(true);

  // FR-012: 15-second client-side timeout
  const timeoutPromise = new Promise((resolve) =>
    setTimeout(() => resolve({ ok: false, timeout: true }), 15000)
  );

  let result;
  try {
    result = await Promise.race([
      window.bridge.startExam(quizCode),
      timeoutPromise,
    ]);
  } catch (_err) {
    result = { ok: false, error: { code: 'BRIDGE_ERROR' } };
  }

  // Timeout — treat as network failure
  if (result.timeout === true) {
    showError({ code: 'BRIDGE_ERROR' });
    setLoading(false);
    return;
  }

  // FR-008: UNAUTHORIZED — session was already cleared in main.js, just redirect
  if (result.redirect === 'login') {
    window.location.href = '../login/index.html';
    return;
  }

  // FR-006: navigate only when full response is available; validate shape (SC-004)
  if (result.ok === true) {
    if (!result.data?.attemptId || !Array.isArray(result.data?.questions)) {
      // Malformed response — do not navigate; show generic error
      showError({ code: 'BRIDGE_ERROR' });
      setLoading(false);
      return;
    }
    // examSession is already stored in main.js — no data to pass via URL.
    // Route through AI readiness so local model calibration finishes pre-exam.
    window.location.href = '../identity-verification/index.html';
    return;
  }

  // FR-007: typed error — input re-enabled, value preserved, error shown
  showError(result.error);
  setLoading(false);
}
