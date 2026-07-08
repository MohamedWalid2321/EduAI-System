/**
 * login.js — Form controller for the Lumina AI login page.
 *
 * Responsibilities:
 *   - Client-side validation (email format, non-empty password)
 *   - LOADING state management (disable form, show spinner)
 *   - IPC call to window.bridge.login() and navigation on success
 *   - Safe error display via ERROR_MESSAGES lookup (no raw server text)
 *   - Enter-key handling on both input fields
 *   - Forgot password / external link delegation to window.bridge.openExternal()
 *
 * Security:
 *   - This module never calls fetch() or XMLHttpRequest directly.
 *   - All network communication goes through window.bridge.login() (IPC).
 *   - No token, password, or raw server error is ever written to the DOM
 *     or logged to the console.
 */

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const loginForm        = document.getElementById('login-form');
const emailInput       = document.getElementById('email-input');
const passwordInput    = document.getElementById('password-input');
const passwordToggle   = document.getElementById('password-toggle');
const iconEyeOpen      = document.getElementById('icon-eye-open');
const iconEyeOff       = document.getElementById('icon-eye-off');
const rememberCheckbox = document.getElementById('remember-checkbox');
const loginBtn         = document.getElementById('login-btn');
const loginBtnText     = document.getElementById('login-btn-text');
const loginSpinner     = document.getElementById('login-spinner');
const loginError       = document.getElementById('login-error');
const forgotLink       = document.getElementById('forgot-link');
const fabInfo          = document.getElementById('fab-info');

// ---------------------------------------------------------------------------
// Error Message Map
// BridgeLoginError codes → safe, sanitised display strings.
// These exact values are defined in data-model.md "Static message map".
// ---------------------------------------------------------------------------

const ERROR_MESSAGES = {
  INVALID_CREDENTIALS: 'Invalid email or password.',
  EMAIL_NOT_CONFIRMED: 'Please confirm your email address before logging in.',
  LOCKED_OUT:          'Your account is temporarily locked. Please contact your administrator.',
  ACCOUNT_DISABLED:    'Your account has been disabled. Please contact your administrator.',
  BRIDGE_ERROR:        'Unable to reach the server. Please check your connection and try again.',
};

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

function showError(message) {
  loginError.textContent = message;
}

function clearError() {
  loginError.textContent = '';
}

/**
 * Toggle the form between IDLE and LOADING states.
 *
 * LOADING: disables all inputs, hides button text, shows spinner.
 * IDLE:    re-enables all inputs, shows button text, hides spinner.
 *
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
  loginBtn.disabled         = isLoading;
  emailInput.disabled       = isLoading;
  passwordInput.disabled    = isLoading;
  passwordToggle.disabled   = isLoading;
  rememberCheckbox.disabled = isLoading;
  loginBtnText.hidden       = isLoading;
  loginSpinner.hidden       = !isLoading;
  loginSpinner.setAttribute('aria-hidden', String(!isLoading));
}

// ---------------------------------------------------------------------------
// Password visibility toggle
// ---------------------------------------------------------------------------

/**
 * Toggle the password field between masked (type=password) and visible (type=text).
 * Swaps the eye / eye-off SVG icons and updates the button's aria-label.
 */
function togglePasswordVisibility() {
  const isHidden = passwordInput.type === 'password';
  passwordInput.type = isHidden ? 'text' : 'password';
  // Toggle the CSP-safe class — matches the .pw-icon--hidden rule in login.css
  iconEyeOpen.classList.toggle('pw-icon--hidden', isHidden);   // hide when visible
  iconEyeOff.classList.toggle('pw-icon--hidden', !isHidden);   // show when visible
  passwordToggle.setAttribute(
    'aria-label',
    isHidden ? 'Hide password' : 'Show password'
  );
}

// ---------------------------------------------------------------------------
// Client-side validation (Layer 1 — runs before any IPC call)
// ---------------------------------------------------------------------------

/**
 * Validate email and password inputs.
 * Displays an inline error and focuses the offending field on failure.
 *
 * @param {string} email     Trimmed email value
 * @param {string} password  Raw password value
 * @returns {boolean} true if inputs are valid, false otherwise.
 */
function validateInputs(email, password) {
  if (!email) {
    showError('Please enter your email address.');
    emailInput.focus();
    return false;
  }

  const atIndex = email.indexOf('@');
  const hasDotAfterAt = atIndex > 0 && email.indexOf('.', atIndex + 1) > atIndex + 1;
  if (!hasDotAfterAt) {
    showError('Please enter a valid email address.');
    emailInput.focus();
    return false;
  }

  if (!password) {
    showError('Please enter your password.');
    passwordInput.focus();
    return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------------

/**
 * Handle the login form submit event.
 * Flow: validate → LOADING state → IPC bridge:login → navigate or show error.
 *
 * @param {SubmitEvent} e
 */
async function submitForm(e) {
  e.preventDefault();
  clearError();

  const email    = emailInput.value.trim();
  const password = passwordInput.value;

  if (!validateInputs(email, password)) {
    return;
  }

  setLoading(true);

  let result;
  try {
    result = await window.bridge.login({
      email,
      password,
      remember: rememberCheckbox.checked,
    });
  } catch (_err) {
    // IPC channel failure (should never happen in normal operation)
    setLoading(false);
    showError(ERROR_MESSAGES.BRIDGE_ERROR);
    return;
  }

  setLoading(false);

  if (result.ok) {
    // Success → navigate to exam-code page
    window.location.href = '../exam-code/index.html';
  } else {
    // Bridge returned a typed error — look up safe display string
    const code = result.error?.code;
    showError(ERROR_MESSAGES[code] ?? ERROR_MESSAGES.BRIDGE_ERROR);
  }
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

loginForm.addEventListener('submit', submitForm);

// Email field: Enter moves focus to password if password is empty,
// otherwise submits the form.
emailInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    if (!passwordInput.value) {
      passwordInput.focus();
    } else {
      loginForm.requestSubmit();
    }
  }
});

// Password field: Enter submits the form.
passwordInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    loginForm.requestSubmit();
  }
});

// Forgot password — open external URL when configured.
// The URL is intentionally left unconfigured here; supply it when the LMS
// "forgot password" endpoint URL is known, e.g.:
//   window.bridge.openExternal('https://your-lms.edu/forgot-password')
forgotLink.addEventListener('click', (e) => {
  e.preventDefault();
  // Placeholder — no-op until forgot-password URL is configured
});

// Password visibility toggle
passwordToggle.addEventListener('click', togglePasswordVisibility);

// FAB info button
fabInfo.addEventListener('click', () => {
  // Placeholder — info panel to be implemented in a future spec
});
