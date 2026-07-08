'use strict';

/**
 * preload.js — Electron preload script for Lumina AI.
 *
 * Exposes a minimal, safe IPC surface to renderer pages via contextBridge.
 * No raw Node.js or Electron APIs are exposed directly to the renderer.
 * All communication goes through the typed 'bridge' API below.
 *
 * Security requirements (constitution.md — Security by Default):
 *   - contextIsolation: true  (set in BrowserWindow webPreferences)
 *   - nodeIntegration: false  (set in BrowserWindow webPreferences)
 *   - Only explicitly whitelisted IPC channels are forwarded
 */

const { contextBridge, ipcRenderer } = require('electron');

/**
 * Whitelisted IPC channels the renderer is allowed to listen on.
 * Adding a channel here is the only way a renderer page can receive events.
 */
const ALLOWED_RECEIVE_CHANNELS = [
  'bridge:status',    // BridgeStatus state updates from main process
  'bridge:ai-event',  // AI proctoring events (detections, alerts, risk scores)
  'clip:upload-error', // Violation clip upload failure notification
  'lockdown:vm-detected',             // VM or remote desktop detected during exam
  'lockdown:screen-capture-detected', // Screen-capture application detected during exam
  // spec 014 — offline resilience channels
  'offline:state-changed',    // State machine transitions: ONLINE/OFFLINE/LOCKED
  'offline:snapshot-request', // Heartbeat request from main to renderer for snapshot data
  'proctoring:pause',         // Pause all AI service polling (while offline)
  'proctoring:resume',        // Resume all AI service polling (after reconnect)
];

/**
 * Whitelisted IPC channels the renderer is allowed to invoke (request/reply).
 */
const ALLOWED_INVOKE_CHANNELS = [
  'bridge:get-status',          // Returns current BridgeStatus state string
  'bridge:ai-rpc',               // JSON-RPC calls to the AI router
  'bridge:login',               // Proxy login credentials to Python bridge
  'bridge:get-saved-session',   // Restore previously saved session
  'bridge:clear-session',       // Delete all session data
  'bridge:open-external',       // Open https:// URL in system browser
  'bridge:start-exam',          // Validate exam code and start attempt
  'bridge:get-exam-session',    // Retrieve stored ExamSession (used by Exam page)
  'bridge:submit-exam',         // Submit exam answers and get result
  'bridge:get-submit-result',   // Retrieve stored SubmitResult (used by Result page)
  'bridge:get-result',          // Recover result from LMS when submitResult is absent
  'bridge:clear-submit-result', // Clear submitResult + examSession; navigate to Exam Access
  'bridge:get-session-log',     // Retrieve session JSONL log lines
  'bridge:export-pdf',          // Export current page to PDF
  'bridge:enroll-reference',    // Enroll a captured reference photo for face recognition
  'bridge:get-enrollment-status', // Check whether enrollment succeeded
  'bridge:get-ui-config',         // Return ui section of config.json (intervals etc.)
  'save-and-upload-clip',          // Save webm blob to temp file and upload via Python bridge
  'lockdown:start',                // Activate fullscreen/kiosk lockdown (called from exam instructions)
];

contextBridge.exposeInMainWorld('bridge', {
  /**
   * Subscribe to AI proctoring events from the Python bridge.
   * @param {(event: object) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onAiEvent(callback) {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on('bridge:ai-event', handler);
    return () => ipcRenderer.removeListener('bridge:ai-event', handler);
  },

  /**
   * Retrieve the JSONL log for a specific session.
   */
  getSessionLog(sessionId) {
    return ipcRenderer.invoke('bridge:get-session-log', { sessionId });
  },

  /**
   * Export the current view to a PDF.
   */
  exportPdf(filename) {
    return ipcRenderer.invoke('bridge:export-pdf', { filename });
  },

  /**
   * Subscribe to bridge status events from the main process.
   *
   * The callback receives an event payload:
   *   { type: 'ready' | 'failed' | 'crashed' | 'config-error', code?: string, message?: string }
   *
   * Returns a cleanup function — call it to unsubscribe.
   *
   * @param {(payload: object) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onBridgeStatus(callback) {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on('bridge:status', handler);
    // Return cleanup so callers can avoid memory leaks on page navigation
    return () => ipcRenderer.removeListener('bridge:status', handler);
  },

  /**
   * Query the current bridge state synchronously via IPC invoke.
   * Returns one of: 'idle' | 'starting' | 'ready' | 'failed' | 'crashed' | 'stopping'
   *
   * @returns {Promise<string>}
   */
  getBridgeStatus() {
    return ipcRenderer.invoke('bridge:get-status');
  },

  /**
   * Send a JSON-RPC request to the AI router.
   *
   * @param {string} method
   * @param {object} params
   * @param {{ timeoutMs?: number }} [options]
   * @returns {Promise<{ok: true, result: object} | {ok: false, error: object}>}
   */
  aiRpc(method, params, options = {}) {
    return ipcRenderer.invoke('bridge:ai-rpc', {
      method,
      params,
      timeoutMs: options.timeoutMs,
    });
  },

  /**
   * Submit login credentials to the main process for forwarding to the bridge.
   *
   * @param {{ email: string, password: string, remember: boolean }} creds
   * @returns {Promise<{ok: true, data: object} | {ok: false, error: object}>}
   */
  login(creds) {
    return ipcRenderer.invoke('bridge:login', creds);
  },

  /**
   * Retrieve a previously saved session from the OS keychain.
   *
   * @returns {Promise<{ok: true, session: object} | {ok: false}>}
   */
  getSavedSession() {
    return ipcRenderer.invoke('bridge:get-saved-session');
  },

  /**
   * Delete all keytar session entries and in-memory session data.
   *
   * @returns {Promise<{ok: true}>}
   */
  clearSession() {
    return ipcRenderer.invoke('bridge:clear-session');
  },

  /**
   * Open an https:// URL in the system default browser.
   * Non-https URLs are silently ignored by the main process.
   *
   * @param {string} url
   * @returns {Promise<void>}
   */
  openExternal(url) {
    return ipcRenderer.invoke('bridge:open-external', url);
  },

  /**
   * Submit an exam code to the main process for forwarding to the bridge.
   * The JWT is attached by main.js — the renderer never holds tokens.
   *
   * @param {string} quizCode  Exam code entered by the student (whitespace trimmed)
   * @returns {Promise<
   *   {ok: true, data: object} |
   *   {ok: false, redirect: 'login'} |
   *   {ok: false, error: object}
   * >}
   */
  startExam(quizCode) {
    return ipcRenderer.invoke('bridge:start-exam', { quizCode });
  },

  /**
   * Retrieve the stored ExamSession from main.js (used by the Exam page).
   *
   * @returns {Promise<{ok: true, session: object} | {ok: false}>}
   */
  getExamSession() {
    return ipcRenderer.invoke('bridge:get-exam-session');
  },

  /**
   * Submit exam answers to the main process for forwarding to the bridge.
   * The JWT is attached by main.js — the renderer never holds tokens.
   *
   * @param {Array<{questionId: number, choiceId: number}>} answers
   * @returns {Promise<
   *   {ok: true, data: object} |
   *   {ok: false, redirect: 'login'} |
   *   {ok: false, error: object}
   * >}
   */
  submitExam(answers) {
    return ipcRenderer.invoke('bridge:submit-exam', { answers });
  },

  /**
   * Retrieve the stored SubmitResult from main.js (used by the Result page).
   *
   * @returns {Promise<{ok: true, data: object} | {ok: false}>}
   */
  getSubmitResult() {
    return ipcRenderer.invoke('bridge:get-submit-result');
  },

  /**
   * Recover the exam result from the LMS when submitResult is absent in memory.
   * main.js reads examSession.attemptId and the stored JWT; renderer never holds tokens.
   *
   * @returns {Promise<
   *   {ok: true, data: object} |
   *   {ok: false, redirect: 'login'} |
   *   {ok: false, error: object}
   * >}
   */
  getResult() {
    return ipcRenderer.invoke('bridge:get-result');
  },

  /**
   * Clear the in-memory submitResult and examSession in main.js, then
   * navigate the Electron window to the Exam Access page.
   * The authenticated session (keytar / sessionMemory) is NOT cleared.
   *
   * @returns {Promise<{ok: true}>}
   */
  clearSubmitResult() {
    return ipcRenderer.invoke('bridge:clear-submit-result');
  },

  /**
   * Enroll a captured reference photo for face recognition.
   * Main process chains face-detect → enroll via the AI router.
   *
   * @param {string} frame  Base64 data URL of the captured JPEG
   * @returns {Promise<{ok: true} | {ok: false, error: {code: string, message: string}}>}
   */
  enrollReference(frame) {
    return ipcRenderer.invoke('bridge:enroll-reference', { frame });
  },

  /**
   * Check whether the student has successfully enrolled a reference photo.
   *
   * @returns {Promise<{enrolled: boolean}>}
   */
  getEnrollmentStatus() {
    return ipcRenderer.invoke('bridge:get-enrollment-status');
  },

  /**
   * Return the ui section of config.json (polling intervals, etc.) to the renderer.
   * Falls back to { ok: false } if config cannot be read.
   *
   * @returns {Promise<{ok: true, ui: object} | {ok: false}>}
   */
  getUiConfig() {
    return ipcRenderer.invoke('bridge:get-ui-config');
  },

  /**
   * Send a captured violation clip (as ArrayBuffer + metadata) to the main process
   * for temp-file write, ffmpeg encode, and Bunny CDN upload via the Python bridge.
   *
   * Security: no CDN credentials are passed here. The main process reads them
   * exclusively from environment variables inside the Python bridge process.
   *
   * @param {{ blobArrayBuffer: ArrayBuffer | null, metadata: object }} payload
   * @returns {Promise<{ok: true, result: object} | {ok: false, error: object}>}
   */
  saveAndUploadClip(payload) {
    return ipcRenderer.invoke('save-and-upload-clip', payload);
  },

  /**
   * Subscribe to clip upload failure events pushed from the main process.
   * Fires when the Python bridge exhausts all upload retries.
   *
   * @param {(payload: {studentId: string, examAttemptId: string, sessionId: string, reasonCode: string, timestamp: string}) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onClipUploadError(callback) {
    const handler = (_e, payload) => callback(payload);
    ipcRenderer.on('clip:upload-error', handler);
    return () => ipcRenderer.removeListener('clip:upload-error', handler);
  },

  /**
   * Subscribe to VM or remote desktop detection events from the main process.
   * Fired when POST /check-environment detects vm_detected or rdp_detected.
   *
   * @param {(payload: {reason: string}) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onLockdownVmDetected(callback) {
    const handler = (_e, payload) => callback(payload);
    ipcRenderer.on('lockdown:vm-detected', handler);
    return () => ipcRenderer.removeListener('lockdown:vm-detected', handler);
  },

  /**
   * Subscribe to screen-capture application detection events from the main process.
   * Fired when POST /check-environment detects screen_capture_detected.
   *
   * @param {(payload: {reason: string}) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onLockdownCaptureDetected(callback) {
    const handler = (_e, payload) => callback(payload);
    ipcRenderer.on('lockdown:screen-capture-detected', handler);
    return () => ipcRenderer.removeListener('lockdown:screen-capture-detected', handler);
  },

  /**
   * Activate fullscreen/kiosk lockdown controls.
   * Called from the Exam Instructions page when the student clicks "Start Exam",
   * ensuring lockdown only applies during the actual exam (not during pre-exam phases).
   *
   * @returns {Promise<void>}
   */
  startLockdown() {
    return ipcRenderer.invoke('lockdown:start');
  },

  /**
   * Subscribe to offline state change events (ONLINE → OFFLINE → LOCKED).
   * Payload: OfflineStateEvent (see contracts/offline-ipc.md)
   *
   * @param {(payload: object) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onOfflineStateChanged(callback) {
    const handler = (_e, payload) => callback(payload);
    ipcRenderer.on('offline:state-changed', handler);
    return () => ipcRenderer.removeListener('offline:state-changed', handler);
  },

  /**
   * Subscribe to snapshot heartbeat requests from the main process.
   * Renderer should respond immediately with sendSnapshotData.
   *
   * @param {(payload: null) => void} callback
   * @returns {() => void} unsubscribe function
   */
  onSnapshotRequest(callback) {
    const handler = () => callback();
    ipcRenderer.on('offline:snapshot-request', handler);
    return () => ipcRenderer.removeListener('offline:snapshot-request', handler);
  },

  /**
   * Subscribe to proctoring pause events (all AI polling must stop).
   * Fired when the app transitions to the OFFLINE state.
   *
   * @param {() => void} callback
   * @returns {() => void} unsubscribe function
   */
  onProctoringPause(callback) {
    const handler = () => callback();
    ipcRenderer.on('proctoring:pause', handler);
    return () => ipcRenderer.removeListener('proctoring:pause', handler);
  },

  /**
   * Subscribe to proctoring resume events (restart all AI polling).
   * Fired when the app transitions from OFFLINE back to ONLINE.
   *
   * @param {() => void} callback
   * @returns {() => void} unsubscribe function
   */
  onProctoringResume(callback) {
    const handler = () => callback();
    ipcRenderer.on('proctoring:resume', handler);
    return () => ipcRenderer.removeListener('proctoring:resume', handler);
  },

  /**
   * Send current exam state snapshot data to the main process for persistence.
   * Called on every answer change and in response to onSnapshotRequest.
   *
   * @param {{ currentQuestionIndex: number, answers: object, frozenTimerSeconds: number }} data
   */
  sendSnapshotData(data) {
    ipcRenderer.send('offline:snapshot-data', data);
  },
});
