'use strict';

const { app, BrowserWindow, ipcMain, net, shell, globalShortcut } = require('electron');
const path = require('path');
const os = require('os');
const fs = require('fs');
const { spawn } = require('child_process');
const keytar = require('keytar');

// ---------------------------------------------------------------------------
// Module-level state (exported for internal use by later task expansions)
// ---------------------------------------------------------------------------
// changes
/** @type {BrowserWindow | null} */
let mainWindow = null;

/** @type {import('child_process').ChildProcess | null} */
let pythonProcess = null;

/** @type {'idle' | 'starting' | 'ready' | 'failed' | 'crashed' | 'stopping'} */
let bridgeState = 'idle';

/** @type {import('child_process').ChildProcess | null} */
let aiRouterProcess = null;

let aiRpcNextId = 1;
const aiRpcPending = new Map();

/**
 * Resolve the absolute path to config.json based on packaging state.
 * @returns {string}
 */
function resolveConfigPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'config.json');
  }
  return path.join(__dirname, '..', 'config.json');
}

/**
 * Spawn the AI Router process (router.py).
 * Captures JSON-RPC notifications from stdout and forwards them to the renderer.
 */
function startAIRouter() {
  const routerScript = path.join(__dirname, '..', 'python_bridge', 'router.py');
  const configPath = resolveConfigPath();
  const projectRoot = path.join(__dirname, '..');

  // Prefer the venv Python so all AI packages are available.
  // Fall back to the system 'python' / 'python3' if venv is absent.
  const venvPython = process.platform === 'win32'
    ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(projectRoot, '.venv', 'bin', 'python');
  const pythonExe = fs.existsSync(venvPython) ? venvPython : 'python';

  aiRouterProcess = spawn(pythonExe, [routerScript], {
    env: { ...process.env, LUMINA_CONFIG_PATH: configPath },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  let buffer = '';
  aiRouterProcess.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let lines = buffer.split('\n');
    buffer = lines.pop(); // Keep partial line in buffer

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.jsonrpc !== '2.0') continue;

        if (msg.id != null) {
          const pending = aiRpcPending.get(msg.id);
          if (pending) {
            aiRpcPending.delete(msg.id);
            clearTimeout(pending.timer);
            if (msg.error) {
              pending.resolve({ ok: false, error: msg.error });
            } else {
              pending.resolve({ ok: true, result: msg.result });
            }
          }
          continue;
        }

        // Only forward notifications (detection, alert, riskScore) to renderer
        if (msg.method && !msg.id) {
          mainWindow?.webContents.send('bridge:ai-event', msg);
        }
      } catch (err) {
        process.stderr.write(`[router] JSON parse error: ${err.message}\n`);
      }
    }
  });

  aiRouterProcess.stderr.on('data', (chunk) => {
    process.stderr.write(`[router-err] ${chunk}`);
  });

  aiRouterProcess.on('close', (code) => {
    process.stderr.write(`[router] exited with code ${code}\n`);
  });
}

/**
 * Send a JSON-RPC request to the AI router stdin and await the response.
 * @param {string} method
 * @param {object} params
 * @param {number} timeoutMs
 * @returns {Promise<{ok: true, result: object} | {ok: false, error: object}>}
 */
function sendAiRpc(method, params = {}, timeoutMs = 10000) {
  if (!aiRouterProcess || !aiRouterProcess.stdin || aiRouterProcess.killed) {
    return Promise.resolve({
      ok: false,
      error: { code: 'ROUTER_NOT_READY', message: 'AI router process is not running.' },
    });
  }

  if (!method || typeof method !== 'string') {
    return Promise.resolve({
      ok: false,
      error: { code: 'INVALID_REQUEST', message: 'AI router method must be a string.' },
    });
  }

  const id = aiRpcNextId++;
  const payload = { jsonrpc: '2.0', id, method, params };

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      aiRpcPending.delete(id);
      resolve({
        ok: false,
        error: { code: 'ROUTER_TIMEOUT', message: `AI router timed out for ${method}.` },
      });
    }, timeoutMs);

    aiRpcPending.set(id, { resolve, timer });

    try {
      aiRouterProcess.stdin.write(`${JSON.stringify(payload)}\n`);
    } catch (err) {
      aiRpcPending.delete(id);
      clearTimeout(timer);
      resolve({
        ok: false,
        error: { code: 'ROUTER_WRITE_FAILED', message: err.message },
      });
    }
  });
}

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lockdown helpers (spec 013)
// ---------------------------------------------------------------------------

/**
 * Append a LockdownAlertRecord to the session's JSONL log file.
 * Best-effort — never throws. Consistent with clip_upload_service.py log pattern.
 *
 * @param {'VIRTUAL_ENVIRONMENT' | 'SCREEN_CAPTURE_DETECTED' | 'FULLSCREEN_ESCAPE_ATTEMPT'} type
 * @param {string | null} reason
 */
function appendLockdownAlert(type, reason) {
  try {
    if (!examSession?.attemptId) return;
    const record = JSON.stringify({
      type,
      timestamp: new Date().toISOString(),
      sessionId: String(examSession.attemptId),
      reason: reason ?? null,
    });
    const logPath = path.join(__dirname, '..', 'sessions', `${examSession.attemptId}.jsonl`);
    fs.appendFileSync(logPath, record + '\n');
  } catch (_err) {
    // Best-effort — never propagate
  }
}

/**
 * Keyboard shortcuts that must be silently consumed during an active exam.
 * Registered via globalShortcut (system-wide) AND guarded in before-input-event.
 * OS-reserved shortcuts (Alt+Tab, Win+D, Win+L, Ctrl+Shift+Esc) are not listed
 * here because globalShortcut.register() silently fails for them on Windows.
 */
const BLOCKED_SHORTCUTS = [
  'PrintScreen',
  'Alt+PrintScreen',
  'F11',
  'Escape',
  'Ctrl+Shift+I',
  'Ctrl+W',
  'Ctrl+A',
  'Ctrl+C',
  'Ctrl+V',
  'Ctrl+X',
];

/**
 * Run a single environment check against the Python bridge.
 * On success pushes lockdown IPC events to the renderer for any detected violations.
 * Fails silently on any error (FR-017).
 */
async function runEnvCheck() {
  // Guard: only run while an exam session is active
  if (!examSession) return;
  try {
    const response = await net.fetch(`http://127.0.0.1:${bridgePort}/check-environment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    if (!response.ok) return;
    const result = await response.json();

    if (!examSession) return; // session may have ended while awaiting

    if (result.vm_detected || result.rdp_detected) {
      const reason = result.vm_reason || result.rdp_reason;
      appendLockdownAlert('VIRTUAL_ENVIRONMENT', reason);
      mainWindow?.webContents.send('lockdown:vm-detected', { reason });
    }

    if (result.screen_capture_detected) {
      appendLockdownAlert('SCREEN_CAPTURE_DETECTED', result.screen_capture_reason);
      mainWindow?.webContents.send('lockdown:screen-capture-detected', {
        reason: result.screen_capture_reason,
      });
    }
  } catch (_err) {
    // Silent failure per FR-017 — bridge unavailable is treated as non-violation
  }
}

/**
 * Activate all lockdown controls when an exam session becomes active.
 * Idempotent — guarded by lockdownActive flag.
 */
function activateLockdown() {
  if (lockdownActive) return;
  lockdownActive = true;

  // Window constraints
  if (!lockdownDisabled.fullscreen) {
    mainWindow?.setFullScreen(true);
    mainWindow?.setKiosk(true);
    mainWindow?.setAlwaysOnTop(true);
    mainWindow?.setResizable(false);
    mainWindow?.setMovable(false);
    mainWindow?.setMinimizable(false);
  }

  // Content protection (renders window as blank in OS screenshots)
  if (!lockdownDisabled.contentProtection) {
    mainWindow?.setContentProtection(true);
  }

  // System-wide shortcut blocking (globalShortcut layer)
  if (!lockdownDisabled.shortcutBlocking) {
    for (const shortcut of BLOCKED_SHORTCUTS) {
      try {
        globalShortcut.register(shortcut, () => { /* silently consume */ });
      } catch (_err) {
        // Registration failure is silently ignored per spec assumption 5
      }
    }
  }

  // Environment check: immediate + periodic
  if (!lockdownDisabled.envChecks) {
    runEnvCheck();
    envCheckInterval = setInterval(runEnvCheck, 60_000);
  }
}

/**
 * Deactivate all lockdown controls when the exam session ends.
 * Idempotent — guarded by lockdownActive flag.
 * Must be called on all three exam exit paths.
 */
function deactivateLockdown() {
  if (!lockdownActive) return;
  lockdownActive = false;

  // Cancel periodic environment check
  clearInterval(envCheckInterval);
  envCheckInterval = null;

  // Unregister all system-wide shortcuts
  globalShortcut.unregisterAll();

  // Restore window to normal state
  mainWindow?.setContentProtection(false);
  if (!lockdownDisabled.fullscreen) {
    mainWindow?.setKiosk(false);
    mainWindow?.setFullScreen(false);
    mainWindow?.setAlwaysOnTop(false);
    mainWindow?.setResizable(true);
    mainWindow?.setMovable(true);
    mainWindow?.setMinimizable(true);
  }
}

ipcMain.handle('bridge:get-status', () => bridgeState);

/**
 * bridge:get-ui-config — Return the ui and clip_recording sections of config.json to the renderer.
 * Used by exam.js to read polling intervals and clip recording parameters without
 * hardcoding them in JS.
 * Returns: { ok: true, ui: object } | { ok: false }
 */
ipcMain.handle('bridge:get-ui-config', () => {
  try {
    const configPath = resolveConfigPath();
    const data = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    return { ok: true, ui: { ...(data.ui ?? {}), clip_recording: data.clip_recording ?? {} } };
  } catch (err) {
    process.stderr.write(`[config] bridge:get-ui-config error: ${err.message}\n`);
    return { ok: false };
  }
});

/**
 * bridge:ai-rpc — Forward a JSON-RPC request to the AI router stdin.
 * Expected args: { method: string, params?: object, timeoutMs?: number }
 */
ipcMain.handle('bridge:ai-rpc', async (_event, { method, params, timeoutMs } = {}) => {
  return sendAiRpc(method, params || {}, Number.isFinite(timeoutMs) ? timeoutMs : 10000);
});

/**
 * bridge:get-session-log — Read a session's JSONL log file from disk.
 * Arg: { sessionId: string }
 */
ipcMain.handle('bridge:get-session-log', async (_event, { sessionId }) => {
  try {
    const logPath = path.join(__dirname, '..', 'sessions', `${sessionId}.jsonl`);
    if (!fs.existsSync(logPath)) {
      return { ok: false, error: { code: 'LOG_NOT_FOUND', message: 'Session log not found.' } };
    }
    const content = fs.readFileSync(logPath, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim());
    return { ok: true, data: lines };
  } catch (err) {
    return { ok: false, error: { code: 'IO_ERROR', message: err.message } };
  }
});

/**
 * bridge:export-pdf — Export current webContents to a PDF file.
 */
ipcMain.handle('bridge:export-pdf', async (_event, { filename }) => {
  try {
    const { filePath } = await shell.showSaveDialog(mainWindow, {
      defaultPath: filename,
      filters: [{ name: 'PDF Files', extensions: ['pdf'] }]
    });

    if (!filePath) return { ok: false };

    const data = await mainWindow.webContents.printToPDF({
      printBackground: true,
      margins: { top: 0, bottom: 0, left: 0, right: 0 }
    });

    fs.writeFileSync(filePath, data);
    return { ok: true, path: filePath };
  } catch (err) {
    process.stderr.write(`[pdf] export failed: ${err.message}\n`);
    return { ok: false };
  }
});

// ---------------------------------------------------------------------------
// Session IPC handlers (T008 bridge:login, T014 get-saved-session + clear-session)
// ---------------------------------------------------------------------------

/**
 * Port the Python bridge is listening on.
 * Set during startup after config is read; used by IPC handlers.
 * @type {number}
 */
let bridgePort = 5050;

/**
 * In-memory session for users who did NOT check "Remember this device".
 * Cleared when the app quits. Never written to the OS keychain.
 * @type {object | null}
 */
let sessionMemory = null;

/**
 * Exam session data returned by the LMS on a successful exam start.
 * Stored here so the Exam page can retrieve it via bridge:get-exam-session.
 * In-memory only — not persisted across app restarts.
 * @type {object | null}
 */
let examSession = null;

/**
 * Cheating report ID created at exam start via POST /api/CheatingReport/attempt/{attemptId}.
 * Stored so every subsequent clip upload can POST to /api/CheatingReport/{reportId}/violations.
 * Cleared on all exam exit paths alongside examSession.
 * @type {number | null}
 */
let cheatingReportId = null;

/**
 * Handle for the 60-second periodic environment check.
 * Set on exam start via activateLockdown(), cleared on all three exit paths
 * via deactivateLockdown().
 * @type {ReturnType<typeof setInterval> | null}
 */
let envCheckInterval = null;

/**
 * True while globalShortcut registrations are active.
 * Guards against double-registration if bridge:start-exam is called twice.
 * @type {boolean}
 */
let lockdownActive = false;

/**
 * Per-feature lockdown kill-switches read from config.json at startup.
 * Each defaults to false (controls active). Set to true in config.json to
 * disable the corresponding control. Development / testing only.
 */
let lockdownDisabled = {   // kept as object so callers stay readable
  fullscreen: false,
  contentProtection: false,
  shortcutBlocking: false,
  envChecks: false,
  closePrevention: false,
};

/**
 * Submit result returned by the LMS after the exam is submitted.
 * Stored here so the Result page can retrieve it via bridge:get-submit-result.
 * In-memory only — not persisted across app restarts.
 * @type {object | null}
 */
let submitResult = null;

// ---------------------------------------------------------------------------
// OfflineManager (spec 014 — offline resilience)
// ---------------------------------------------------------------------------

/**
 * OfflineManager — owns the ping loop, state machine, local snapshot
 * persistence, proctoring pause/resume signalling, auto-submit on reconnect,
 * and session flagging.
 *
 * State machine: ONLINE → OFFLINE → LOCKED (terminal until auto-submit)
 * Crash recovery: reads sessions/{attemptId}_offline_snapshot.json on startup.
 */
class OfflineManager {
  constructor() {
    /** @type {'ONLINE'|'OFFLINE'|'LOCKED'} */
    this.state = 'ONLINE';

    // Config values — populated in init()
    this.maxOfflineMs = 10 * 60 * 1000; // 10 min default
    this.maxDisconnections = 3;
    this.pingIntervalMs = 5_000;
    this.flickerThresholdMs = 2_000;

    // Runtime stats
    this.cumulativeOfflineMs = 0;
    this.disconnectionCount = 0;
    this.lastDisconnectAt = null;
    this.lastReconnectAt = null;

    // Interval handles
    this._pingInterval = null;
    this._snapshotInterval = null;
    this._offlineTickInterval = null;

    // Flicker guard
    this._disconnectStartTime = null;

    // Auto-submit guard (prevent double-submit)
    this._autoSubmitDone = false;

    // Last known snapshot data from renderer
    this._lastSnapshotData = null;
  }

  /**
   * Initialise from config and start the ping loop.
   * Call once the exam session is established.
   * @param {object} cfg  The parsed config.json object
   */
  init(cfg) {
    const or = cfg.offline_resilience ?? {};
    this.maxOfflineMs = (or.max_offline_minutes ?? 10) * 60 * 1000;
    this.maxDisconnections = or.max_disconnections ?? 3;
    this.pingIntervalMs = (or.ping_interval_seconds ?? 5) * 1000;
    this.flickerThresholdMs = (or.flicker_threshold_seconds ?? 2) * 1000;

    // Reset runtime stats for fresh session
    this.cumulativeOfflineMs = 0;
    this.disconnectionCount = 0;
    this.lastDisconnectAt = null;
    this.lastReconnectAt = null;
    this._autoSubmitDone = false;
    this._lastSnapshotData = null;
    this._disconnectStartTime = null;
    this.state = 'ONLINE';

    this._startPingLoop();
    this._startSnapshotHeartbeat();
  }

  /**
   * Restore stats from a previously saved snapshot (crash recovery).
   * Called before init() if a snapshot file is found on startup.
   * @param {object} snapshot
   */
  restoreStats(snapshot) {
    const s = snapshot.offlineStats ?? {};
    this.cumulativeOfflineMs = s.cumulativeOfflineMs ?? 0;
    this.disconnectionCount = s.disconnectionCount ?? 0;
    this.lastDisconnectAt = s.lastDisconnectAt ? new Date(s.lastDisconnectAt) : null;
    this.lastReconnectAt = s.lastReconnectAt ? new Date(s.lastReconnectAt) : null;
    this._lastSnapshotData = {
      currentQuestionIndex: snapshot.currentQuestionIndex ?? 0,
      answers: snapshot.answers ?? {},
      frozenTimerSeconds: snapshot.frozenTimerSeconds ?? 0,
    };
  }

  /** Tear down all intervals (called on exam end / logout). */
  stop() {
    clearInterval(this._pingInterval);
    clearInterval(this._snapshotInterval);
    clearInterval(this._offlineTickInterval);
    this._pingInterval = null;
    this._snapshotInterval = null;
    this._offlineTickInterval = null;
  }

  // ---- private helpers ----

  _startPingLoop() {
    clearInterval(this._pingInterval);
    this._pingInterval = setInterval(() => this._doPing(), this.pingIntervalMs);
  }

  _startSnapshotHeartbeat() {
    clearInterval(this._snapshotInterval);
    this._snapshotInterval = setInterval(() => {
      // Ask renderer for current state
      mainWindow?.webContents.send('offline:snapshot-request');
    }, 30_000);
  }

  async _doPing() {
    if (!examSession) return; // no active session

    let reachable = false;
    try {
      const cfg = readConfig();
      const response = await net.fetch(cfg.baseUrl, {
        method: 'HEAD',
        signal: AbortSignal.timeout(this.pingIntervalMs - 500),
      });
      reachable = response.status < 500;
    } catch (_err) {
      reachable = false;
    }

    if (reachable) {
      this._onPingSuccess();
    } else {
      this._onPingFailure();
    }
  }

  _onPingSuccess() {
    if (this.state === 'LOCKED') {
      // Attempt auto-submit — do not transition to ONLINE
      this._attemptAutoSubmit();
      return;
    }

    if (this.state === 'OFFLINE') {
      // Accumulate elapsed offline time
      if (this._disconnectStartTime) {
        this.cumulativeOfflineMs += Date.now() - this._disconnectStartTime;
        this._disconnectStartTime = null;
      }
      clearInterval(this._offlineTickInterval);
      this._offlineTickInterval = null;
      this.lastReconnectAt = new Date();
      this._transitionTo('ONLINE', null);
    }
    // else already ONLINE — nothing to do
  }

  _onPingFailure() {
    if (this.state === 'LOCKED') return; // terminal state

    if (this.state === 'ONLINE') {
      // Start timing the disconnection
      if (!this._disconnectStartTime) {
        this._disconnectStartTime = Date.now();
      }
      const elapsed = Date.now() - this._disconnectStartTime;
      if (elapsed >= this.flickerThresholdMs) {
        // Confirmed offline — not a flicker
        this.disconnectionCount += 1;
        this.lastDisconnectAt = new Date();

        // Check disconnection limit BEFORE transitioning
        if (this.disconnectionCount > this.maxDisconnections) {
          this._transitionTo('LOCKED', 'disconnection_limit');
          return;
        }
        this._transitionTo('OFFLINE', null);
        this._startOfflineTick();
      }
      return;
    }

    // Already OFFLINE — keep accumulating
  }

  _startOfflineTick() {
    clearInterval(this._offlineTickInterval);
    this._offlineTickInterval = setInterval(() => {
      if (!this._disconnectStartTime) return;
      const elapsed = Date.now() - this._disconnectStartTime;
      const total = this.cumulativeOfflineMs + elapsed;
      if (total >= this.maxOfflineMs) {
        this.cumulativeOfflineMs = total;
        this._disconnectStartTime = null;
        clearInterval(this._offlineTickInterval);
        this._offlineTickInterval = null;
        this._transitionTo('LOCKED', 'budget_exhausted');
      }
    }, 1_000);
  }

  _buildStatePayload(lockReason) {
    const elapsed = this._disconnectStartTime ? Date.now() - this._disconnectStartTime : 0;
    const total = this.cumulativeOfflineMs + elapsed;
    const budgetRemainingMs = Math.max(0, this.maxOfflineMs - total);
    const answeredCount = this._lastSnapshotData
      ? Object.keys(this._lastSnapshotData.answers ?? {}).length
      : 0;
    return {
      state: this.state,
      budgetRemainingMs,
      budgetTotalMs: this.maxOfflineMs,
      disconnectionCount: this.disconnectionCount,
      maxDisconnections: this.maxDisconnections,
      answeredCount,
      lockReason: lockReason ?? null,
    };
  }

  _transitionTo(newState, lockReason) {
    const prev = this.state;
    this.state = newState;

    // Log the transition to the session JSONL file
    try {
      if (examSession?.attemptId) {
        const logPath = path.join(__dirname, '..', 'sessions', `${examSession.attemptId}.jsonl`);
        const record = JSON.stringify({
          type: newState === 'OFFLINE' ? 'NETWORK_DISCONNECTED' : (newState === 'ONLINE' ? 'NETWORK_RECONNECTED' : 'NETWORK_LOCKED'),
          timestamp: new Date().toISOString(),
          sessionId: String(examSession.attemptId),
          cumulativeOfflineMs: this.cumulativeOfflineMs,
          disconnectionCount: this.disconnectionCount,
          lockReason: lockReason ?? null,
        });
        fs.appendFileSync(logPath, record + '\n');
      }
    } catch (_) { /* best effort */ }

    if (prev === 'ONLINE' && newState === 'OFFLINE') {
      // Exit fullscreen, pause proctoring
      if (!lockdownDisabled.fullscreen) {
        mainWindow?.setKiosk(false);
        mainWindow?.setFullScreen(false);
        mainWindow?.setAlwaysOnTop(false);
        mainWindow?.setMinimizable(true);
      }
      mainWindow?.webContents.send('proctoring:pause');
      // Navigate renderer to offline page
      mainWindow?.webContents.send('offline:state-changed', this._buildStatePayload(null));
    } else if (prev === 'OFFLINE' && newState === 'ONLINE') {
      // Resume fullscreen and proctoring
      if (!lockdownDisabled.fullscreen) {
        mainWindow?.setFullScreen(true);
        mainWindow?.setKiosk(true);
        mainWindow?.setAlwaysOnTop(true);
        mainWindow?.setMinimizable(false);
      }
      mainWindow?.webContents.send('proctoring:resume');
      mainWindow?.webContents.send('offline:state-changed', this._buildStatePayload(null));
    } else if (newState === 'LOCKED') {
      // Save locked snapshot immediately
      if (this._lastSnapshotData) {
        this.saveSnapshot(this._lastSnapshotData, 'locked');
      }
      mainWindow?.webContents.send('offline:state-changed', this._buildStatePayload(lockReason));
    }
  }

  /**
   * Save the local exam snapshot synchronously.
   * Best-effort — never throws. Called from IPC handler and heartbeat.
   * @param {{ currentQuestionIndex: number, answers: object, frozenTimerSeconds: number }} data
   * @param {'active'|'locked'} [lockStatus]
   */
  saveSnapshot(data, lockStatus) {
    if (!examSession?.attemptId) return;
    this._lastSnapshotData = data;
    const status = lockStatus ?? (this.state === 'LOCKED' ? 'locked' : 'active');
    const snapshot = {
      attemptId: String(examSession.attemptId),
      studentId: examSession.studentId ?? examSession.userId ?? null,
      currentQuestionIndex: data.currentQuestionIndex ?? 0,
      answers: data.answers ?? {},
      frozenTimerSeconds: data.frozenTimerSeconds ?? 0,
      lockStatus: status,
      savedAt: new Date().toISOString(),
      offlineStats: {
        cumulativeOfflineMs: this.cumulativeOfflineMs,
        disconnectionCount: this.disconnectionCount,
        lastDisconnectAt: this.lastDisconnectAt?.toISOString() ?? null,
        lastReconnectAt: this.lastReconnectAt?.toISOString() ?? null,
      },
    };
    try {
      const snapshotPath = path.join(
        __dirname, '..', 'sessions', `${examSession.attemptId}_offline_snapshot.json`
      );
      fs.writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2));
    } catch (writeErr) {
      // Best-effort — log to JSONL but never propagate
      try {
        const record = JSON.stringify({
          type: 'SNAPSHOT_WRITE_FAILED',
          timestamp: new Date().toISOString(),
          sessionId: String(examSession.attemptId),
          reason: writeErr.message,
        });
        const logPath = path.join(
          __dirname, '..', 'sessions', `${examSession.attemptId}.jsonl`
        );
        fs.appendFileSync(logPath, record + '\n');
      } catch (_) { /* nothing */ }
    }
  }

  /**
   * Delete the snapshot file after successful auto-submit.
   * Best-effort — never throws.
   */
  deleteSnapshot() {
    if (!examSession?.attemptId) return;
    try {
      const snapshotPath = path.join(
        __dirname, '..', 'sessions', `${examSession.attemptId}_offline_snapshot.json`
      );
      if (fs.existsSync(snapshotPath)) fs.unlinkSync(snapshotPath);
    } catch (_) { /* best-effort */ }
  }

  /**
   * Attempt to restore a prior session from a snapshot file.
   * Called at app startup before loading any page.
   * Returns the snapshot object if valid and active, null otherwise.
   * @returns {{ snapshot: object, locked: boolean } | null}
   */
  tryRestoreSession() {
    // Determine attemptId from examSession (may not be set yet at startup)
    // We scan for any existing snapshot file in sessions/
    try {
      const sessionsDir = path.join(__dirname, '..', 'sessions');
      if (!fs.existsSync(sessionsDir)) return null;
      const files = fs.readdirSync(sessionsDir)
        .filter(f => f.endsWith('_offline_snapshot.json'));
      if (files.length === 0) return null;

      // Use the most recently modified snapshot
      const latest = files
        .map(f => ({ f, mtime: fs.statSync(path.join(sessionsDir, f)).mtimeMs }))
        .sort((a, b) => b.mtime - a.mtime)[0].f;

      const raw = fs.readFileSync(path.join(sessionsDir, latest), 'utf-8');
      const snapshot = JSON.parse(raw);

      if (!snapshot.attemptId) return null;
      return { snapshot, locked: snapshot.lockStatus === 'locked' };
    } catch (_err) {
      process.stderr.write(`[offline] tryRestoreSession: could not read snapshot: ${_err.message}\n`);
      return null;
    }
  }

  /**
   * Auto-submit all answered questions after the exam is locked and
   * the student reconnects. Uses the existing bridge:submit-exam path.
   * Best-effort — logs result to JSONL.
   */
  async _attemptAutoSubmit() {
    if (this._autoSubmitDone) return;
    this._autoSubmitDone = true;

    try {
      if (!this._lastSnapshotData || !examSession) return;

      // Build answers array in the format expected by bridge:submit-exam
      const answersObj = this._lastSnapshotData.answers ?? {};
      const answers = Object.entries(answersObj).map(([questionId, choiceId]) => ({
        questionId: Number(questionId),
        choiceId: typeof choiceId === 'number' ? choiceId : Number(choiceId),
      }));

      // Reuse the bridge:submit-exam IPC handler logic via a direct call
      // to the Flask bridge using the stored session token
      const cfg = readConfig();
      const session = await getSavedSession();
      const token = session?.accessToken ?? null;
      if (!token) {
        process.stderr.write('[offline] auto-submit: no auth token available\n');
        return;
      }

      const response = await net.fetch(
        `${cfg.baseUrl}/api/Exam/SubmitExam/${examSession.attemptId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ answers }),
        }
      );

      const logRecord = JSON.stringify({
        type: response.ok ? 'OFFLINE_AUTO_SUBMIT_SUCCESS' : 'OFFLINE_AUTO_SUBMIT_FAILED',
        timestamp: new Date().toISOString(),
        sessionId: String(examSession.attemptId),
        httpStatus: response.status,
        answeredCount: answers.length,
      });
      const logPath = path.join(
        __dirname, '..', 'sessions', `${examSession.attemptId}.jsonl`
      );
      fs.appendFileSync(logPath, logRecord + '\n');

      if (response.ok) {
        this.deleteSnapshot();
        // Notify renderer the submit is done
        mainWindow?.webContents.send('offline:state-changed', {
          state: 'SUBMITTED',
          budgetRemainingMs: 0,
          budgetTotalMs: this.maxOfflineMs,
          disconnectionCount: this.disconnectionCount,
          maxDisconnections: this.maxDisconnections,
          answeredCount: answers.length,
          lockReason: 'auto_submitted',
        });
      }
    } catch (err) {
      process.stderr.write(`[offline] auto-submit error: ${err.message}\n`);
      this._autoSubmitDone = false; // allow retry on next ping
    }
  }

  /**
   * Append a OFFLINE_SESSION_FLAGGED record to the session JSONL log.
   * Called when the app closes with a locked, unsubmitted session.
   */
  flagSessionForReview() {
    if (!examSession?.attemptId || this._autoSubmitDone) return;
    try {
      const answeredCount = this._lastSnapshotData
        ? Object.keys(this._lastSnapshotData.answers ?? {}).length
        : 0;
      const record = JSON.stringify({
        type: 'OFFLINE_SESSION_FLAGGED',
        timestamp: new Date().toISOString(),
        sessionId: String(examSession.attemptId),
        reason: 'never_reconnected',
        cumulativeOfflineMs: this.cumulativeOfflineMs,
        disconnectionCount: this.disconnectionCount,
        answeredCount,
      });
      const logPath = path.join(
        __dirname, '..', 'sessions', `${examSession.attemptId}.jsonl`
      );
      fs.appendFileSync(logPath, record + '\n');
    } catch (_err) {
      // Best-effort — never propagate on quit
    }
  }
}

/** Singleton instance — created once; reset on exam end. */
const offlineManager = new OfflineManager();

/**
 * Enrollment state for the current exam attempt.
 * Set after a successful bridge:enroll-reference call.
 * Cleared on all exam exit paths (submit, back-to-home, clear-session).
 * @type {{ sessionId: string, enrolledAt: Date, succeeded: boolean } | null}
 */
let enrollmentState = null;

/** keytar service identifier shared across all session keys. */
const KEYTAR_SERVICE = 'lumina-ai-proctoring';

// ---------------------------------------------------------------------------
// Config resolution (research.md decision 4)
// Packaged:     <resources>/config.json   (electron-builder extraResources)
// Development:  <project root>/config.json
// ---------------------------------------------------------------------------

/**
 * Resolve the absolute path to config.json based on packaging state.
 * @returns {string}
 */
function resolveConfigPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'config.json');
  }
  // In development, main.js lives at frontend/main.js — config.json is one
  // level up at the project root.
  return path.join(__dirname, '..', 'config.json');
}

/**
 * Read and parse config.json.
 * Returns the parsed object, or throws with a typed error shape that mirrors
 * the Python ConfigError so Electron can surface it uniformly.
 *
 * @returns {{ baseUrl: string, pythonPort: number }}
 */
function readConfig() {
  const configPath = resolveConfigPath();

  if (!fs.existsSync(configPath)) {
    const err = new Error(`config.json not found at path: ${configPath}`);
    err.code = 'FILE_NOT_FOUND';
    throw err;
  }

  let raw;
  try {
    raw = fs.readFileSync(configPath, 'utf-8');
  } catch (ioErr) {
    const err = new Error(`Could not read config.json: ${ioErr.message}`);
    err.code = 'FILE_NOT_FOUND';
    throw err;
  }

  let data;
  try {
    data = JSON.parse(raw);
  } catch (parseErr) {
    const err = new Error(`config.json contains invalid JSON: ${parseErr.message}`);
    err.code = 'INVALID_JSON';
    throw err;
  }

  const { baseUrl, pythonPort = 5050 } = data;

  if (!baseUrl || typeof baseUrl !== 'string' || !baseUrl.trim()) {
    const err = new Error("config.json must contain a non-empty 'baseUrl' string.");
    err.code = 'MISSING_BASE_URL';
    throw err;
  }

  if (!baseUrl.trim().startsWith('https://')) {
    const err = new Error('baseUrl must use HTTPS (https://). HTTP URLs are not permitted.');
    err.code = 'INSECURE_PROTOCOL';
    throw err;
  }

  if (typeof pythonPort !== 'number' || !Number.isInteger(pythonPort) || pythonPort < 1024 || pythonPort > 65535) {
    const err = new Error(`pythonPort must be an integer between 1024 and 65535, got ${pythonPort}.`);
    err.code = 'INVALID_PORT';
    throw err;
  }

  const ld = data.lockdown ?? {};
  const lockdownDisabled = {
    fullscreen: ld.disable_fullscreen === true,
    contentProtection: ld.disable_content_protection === true,
    shortcutBlocking: ld.disable_shortcut_blocking === true,
    envChecks: ld.disable_env_checks === true,
    closePrevention: ld.disable_close_prevention === true,
  };
  return { baseUrl: baseUrl.trim().replace(/\/$/, ''), pythonPort, lockdownDisabled };
}

// ---------------------------------------------------------------------------
// Session helpers (keytar + in-memory)
// ---------------------------------------------------------------------------

/**
 * Delete all 6 keytar entries for this application.
 * Idempotent — safe to call even if no entries exist.
 * Errors are swallowed and logged to stderr (never thrown).
 */
async function clearAllKeytarEntries() {
  const keys = [
    'access-token',
    'refresh-token',
    'token-expiry',
    'refresh-expiry',
    'user-profile',
    'remember-flag',
  ];
  for (const key of keys) {
    try {
      await keytar.deletePassword(KEYTAR_SERVICE, key);
    } catch (err) {
      process.stderr.write(`[session] clearAllKeytarEntries: key=${key} error=${err.message}\n`);
    }
  }
}

/**
 * Persist a successful login session.
 *
 * If remember=true, write all 6 entries to the OS keychain.
 * If remember=false, store the session object in module-level memory only
 * (cleared automatically when the app quits).
 *
 * Security: the raw password is NOT passed to this function and never stored.
 *
 * @param {object} data    LoginResponse object from the LMS (relayed by bridge).
 * @param {boolean} remember  Whether the user checked "Remember this device".
 */
async function storeSession(data, remember) {
  const tokenExpiry = new Date(Date.now() + data.expinresIn * 1000).toISOString();
  const userProfile = JSON.stringify({
    id: data.id,
    email: data.email,
    firstName: data.firstName,
    lastName: data.lastName,
    profilePictureUrl: data.profilePictureUrl ?? null,
  });

  const session = {
    accessToken: data.token,
    refreshToken: data.refreshToken,
    tokenExpiry,
    refreshExpiry: data.refreshTokenExpiration,
    userProfile: JSON.parse(userProfile),
  };

  if (!remember) {
    sessionMemory = session;
    return;
  }

  // Persist to OS keychain
  const entries = {
    'access-token': data.token,
    'refresh-token': data.refreshToken,
    'token-expiry': tokenExpiry,
    'refresh-expiry': data.refreshTokenExpiration,
    'user-profile': userProfile,
    'remember-flag': '1',
  };

  for (const [key, value] of Object.entries(entries)) {
    try {
      await keytar.setPassword(KEYTAR_SERVICE, key, value);
    } catch (err) {
      process.stderr.write(`[session] storeSession: key=${key} error=${err.message}\n`);
    }
  }
}

/**
 * Attempt to restore a previously saved session from the OS keychain.
 *
 * Checks (in order):
 *   1. remember-flag === '1'
 *   2. refresh-expiry is a future date
 *   3. All remaining 4 keys are present and user-profile is valid JSON
 *
 * On any failure: clears all keytar entries and returns null.
 *
 * @returns {Promise<object|null>} Session object or null if no valid session.
 */
async function getSavedSession() {
  try {
    const flag = await keytar.getPassword(KEYTAR_SERVICE, 'remember-flag');
    if (flag !== '1') return null;

    const refreshExpiry = await keytar.getPassword(KEYTAR_SERVICE, 'refresh-expiry');
    if (!refreshExpiry || new Date(refreshExpiry) <= new Date()) {
      await clearAllKeytarEntries();
      return null;
    }

    const accessToken = await keytar.getPassword(KEYTAR_SERVICE, 'access-token');
    const refreshToken = await keytar.getPassword(KEYTAR_SERVICE, 'refresh-token');
    const tokenExpiry = await keytar.getPassword(KEYTAR_SERVICE, 'token-expiry');
    const profileRaw = await keytar.getPassword(KEYTAR_SERVICE, 'user-profile');

    if (!accessToken || !refreshToken || !tokenExpiry || !profileRaw) {
      await clearAllKeytarEntries();
      return null;
    }

    let userProfile;
    try {
      userProfile = JSON.parse(profileRaw);
    } catch (_parseErr) {
      await clearAllKeytarEntries();
      return null;
    }

    return { accessToken, refreshToken, tokenExpiry, refreshExpiry, userProfile };
  } catch (err) {
    process.stderr.write(`[session] getSavedSession error: ${err.message}\n`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Single-instance lock (research.md decision 5)
// Prevents two Electron windows + two Python bridges competing for port 5050.
// MUST be called before app.whenReady().
// ---------------------------------------------------------------------------

const gotLock = app.requestSingleInstanceLock();

if (!gotLock) {
  // Another instance is already running — focus its window and quit.
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

// ---------------------------------------------------------------------------
// Exports (used by T011/T012 expansions and tests)
// ---------------------------------------------------------------------------

module.exports = {
  getMainWindow: () => mainWindow,
  getPythonProcess: () => pythonProcess,
  getBridgeState: () => bridgeState,
  resolveConfigPath,
  readConfig,
  // Setters used internally by startup sequence tasks
  _setMainWindow: (w) => { mainWindow = w; },
  _setPythonProcess: (p) => { pythonProcess = p; },
  _setBridgeState: (s) => { bridgeState = s; },
};

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Session IPC handlers (T008 bridge:login, T014 get-saved-session + clear-session)
// ---------------------------------------------------------------------------

/**
 * bridge:login — Proxy login credentials to the Python bridge.
 *
 * Expected args: { email: string, password: string, remember: boolean }
 * Returns: { ok: true, data: LoginResponse } | { ok: false, error: BridgeLoginError }
 *
 * Security: password is never logged.
 */
ipcMain.handle('bridge:login', async (_event, { email, password, remember }) => {
  try {
    const response = await net.fetch(`http://127.0.0.1:${bridgePort}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    const body = await response.json();

    if (response.ok) {
      await storeSession(body, Boolean(remember));
      return { ok: true, data: body };
    }

    return { ok: false, error: body };
  } catch (_err) {
    return {
      ok: false,
      error: {
        code: 'BRIDGE_ERROR',
        message: 'Unable to reach the server. Please check your connection and try again.',
      },
    };
  }
});

/**
 * bridge:get-saved-session — Return a previously stored session or indicate none.
 *
 * Returns: { ok: true, session: StoredSession } | { ok: false }
 */
ipcMain.handle('bridge:get-saved-session', async () => {
  const session = await getSavedSession();
  return session ? { ok: true, session } : { ok: false };
});

/**
 * bridge:clear-session — Delete all keytar entries and in-memory session.
 *
 * Idempotent. Returns: { ok: true }
 */
ipcMain.handle('bridge:clear-session', async () => {
  await clearAllKeytarEntries();
  sessionMemory = null;
  deactivateLockdown(); // spec 013 — release all lockdown controls on logout
  offlineManager.stop(); // spec 014 — stop ping loop on logout
  examSession = null;
  cheatingReportId = null;
  // T029 — Unenroll face recognition embedding on logout (fire-and-forget)
  sendAiRpc('unenrollReference', { sessionId: enrollmentState?.sessionId }).catch(() => { });
  enrollmentState = null;
  return { ok: true };
});
/**
 * offline:snapshot-data — Receive current exam state from the renderer
 * and persist it to disk as a crash-recovery snapshot.
 * Also triggered as a heartbeat response every 30 seconds.
 * Payload: { currentQuestionIndex, answers, frozenTimerSeconds }
 */
ipcMain.on('offline:snapshot-data', (_event, data) => {
  offlineManager.saveSnapshot(data, 'active');
});


/**
 * bridge:enroll-reference — Enroll a captured reference photo for face recognition.
 *
 * Expected args: { frame: string }  — base64 data URL of the captured JPEG
 * Returns: { ok: true } | { ok: false, error: { code, message } }
 *
 * Chains face-detect → enroll via the AI router (FaceRecognitionService.enroll).
 * On success sets enrollmentState so the exam page guard can pass.
 */
ipcMain.handle('bridge:enroll-reference', async (_event, { frame } = {}) => {
  if (!frame || typeof frame !== 'string') {
    return { ok: false, error: { code: 'BRIDGE_ERROR', message: 'No frame provided.' } };
  }

  const sessionId = examSession?.attemptId ? String(examSession.attemptId) : 'default-session';

  // Retrieve the official profile picture URL from the stored session.
  // sessionMemory is preferred (in-memory, set on login); falls back to keytar for
  // "Remember this device" sessions. The renderer never holds this value.
  let profilePictureUrl = null;
  try {
    if (sessionMemory?.userProfile?.profilePictureUrl) {
      profilePictureUrl = sessionMemory.userProfile.profilePictureUrl;
    } else {
      const profileRaw = await keytar.getPassword(KEYTAR_SERVICE, 'user-profile');
      if (profileRaw) {
        const parsed = JSON.parse(profileRaw);
        profilePictureUrl = parsed?.profilePictureUrl ?? null;
      }
    }
  } catch (_err) {
    // profilePictureUrl stays null — enrollment proceeds without identity confirmation
    process.stderr.write(`[enroll] failed to read profilePictureUrl: ${_err.message}\n`);
  }

  const rpcResult = await sendAiRpc('enrollReference', {
    frame,
    sessionId,
    profilePictureUrl,
  }, 45000);

  if (rpcResult.ok && rpcResult.result?.ok === true) {
    enrollmentState = { sessionId, enrolledAt: new Date(), succeeded: true };
    return { ok: true };
  }

  // Propagate typed error from the service layer when available
  const error = rpcResult.result?.error || rpcResult.error || {
    code: 'ENROLLMENT_FAILED',
    message: 'Enrollment did not succeed.',
  };
  return { ok: false, error };
});

/**
 * bridge:get-enrollment-status — Check whether enrollment has succeeded.
 *
 * Returns: { enrolled: boolean }
 * Returns { enrolled: false } (not an error) when no enrollment has occurred.
 */
ipcMain.handle('bridge:get-enrollment-status', () => ({
  enrolled: enrollmentState?.succeeded === true,
}));

/**
 * bridge:open-external — Open a URL in the system default browser.
 *
 * Only https:// URLs are allowed; all others are silently ignored
 * to prevent open-redirect abuse.
 */
ipcMain.handle('bridge:open-external', async (_event, url) => {
  if (typeof url === 'string' && url.startsWith('https://')) {
    await shell.openExternal(url);
  }
});

// ---------------------------------------------------------------------------
// Exam IPC handlers (spec 003)
// ---------------------------------------------------------------------------

/**
 * bridge:start-exam — Validate an exam code and start an attempt.
 *
 * Expected args: { quizCode: string }
 * Returns:
 *   { ok: true, data: ExamSession }         — success
 *   { ok: false, redirect: 'login' }        — expired token; session cleared in main.js
 *   { ok: false, error: BridgeExamError }   — typed error
 *
 * Security: token is read from session store only; never from renderer args;
 *           never logged; never echoed in error responses.
 */
ipcMain.handle('bridge:start-exam', async (_event, { quizCode }) => {
  try {
    const accessToken =
      sessionMemory?.accessToken ||
      (await keytar.getPassword(KEYTAR_SERVICE, 'access-token'));

    const response = await net.fetch(`http://127.0.0.1:${bridgePort}/exam-access`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quizCode, token: accessToken }),
    });

    const body = await response.json();

    if (response.ok) {
      examSession = body;
      // Lockdown is activated later — when the student clicks "Start Exam" on the
      // instructions page (after identity verification and model readiness).
      // See lockdown:start IPC handler below.

      // The Python bridge's /exam-access route creates the CheatingReport
      // container and merges reportId into the session payload.
      cheatingReportId = examSession?.reportId ?? null;
      if (cheatingReportId !== null) {
        process.stderr.write(`[cheating-report] reportId=${cheatingReportId} for attemptId=${examSession?.attemptId}\n`);
      }

      const sessionId = examSession?.attemptId ? String(examSession.attemptId) : 'default-session';
      await Promise.all([
        sendAiRpc('startService', { service: 'eye-gaze', sessionId }),
        sendAiRpc('startService', { service: 'speech-detection', sessionId }),
        // Cloud (Modal) services. If not configured, router responds with an error
        // and the UI will remain "Inactive" (status poll retries continuously).
        sendAiRpc('startService', { service: 'face-recognition', sessionId }),
        sendAiRpc('startService', { service: 'face-detection', sessionId }),
        sendAiRpc('startService', { service: 'object-detection', sessionId }),
      ]);

      // Initialise offline resilience for this session (spec 014)
      try {
        const cfgData = JSON.parse(fs.readFileSync(resolveConfigPath(), 'utf-8'));
        offlineManager.init(cfgData);
        // Write initial snapshot for crash recovery baseline
        offlineManager.saveSnapshot({ currentQuestionIndex: 0, answers: {}, frozenTimerSeconds: 0 }, 'active');
      } catch (_cfgErr) {
        process.stderr.write(`[offline] init failed: ${_cfgErr.message}\n`);
      }

      return { ok: true, data: examSession };
    }

    // Expired/invalid token — clear session and signal the renderer to redirect
    if (body?.code === 'UNAUTHORIZED') {
      await clearAllKeytarEntries();
      sessionMemory = null;
      return { ok: false, redirect: 'login' };
    }

    return { ok: false, error: body };
  } catch (_err) {
    return {
      ok: false,
      error: {
        code: 'BRIDGE_ERROR',
        message: 'Unable to reach the server. Please check your connection and try again.',
      },
    };
  }
});

/**
 * lockdown:start — Activate all lockdown controls (fullscreen, kiosk, shortcuts).
 *
 * Called by the Exam Instructions page when the student clicks "Start Exam",
 * i.e. after identity verification and model readiness are complete.
 * Idempotent — safe to call if lockdown is already active.
 */
ipcMain.handle('lockdown:start', () => {
  activateLockdown();
});

/**
 * bridge:get-exam-session — Return the stored ExamSession to the Exam page.
 *
 * Returns: { ok: true, session: ExamSession } | { ok: false }
 */
ipcMain.handle('bridge:get-exam-session', async () => {
  if (examSession) {
    return { ok: true, session: examSession };
  }
  return { ok: false };
});

/**
 * bridge:submit-exam — Forward exam answers to the Python bridge.
 *
 * Reads the stored JWT token from keytar to attach as Authorization header.
 * Returns { ok: true, data } on success or { ok: false, redirect: 'login' } /
 * { ok: false, error } on failure.
 *
 * Returns: { ok: true, data: object } | { ok: false, redirect: 'login' } | { ok: false, error: object }
 */
ipcMain.handle('bridge:submit-exam', async (_event, { answers }) => {
  try {
    const accessToken =
      sessionMemory?.accessToken ||
      (await keytar.getPassword(KEYTAR_SERVICE, 'access-token'));

    const attemptId = examSession?.attemptId;

    const response = await net.fetch(`http://127.0.0.1:${bridgePort}/submit-exam`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attemptId, answers, token: accessToken }),
    });

    const body = await response.json();

    if (response.ok) {
      submitResult = body;
      deactivateLockdown(); // spec 013 — release all lockdown controls on submit

      // Stop local AI services (eye-gaze, speech-detection) so they do not
      // continue monitoring after the student has submitted the exam.
      // Fire-and-forget — never block the submit response.
      sendAiRpc('stopService', { service: 'eye-gaze' }).catch(() => { });
      sendAiRpc('stopService', { service: 'speech-detection' }).catch(() => { });

      // Capture session metadata BEFORE clearing examSession
      const submitSessionId = examSession?.attemptId ? String(examSession.attemptId) : null;
      const submitTotalQuestions = examSession?.questions?.length ?? 0;
      // Filter out any undefined/null ids so the estimator receives only real LMS IDs.
      // If no valid ids can be found, warn loudly — the estimator will fall back to
      // sequential indices (1, 2, 3 …) which causes incorrect cohort grouping on the backend.
      const rawQuestionIds = (examSession?.questions ?? [])
        .map(q => q.id)
        .filter(id => id != null && id !== '');
      if (rawQuestionIds.length === 0 && (examSession?.questions?.length ?? 0) > 0) {
        process.stderr.write(
          '[report] WARNING: examSession.questions is present but no valid .id fields found — ' +
          'estimator will use sequential indices. Cohort grouping on the LMS will be incorrect.\n'
        );
      }
      const questionIds = rawQuestionIds.join(',');
      examSession = null;   // spec 013 — clear exam session so window can close normally
      cheatingReportId = null;

      // ── Generate per-question violation report (fire-and-forget) ──────────
      // Runs risk_estimator.py immediately after submit so the JSONL log is
      // complete and the report is available before the student sees the result.
      try {
        const sessionId = submitSessionId;
        const totalQuestions = submitTotalQuestions;

        if (sessionId && totalQuestions > 0) {
          const projectRoot = path.join(__dirname, '..');
          const logPath = path.join(projectRoot, 'sessions', `${sessionId}.jsonl`);
          const estimatorPath = path.join(projectRoot, 'python_bridge', 'risk_estimator.py');
          const venvPython = process.platform === 'win32'
            ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
            : path.join(projectRoot, '.venv', 'bin', 'python');
          const pythonExe = fs.existsSync(venvPython) ? venvPython : 'python';

          let studentIdToUse = sessionId;
          try {
            const sessionToUse = sessionMemory || await getSavedSession();
            if (sessionToUse?.userProfile?.id) {
              studentIdToUse = String(sessionToUse.userProfile.id);
            }
          } catch (e) { }

          if (fs.existsSync(logPath)) {
            const args = [
              estimatorPath,
              logPath,
              '--total-questions', String(totalQuestions),
              '--student-id', studentIdToUse,
              '--exam-id', sessionId,
            ];
            if (questionIds) {
              args.push('--question-ids', questionIds);
            }

            const reportProc = spawn(pythonExe, args, { stdio: ['ignore', 'pipe', 'pipe'] });

            reportProc.stderr.on('data', (chunk) => {
              process.stderr.write(`[report] ${chunk}`);
            });
            reportProc.on('close', async (code) => {
              if (code === 0) {
                process.stderr.write(`[report] question report generated for session ${sessionId}\n`);

                // ── POST question report to LMS /api/risk-analysis ────────────
                try {
                  const outDir = path.join(projectRoot, 'sessions');
                  const reportPath = path.join(outDir, `${sessionId}_question_report.json`);

                  if (!fs.existsSync(reportPath)) {
                    process.stderr.write(`[risk-analysis] report file not found, skipping POST: ${reportPath}\n`);
                    return;
                  }

                  const reportPayload = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

                  // Read baseUrl from config.json (already validated at startup)
                  let baseUrl = '';
                  try {
                    const cfg = readConfig();
                    baseUrl = cfg.baseUrl;
                  } catch (cfgErr) {
                    process.stderr.write(`[risk-analysis] could not read baseUrl: ${cfgErr.message}\n`);
                    return;
                  }

                  const endpoint = `${baseUrl}/api/risk-analysis`;
                  const delays = [1000, 3000, 9000]; // exponential back-off for 5xx
                  let lastStatus = null;

                  for (let attempt = 0; attempt <= delays.length; attempt++) {
                    try {
                      const riskRes = await net.fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(reportPayload),
                        // Electron net.fetch does not support a timeout option directly;
                        // 10-second guard via AbortSignal is handled by the OS stack.
                      });

                      lastStatus = riskRes.status;

                      if (riskRes.status === 202) {
                        const ack = await riskRes.json().catch(() => ({}));
                        process.stderr.write(
                          `[risk-analysis] 202 Accepted — risk job enqueued for attemptId=${ack.attemptId ?? sessionId}\n`
                        );
                        return; // success — no further retries needed
                      }

                      if (riskRes.status === 400) {
                        const body = await riskRes.json().catch(() => ({}));
                        process.stderr.write(
                          `[risk-analysis] 400 Bad Request — ${body.message ?? 'check Attempt_Id format'}. Not retrying.\n`
                        );
                        return; // payload error — retrying won't help
                      }

                      if (riskRes.status === 404) {
                        const body = await riskRes.json().catch(() => ({}));
                        process.stderr.write(
                          `[risk-analysis] 404 Not Found — ${body.message ?? 'no CheatingReport for this attempt'}. Not retrying.\n`
                        );
                        return; // missing CheatingReport — retrying won't help
                      }

                      // 5xx or unexpected — retry with back-off
                      process.stderr.write(
                        `[risk-analysis] HTTP ${riskRes.status} on attempt ${attempt + 1}/${delays.length + 1} for session ${sessionId}\n`
                      );
                    } catch (fetchErr) {
                      process.stderr.write(
                        `[risk-analysis] fetch error on attempt ${attempt + 1}: ${fetchErr.message}\n`
                      );
                    }

                    // Wait before next retry (skip delay after last attempt)
                    if (attempt < delays.length) {
                      await new Promise(resolve => setTimeout(resolve, delays[attempt]));
                    }
                  }

                  process.stderr.write(
                    `[risk-analysis] all retries exhausted (last status=${lastStatus}) for session ${sessionId}\n`
                  );
                } catch (postErr) {
                  process.stderr.write(`[risk-analysis] unexpected error: ${postErr.message}\n`);
                }
                // ── End risk-analysis POST ────────────────────────────────────

              } else {
                process.stderr.write(`[report] risk_estimator exited with code ${code} for session ${sessionId}\n`);
              }
            });
          } else {
            process.stderr.write(`[report] session log not found, skipping report: ${logPath}\n`);
          }
        }
      } catch (reportErr) {
        // Never block the submit response due to a report generation error
        process.stderr.write(`[report] report generation failed: ${reportErr.message}\n`);
      }
      // ── End report generation ──────────────────────────────────────────────

      // T027 — Unenroll face recognition embedding on exam submit (fire-and-forget)
      sendAiRpc('unenrollReference', { sessionId: enrollmentState?.sessionId }).catch(() => { });
      enrollmentState = null;
      return { ok: true, data: submitResult };
    }

    if (body?.code === 'UNAUTHORIZED') {
      await clearAllKeytarEntries();
      sessionMemory = null;
      deactivateLockdown(); // spec 013 — release lockdown on forced logout
      examSession = null;
      cheatingReportId = null;
      mainWindow?.loadFile(path.join(__dirname, 'pages/login/index.html'));
      return; // renderer IPC call never resolves — main.js navigates away
    }

    return { ok: false, error: body };
  } catch {
    return {
      ok: false,
      error: {
        code: 'BRIDGE_ERROR',
        message: 'Unable to reach the server. Please check your connection and try again.',
      },
    };
  }
});

/**
 * bridge:get-submit-result — Return the stored SubmitResult to the Result page.
 *
 * Returns: { ok: true, data: object } | { ok: false }
 */
ipcMain.handle('bridge:get-submit-result', async () => {
  if (submitResult) {
    return { ok: true, data: submitResult };
  }
  return { ok: false };
});

/**
 * bridge:get-result — Recover the result from the LMS when submitResult is absent.
 *
 * Reads examSession.attemptId and the stored JWT from keytar / sessionMemory,
 * then POSTs to the Python bridge /result route which calls
 * GET /api/QuizAttempts/result/{attemptId} on the LMS.
 *
 * On success caches the result as submitResult so subsequent calls to
 * bridge:get-submit-result return it directly.
 *
 * Returns: { ok: true, data: object } | { ok: false, redirect: 'login' } | { ok: false, error: object }
 */
ipcMain.handle('bridge:get-result', async () => {
  try {
    const attemptId = examSession?.attemptId;
    if (attemptId == null) {
      return { ok: false, redirect: 'login' };
    }

    const accessToken =
      sessionMemory?.accessToken ||
      (await keytar.getPassword(KEYTAR_SERVICE, 'access-token'));

    const response = await net.fetch(`http://127.0.0.1:${bridgePort}/result`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attemptId, token: accessToken }),
    });

    const body = await response.json();

    if (response.ok) {
      submitResult = body;
      return { ok: true, data: submitResult };
    }

    if (body?.code === 'UNAUTHORIZED') {
      await clearAllKeytarEntries();
      sessionMemory = null;
      mainWindow?.loadFile(path.join(__dirname, 'pages/login/index.html'));
      return; // renderer IPC call never resolves — main.js navigates away
    }

    return { ok: false, error: body };
  } catch {
    return {
      ok: false,
      error: {
        code: 'BRIDGE_ERROR',
        message: 'Unable to reach the server. Please check your connection and try again.',
      },
    };
  }
});

/**
 * bridge:clear-submit-result — Clear in-memory result + session, navigate to Exam Access page.
 *
 * Called when the student clicks "Back to Home" on the Result page.
 * Clears both submitResult and examSession so subsequent navigations to
 * the Result page redirect to Login (SC-004).
 * Authentication session (keytar / sessionMemory) is NOT cleared (FR-005).
 *
 * Returns: { ok: true }
 */
ipcMain.handle('bridge:clear-submit-result', async () => {
  submitResult = null;
  // T028 — Unenroll face recognition embedding on back-to-home (fire-and-forget)
  sendAiRpc('unenrollReference', { sessionId: enrollmentState?.sessionId }).catch(() => { });
  enrollmentState = null;
  deactivateLockdown(); // spec 013 — release all lockdown controls on back-to-home
  examSession = null;
  cheatingReportId = null;
  mainWindow?.loadFile(path.join(__dirname, 'pages/exam-code/index.html'));
  return { ok: true };
});

// ---------------------------------------------------------------------------
// Clip recording IPC handler (spec 011)
// ---------------------------------------------------------------------------

/**
 * save-and-upload-clip — Accept a violation clip Blob from the renderer,
 * write it to a temporary .webm file, forward it to the AI router for
 * encoding and CDN upload, and push a clip:upload-error event to the renderer
 * on failure.
 *
 * Security audit (T027):
 *   - blobArrayBuffer contents are never logged.
 *   - BUNNY_API_KEY and all CDN credentials are never passed in params;
 *     they are read exclusively from environment variables inside the Python
 *     bridge process.
 *   - tempFilePath is never returned to the renderer.
 *   - The access token is read from keytar/sessionMemory and included in the
 *     params; it is used as a Bearer header by the Python bridge and never
 *     echoed back.
 *   - The value returned to ipcRenderer.invoke is only { ok, result: { uploadStatus, evidenceUrl, reasonCode } }.
 *
 * Expected args: { blobArrayBuffer: ArrayBuffer | null, metadata: ClipMetadata }
 * Returns: { ok: true, result: { uploadStatus, evidenceUrl, reasonCode } } | { ok: false, error }
 */
ipcMain.handle('save-and-upload-clip', async (_event, { blobArrayBuffer, metadata } = {}) => {
  try {
    // Handle webcam-unavailable path (blobArrayBuffer is null)
    if (blobArrayBuffer === null || blobArrayBuffer === undefined) {
      const rpcResult = await sendAiRpc('upload_clip', {
        tempFilePath: null,
        metadata: { ...metadata, reportId: cheatingReportId },
      }, 60000);

      const result = rpcResult.ok
        ? rpcResult.result
        : { uploadStatus: 'upload_failed', evidenceUrl: null, reasonCode: 'ROUTER_ERROR' };

      return { ok: true, result };
    }

    // Write the ArrayBuffer to a temp .webm file
    const tmpPath = path.join(os.tmpdir(), `clip-${Date.now()}.webm`);
    const buffer = Buffer.from(blobArrayBuffer);
    process.stderr.write(`[clip] webm size: ${buffer.length} bytes → ${tmpPath}\n`);
    if (buffer.length === 0) {
      return { ok: true, result: { uploadStatus: 'upload_failed', evidenceUrl: null, reasonCode: 'EMPTY_BLOB' } };
    }
    fs.writeFileSync(tmpPath, buffer);

    // Read the access token (same pattern as bridge:start-exam, bridge:submit-exam)
    const accessToken =
      sessionMemory?.accessToken ||
      (await keytar.getPassword(KEYTAR_SERVICE, 'access-token')) ||
      '';

    // Forward to Python bridge — credentials (BUNNY_*) are never in these params
    const rpcResult = await sendAiRpc('upload_clip', {
      tempFilePath: tmpPath,
      metadata: {
        ...metadata,
        token: accessToken,
        reportId: cheatingReportId,
      },
    }, 120000); // 2-minute timeout covers encode + upload + retry

    const result = rpcResult.ok
      ? rpcResult.result
      : { uploadStatus: 'upload_failed', evidenceUrl: null, reasonCode: 'ROUTER_ERROR' };

    // T024: Push clip:upload-error event to renderer on upload failure
    if (result?.uploadStatus === 'upload_failed') {
      mainWindow?.webContents.send('clip:upload-error', {
        studentId: metadata?.studentId ?? '',
        examAttemptId: metadata?.examAttemptId ?? '',
        sessionId: metadata?.sessionId ?? '',
        reasonCode: result.reasonCode ?? 'UPLOAD_EXHAUSTED',
        timestamp: new Date().toISOString(),
      });
    }

    // Return only safe fields — no tempFilePath, no credentials
    return {
      ok: true,
      result: {
        uploadStatus: result?.uploadStatus ?? 'upload_failed',
        evidenceUrl: result?.evidenceUrl ?? null,
        reasonCode: result?.reasonCode ?? null,
      },
    };
  } catch (err) {
    process.stderr.write(`[clip] save-and-upload-clip error: ${err.message}\n`);
    return {
      ok: false,
      error: { code: 'IPC_ERROR', message: err.message },
    };
  }
});

// ---------------------------------------------------------------------------
// Bridge startup (T011)
// ---------------------------------------------------------------------------

/**
 * Poll GET /ping until the bridge responds with HTTP 200.
 *
 * Per research.md decision 1 and contracts/ping.md:
 *   - 20 retries × 500 ms interval = 10 s total timeout
 *   - Returns true on first 200 OK, false after all retries exhausted
 *
 * Uses Electron's built-in net module (works inside the main process
 * and respects Electron's network stack).
 *
 * @param {string} url  Full URL to poll, e.g. "http://127.0.0.1:5050/ping"
 * @param {number} retries
 * @param {number} intervalMs
 * @returns {Promise<boolean>}
 */
function pollBridgeReady(url, retries = 20, intervalMs = 500) {
  return new Promise((resolve) => {
    let attempt = 0;

    function tryOnce() {
      attempt += 1;
      const request = net.request({ method: 'GET', url });

      request.on('response', (response) => {
        if (response.statusCode === 200) {
          resolve(true);
        } else if (attempt < retries) {
          setTimeout(tryOnce, intervalMs);
        } else {
          resolve(false);
        }
        // Drain the response body to avoid hanging connections
        response.on('data', () => { });
      });

      request.on('error', () => {
        if (attempt < retries) {
          setTimeout(tryOnce, intervalMs);
        } else {
          resolve(false);
        }
      });

      request.end();
    }

    tryOnce();
  });
}

/**
 * Spawn the Python bridge process.
 *
 * Per research.md decision 1: uses child_process.spawn (not exec) so stdout
 * and stderr streams are available for logging and error parsing.
 *
 * @param {number} port
 * @param {string} configPath
 */
function startBridge(port, configPath) {
  bridgeState = 'starting';

  const serverScript = path.join(__dirname, '..', 'python_bridge', 'server.py');

  pythonProcess = spawn('python', [
    serverScript,
    '--port', String(port),
    '--config', configPath,
  ], {
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  // Collect last few lines of stderr for diagnostic display on crash
  const stderrLines = [];
  pythonProcess.stderr.on('data', (chunk) => {
    const lines = chunk.toString().split('\n').filter(Boolean);
    stderrLines.push(...lines);
    if (stderrLines.length > 10) stderrLines.splice(0, stderrLines.length - 10);
  });

  // Crash / unexpected exit handler
  pythonProcess.on('close', (code) => {
    if (bridgeState === 'stopping') return; // intentional shutdown — ignore

    bridgeState = 'crashed';
    const lastLines = stderrLines.slice(-3).join(' | ');

    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('bridge:status', {
        type: 'crashed',
        code: 'BRIDGE_CRASHED',
        message: `Python bridge exited unexpectedly (code ${code}). ${lastLines}`,
      });
    }
  });
}

// ---------------------------------------------------------------------------
// Full startup sequence (T012)
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  // Create the main window — load loading page immediately so the user
  // sees feedback while the bridge starts.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false, // show after loading page is ready to avoid white flash
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // required for preload contextBridge
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow.show());

  // ── Lockdown: prevent window close during active exam session (T008) ───────
  mainWindow.on('close', (event) => {
    if (examSession && !lockdownDisabled.closePrevention) {
      event.preventDefault(); // silently block Alt+F4 and OS close gestures
    }
  });

  // ── Lockdown: re-enter fullscreen on OS-forced exit (T009) ─────────────────
  mainWindow.on('leave-full-screen', () => {
    // Do NOT re-enter fullscreen while the student is disconnected (spec 014).
    // The offline transition intentionally exits fullscreen so the student is
    // not trapped; fighting it back on would lock them out of the OS entirely.
    const offlineActive = offlineManager.state === 'OFFLINE' || offlineManager.state === 'LOCKED';
    if (examSession && !lockdownDisabled.fullscreen && !offlineActive) {
      mainWindow.setFullScreen(true); // single best-effort attempt, no retry loop
      appendLockdownAlert('FULLSCREEN_ESCAPE_ATTEMPT', 'fullscreen exit detected');
    }
  });

  // ── Debug: F12 opens DevTools on any page ─────────────────────────────────
  mainWindow.webContents.on('before-input-event', (_e, input) => {
    if (input.type !== 'keyDown') return;

    if (examSession && !lockdownDisabled.shortcutBlocking) {
      // Block all interceptable shortcuts during an active exam (T013)
      const k = input.key;
      const ctrl = input.control;
      const shift = input.shift;

      const isBlocked =
        k === 'F12' ||
        k === 'F11' ||
        k === 'Escape' ||
        k === 'PrintScreen' ||
        (k === 'PrintScreen' && input.alt) ||  // Alt+PrintScreen
        (ctrl && k === 'c') ||
        (ctrl && k === 'v') ||
        (ctrl && k === 'x') ||
        (ctrl && k === 'a') ||
        (ctrl && k === 'w') ||
        (ctrl && shift && k === 'I') ||
        (ctrl && shift && k === 'i');

      if (isBlocked) {
        _e.preventDefault();
      }
      return; // consume all handling during exam
    }

    // Outside exam: F12 opens DevTools normally
    if (input.key === 'F12') {
      mainWindow.webContents.openDevTools({ mode: 'detach' });
    }
  });

  // ── Debug: auto-open DevTools when exam page loads ────────────────────────
  mainWindow.webContents.on('did-finish-load', () => {
    const url = mainWindow.webContents.getURL();
    if (url.includes('exam/index.html')) {
      mainWindow.webContents.openDevTools({ mode: 'detach' });
    }
  });

  await mainWindow.loadFile(path.join(__dirname, 'pages', 'loading', 'loading.html'));

  // --- Validate config before spawning bridge ---
  let config;
  try {
    config = readConfig();
  } catch (configErr) {
    bridgeState = 'failed';
    mainWindow.webContents.send('bridge:status', {
      type: 'config-error',
      code: configErr.code || 'CONFIG_ERROR',
      message: configErr.message,
    });
    return;
  }

  const { baseUrl: _baseUrl, pythonPort, lockdownDisabled: cfgLockdownDisabled } = config;
  Object.assign(lockdownDisabled, cfgLockdownDisabled);
  const configPath = resolveConfigPath();
  const pingUrl = `http://127.0.0.1:${pythonPort}/ping`;

  // Store port at module scope so IPC handlers can reference it
  bridgePort = pythonPort;

  // --- Spawn bridge ---
  startBridge(pythonPort, configPath);
  startAIRouter();

  // --- Poll until ready or timeout ---
  const isReady = await pollBridgeReady(pingUrl);

  if (isReady) {
    bridgeState = 'ready';
    mainWindow.webContents.send('bridge:status', { type: 'ready' });

    // --- Session restore (T015): check for a valid saved session ---
    // Run BEFORE loading any page so the user never sees a flash of login UI
    // if they are already authenticated.
    const savedSession = await getSavedSession();
    if (savedSession !== null) {
      await mainWindow.loadFile(path.join(__dirname, 'pages', 'exam-code', 'index.html'));
      return;
    }

    // No valid session — show the login page
    await mainWindow.loadFile(path.join(__dirname, 'pages', 'login', 'index.html'));
  } else {
    bridgeState = 'failed';
    mainWindow.webContents.send('bridge:status', {
      type: 'failed',
      code: 'BRIDGE_FAILED',
      message: `Bridge did not respond on ${pingUrl} within 10 seconds.`,
    });
  }
});

// ---------------------------------------------------------------------------
// Window lifecycle
// ---------------------------------------------------------------------------

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    bridgeState = 'stopping';
    if (pythonProcess) pythonProcess.kill();
    app.quit();
  }
});

app.on('before-quit', () => {
  bridgeState = 'stopping';
  if (pythonProcess) pythonProcess.kill();
  // T029/T030 — flag locked sessions that never reconnected
  if (offlineManager.state === 'LOCKED') {
    offlineManager.flagSessionForReview();
  }
  offlineManager.stop();
});
