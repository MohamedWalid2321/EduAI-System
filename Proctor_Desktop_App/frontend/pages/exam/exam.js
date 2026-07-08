'use strict';

// ClipRecorder functions come from clip-recorder.js (loaded as a plain <script defer>
// before this file). They are already in the global scope — no re-declaration needed.
// Guards inside DOMContentLoaded handle the case where clip-recorder.js failed to load.

// ---------------------------------------------------------------------------
// Module-scope state (spec 004 — Exam Page)
// ---------------------------------------------------------------------------
let currentIndex = 0;
let examSession = null;
let answerMap = {};
const flagSet = new Set();
let activeStream = null;
let timerInterval = null;
let timerStartTime = null;
let timerTotalSeconds = 0;
let isSubmitting = false;
let autoSubmitted = false;
let remainingSeconds = 0; // U1 fix: track via variable, not DOM parsing
let dashboard = null;
let aiIntervalId = null;
let speechPollIntervalId = null;
let cloudVisionIntervalId = null;   // face-recognition polling
let objectDetectIntervalId = null;  // object-detection polling (independent)
let faceDetectIntervalId = null;
let aiInFlight = false;
let aiCanvas = null;
let aiContext = null;
// Polling intervals — loaded from config.json ui.intervals at startup.
// Defaults below are used only if config is unavailable.
let EYE_GAZE_STREAM_INTERVAL_MS  = 250;
let SPEECH_POLL_INTERVAL_MS       = 2000;
let FACE_RECOGNITION_INTERVAL_MS  = 1000;  // face-recognition (Modal)
let OBJECT_DETECT_INTERVAL_MS     = 1000;  // object-detection  (Modal, independent)
let FACE_DETECT_INTERVAL_MS       = 60000; // local face-detect

// ---------------------------------------------------------------------------
// T012 — DOMContentLoaded: load session and initialise page
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  const overlay = document.getElementById('skeletonOverlay');
  try {
    // Bind ClipRecorder lazily — if clip-recorder.js failed to load,
    // fall back to no-ops so the exam still works.
    if (window.ClipRecorder) {
      // Functions are already in global scope from clip-recorder.js — no reassignment needed.
      console.debug('[exam] ClipRecorder loaded.');
    } else {
      console.warn('[exam] ClipRecorder not available — clip recording disabled.');
    }

    // Load polling intervals from config.json before starting any AI streaming.
    // Falls back silently to the hardcoded defaults if config is unavailable.
    try {
      const cfgResult = await window.bridge.getUiConfig();
      if (cfgResult?.ok && cfgResult.ui?.intervals) {
        const iv = cfgResult.ui.intervals;
        if (iv.eye_gaze_ms)          EYE_GAZE_STREAM_INTERVAL_MS = iv.eye_gaze_ms;
        if (iv.speech_poll_ms)       SPEECH_POLL_INTERVAL_MS      = iv.speech_poll_ms;
        if (iv.face_recognition_ms)  FACE_RECOGNITION_INTERVAL_MS = iv.face_recognition_ms;
        if (iv.object_detection_ms)  OBJECT_DETECT_INTERVAL_MS    = iv.object_detection_ms;
        if (iv.face_detect_ms)       FACE_DETECT_INTERVAL_MS      = iv.face_detect_ms;
      }
    } catch { /* keep defaults */ }

    // T026 — Guard: require completed enrollment before exam page loads
    const enrollResult = await window.bridge.getEnrollmentStatus();
    if (!enrollResult?.enrolled) {
      window.location.replace('../identity-verification/index.html');
      return;
    }

    const result = await window.bridge.getExamSession();

    if (!result || !result.ok) {
      window.location.replace('../login/index.html');
      return;
    }

    examSession = result.session;
    const questions = examSession.questions;

    // C2 fix: guard for empty or missing questions array
    if (!questions?.length) {
      document.getElementById('examTitle').textContent = 'Error: No Questions';
      return;
    }

    // Parse duration "HH:mm:ss"
    const parts = (examSession.duration || '00:00:00').split(':');
    const h = parseInt(parts[0], 10) || 0;
    const m = parseInt(parts[1], 10) || 0;
    const s = parseInt(parts[2], 10) || 0;
    timerTotalSeconds = h * 3600 + m * 60 + s;
    remainingSeconds = timerTotalSeconds;

    initPage();
    renderQuestion(0);
    startTimer();
    initWebcam();

    // ── spec 014: Crash / offline resume restore ───────────────────────────
    // If the exam was restored after a crash or offline navigation, sessionStorage
    // will have 'offlineResume' = '1' and 'offlineResumeData' = JSON.
    try {
      if (sessionStorage.getItem('offlineResume') === '1') {
        const resumeRaw = sessionStorage.getItem('offlineResumeData');
        if (resumeRaw) {
          const resume = JSON.parse(resumeRaw);
          // Restore answers
          if (resume.answers && typeof resume.answers === 'object') {
            // Map may be keyed by string — convert
            for (const [qId, cId] of Object.entries(resume.answers)) {
              answerMap[Number(qId)] = Number(cId);
            }
          }
          // Restore flagged questions
          if (resume.flagged && Array.isArray(resume.flagged)) {
            resume.flagged.forEach(qId => flagSet.add(Number(qId)));
          }
          // Restore question position
          const idx = Math.max(0, Math.min(
            resume.currentQuestionIndex ?? 0,
            examSession.questions.length - 1
          ));
          currentIndex = idx;
          renderQuestion(currentIndex);
          // Restore frozen timer
          if (typeof resume.frozenTimerSeconds === 'number' && resume.frozenTimerSeconds >= 0) {
            clearInterval(timerInterval);
            timerInterval = null;
            remainingSeconds  = resume.frozenTimerSeconds;
            timerTotalSeconds = resume.frozenTimerSeconds;
            timerStartTime    = Date.now();
            startTimer();
          }
          updatePillStates();
        }
        sessionStorage.removeItem('offlineResume');
        // Keep offlineResumeData for heartbeat responses from this page
      }
    } catch (_resumeErr) {
      console.warn('[exam] offline resume restore failed:', _resumeErr.message);
    }

    // Initialize AI Dashboard
    dashboard = new DashboardController();
    dashboard.init();

    // T019 — Route AI alert events to the clip recorder
    window.bridge.onAiEvent((event) => {
      if (event?.params?.type === 'alert') {
        if (typeof handleViolationEvent === 'function') handleViolationEvent(event.params);
      }
    });

    // T032 — release camera tracks on navigation
    window.addEventListener('beforeunload', () => {
      activeStream?.getTracks().forEach(t => t.stop());
      dashboard?.destroy();
      stopAiStreaming();
      if (typeof stopClipRecorder === 'function') stopClipRecorder();
    });

    // ── Lockdown: VM / remote-desktop violation modal (spec 013) ─────────────
    // A modal queue prevents simultaneous modals from stacking.
    const lockdownModalQueue = [];
    let lockdownModalActive = false;

    function showNextLockdownModal() {
      if (lockdownModalActive || lockdownModalQueue.length === 0) return;
      const showFn = lockdownModalQueue.shift();
      lockdownModalActive = true;
      showFn();
    }

    function setExamInteractionDisabled(disabled) {
      const elements = document.querySelectorAll(
        '#choicesList input, #choicesList button, #submitExamBtn, #prevBtn, #nextBtn, #flagBtn, #jumpInput'
      );
      elements.forEach(el => { el.disabled = disabled; });
    }

    function enqueueVmModal(reason) {
      lockdownModalQueue.push(() => {
        const modal = document.getElementById('lockdown-vm-modal');
        const reasonEl = document.getElementById('lockdown-vm-reason');
        const dismissBtn = document.getElementById('lockdown-vm-dismiss');
        if (!modal) { lockdownModalActive = false; showNextLockdownModal(); return; }
        if (reasonEl) reasonEl.textContent = reason || '';
        setExamInteractionDisabled(true);
        modal.classList.remove('hidden');
        dismissBtn.onclick = () => {
          modal.classList.add('hidden');
          setExamInteractionDisabled(false);
          lockdownModalActive = false;
          showNextLockdownModal();
        };
      });
      showNextLockdownModal();
    }

    function enqueueCaptureModal(reason) {
      lockdownModalQueue.push(() => {
        const modal = document.getElementById('lockdown-capture-modal');
        const reasonEl = document.getElementById('lockdown-capture-reason');
        const dismissBtn = document.getElementById('lockdown-capture-dismiss');
        if (!modal) { lockdownModalActive = false; showNextLockdownModal(); return; }
        if (reasonEl) reasonEl.textContent = reason || '';
        setExamInteractionDisabled(true);
        modal.classList.remove('hidden');
        dismissBtn.onclick = () => {
          modal.classList.add('hidden');
          setExamInteractionDisabled(false);
          lockdownModalActive = false;
          showNextLockdownModal();
        };
      });
      showNextLockdownModal();
    }

    window.bridge.onLockdownVmDetected(({ reason }) => {
      enqueueVmModal(reason);
    });

    window.bridge.onLockdownCaptureDetected(({ reason }) => {
      enqueueCaptureModal(reason);
    });

    // ── spec 014: offline resilience IPC handlers ──────────────────────────

    // Proctoring pause (ONLINE → OFFLINE): stop all AI polling
    window.bridge.onProctoringPause(() => {
      clearInterval(timerInterval); timerInterval = null;
      stopAiStreaming();
    });

    // Proctoring resume (OFFLINE → ONLINE): restart all AI polling
    window.bridge.onProctoringResume(() => {
      timerTotalSeconds = remainingSeconds;
      timerStartTime    = Date.now();
      startTimer();
      startAiStreaming();
    });

    // Offline state changed: navigate to offline page when OFFLINE
    window.bridge.onOfflineStateChanged((payload) => {
      if (payload.state === 'OFFLINE') {
        // Send snapshot immediately before navigating away
        const snapshotData = {
          currentQuestionIndex: currentIndex,
          answers:              answerMap,
          frozenTimerSeconds:   remainingSeconds,
          flagged:              Array.from(flagSet),
        };
        window.bridge.sendSnapshotData(snapshotData);
        // Store resume data and offline page init payload for the offline page
        try {
          sessionStorage.setItem('offlineResumeData', JSON.stringify(snapshotData));
          sessionStorage.setItem('offlineInitPayload', JSON.stringify(payload));
        } catch (_) { /* best-effort */ }
        window.location.href = '../offline/index.html';
      }
    });

    // Snapshot heartbeat response: send current state to main process
    window.bridge.onSnapshotRequest(() => {
      window.bridge.sendSnapshotData({
        currentQuestionIndex: currentIndex,
        answers:              answerMap,
        frozenTimerSeconds:   remainingSeconds,
        flagged:              Array.from(flagSet),
      });
    });
  } catch (err) {
    console.error('[exam] DOMContentLoaded error:', err);
    // Show error visibly so it's easy to diagnose without DevTools
    if (overlay) {
      overlay.innerHTML = `<div class="exam-error-overlay">
        <strong>[exam] Error:</strong><br>${err?.message || String(err)}<br><br>
        <em>Stack:</em><br>${err?.stack?.replace(/\n/g, '<br>') || ''}
      </div>`;
      // Don't hide overlay — leave it visible so the user can read the error
      return;
    }
  } finally {
    // Always hide the skeleton — no matter what happens above.
    // (skipped on error path via return above)
    overlay?.classList.add('hidden');
  }
});

// ---------------------------------------------------------------------------
// T014 — initPage: title, pills, and all event wiring
// ---------------------------------------------------------------------------

function initPage() {
  document.getElementById('examTitle').textContent = examSession.quizTitle || 'Exam';

  // Build navigator pills
  const pillsContainer = document.getElementById('pillsContainer');
  pillsContainer.innerHTML = '';
  examSession.questions.forEach((_, i) => {
    const btn = document.createElement('button');
    btn.className = 'nav-pill';
    btn.dataset.index = i;
    btn.textContent = i + 1;
    btn.setAttribute('role', 'listitem');
    btn.setAttribute('aria-label', `Question ${i + 1}`);
    btn.addEventListener('click', () => {
      currentIndex = i;
      renderQuestion(currentIndex);
    });
    pillsContainer.appendChild(btn);
  });

  // T017 — Prev button
  document.getElementById('prevBtn').addEventListener('click', () => {
    if (currentIndex > 0) {
      currentIndex--;
      renderQuestion(currentIndex);
    }
  });

  // T017 — Next button
  document.getElementById('nextBtn').addEventListener('click', () => {
    if (currentIndex < examSession.questions.length - 1) {
      currentIndex++;
      renderQuestion(currentIndex);
    }
  });

  // T018 — Jump to input
  document.getElementById('jumpInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const val = parseInt(e.target.value, 10);
      if (!isNaN(val) && val >= 1 && val <= examSession.questions.length) {
        currentIndex = val - 1;
        renderQuestion(currentIndex);
      }
      e.target.value = '';
      e.target.focus();
    }
  });

  // T021 — Submit Exam button → open confirmation modal
  document.getElementById('submitExamBtn').addEventListener('click', () => {
    const total = examSession.questions.length;
    const answered = Object.keys(answerMap).length;
    const unanswered = total - answered;

    document.getElementById('modalTitle').textContent = examSession.quizTitle || 'Submit Exam';

    if (answered === 0) {
      document.getElementById('modalMessage').textContent =
        "You haven't answered any questions yet.";
    } else if (unanswered > 0) {
      document.getElementById('modalMessage').textContent =
        `${unanswered} question(s) remaining unanswered.`;
    } else {
      document.getElementById('modalMessage').textContent =
        'All questions answered. Ready to submit.';
    }

    document.getElementById('modalBackdrop').classList.remove('hidden');
  });

  // T022 — modal Cancel
  document.getElementById('modalCancelBtn').addEventListener('click', () => {
    document.getElementById('modalBackdrop').classList.add('hidden');
  });

  // T024 — modal Confirm → submit
  document.getElementById('modalConfirmBtn').addEventListener('click', () => {
    submitExam(false);
  });

  // T025 — Retry button
  document.getElementById('retryBtn').addEventListener('click', () => {
    submitExam(autoSubmitted);
  });
}

// ---------------------------------------------------------------------------
// T013 — renderQuestion: display the question at the given index
// ---------------------------------------------------------------------------

const CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

function renderQuestion(index) {
  const q = examSession.questions[index];
  const total = examSession.questions.length;

  // Update badge
  document.getElementById('questionBadge').textContent = `Question ${index + 1} of ${total}`;

  // Update question text
  document.getElementById('questionText').textContent = q.questionText;

  // Rebuild choices list
  const list = document.getElementById('choicesList');
  list.innerHTML = '';

  q.choices.forEach((choice, ci) => {
    const li = document.createElement('li');
    li.className = 'choice-item';
    li.dataset.questionId = q.id;
    li.dataset.choiceId = choice.id;
    li.setAttribute('role', 'radio');
    li.setAttribute('aria-checked', answerMap[q.id] === choice.id ? 'true' : 'false');

    if (answerMap[q.id] === choice.id) {
      li.classList.add('is-selected');
    }

    const letter = document.createElement('span');
    letter.className = 'choice-letter';
    letter.setAttribute('aria-hidden', 'true');
    letter.textContent = CHOICE_LETTERS[ci] || String(ci + 1);

    const text = document.createElement('span');
    text.className = 'choice-text';
    text.textContent = choice.choiceText;

    li.appendChild(letter);
    li.appendChild(text);

    // T019 — choice click handler
    li.addEventListener('click', () => {
      const qId = +li.dataset.questionId;
      const cId = +li.dataset.choiceId;
      answerMap[qId] = cId;

      list.querySelectorAll('.choice-item').forEach(el => {
        el.classList.remove('is-selected');
        el.setAttribute('aria-checked', 'false');
      });
      li.classList.add('is-selected');
      li.setAttribute('aria-checked', 'true');

      updatePillStates();

      // spec 014 — persist snapshot on every answer change (FR-013)
      if (window.bridge?.sendSnapshotData) {
        window.bridge.sendSnapshotData({
          currentQuestionIndex: currentIndex,
          answers:              answerMap,
          frozenTimerSeconds:   remainingSeconds,
          flagged:              Array.from(flagSet),
        });
      }
    });

    list.appendChild(li);
  });

  // Update Prev/Next disabled states
  document.getElementById('prevBtn').disabled = index === 0;
  document.getElementById('nextBtn').disabled = index === total - 1;

  // Update flag button state (T029 — re-bind per question)
  const flagBtn = document.getElementById('flagBtn');
  if (flagSet.has(q.id)) {
    flagBtn.classList.add('is-flagged');
  } else {
    flagBtn.classList.remove('is-flagged');
  }

  flagBtn.onclick = () => {
    if (flagSet.has(q.id)) {
      flagSet.delete(q.id);
      flagBtn.classList.remove('is-flagged');
    } else {
      flagSet.add(q.id);
      flagBtn.classList.add('is-flagged');
    }
    updatePillStates();

    // spec 014 — persist snapshot on flag change
    if (window.bridge?.sendSnapshotData) {
      window.bridge.sendSnapshotData({
        currentQuestionIndex: currentIndex,
        answers:              answerMap,
        frozenTimerSeconds:   remainingSeconds,
        flagged:              Array.from(flagSet),
      });
    }
  };

  updatePillStates();
}

// ---------------------------------------------------------------------------
// T015 — updatePillStates: reflect answered / current / flagged per pill
// U2 fix: use question.id to look up answerMap, not array index
// ---------------------------------------------------------------------------

function updatePillStates() {
  const pills = document.querySelectorAll('.nav-pill');
  pills.forEach((pill, i) => {
    const question = examSession.questions[i];
    pill.classList.remove('is-answered', 'is-current', 'is-flagged');

    // U2 fix: check by question id, not array index
    if (question.id in answerMap) {
      pill.classList.add('is-answered');
    }
    if (i === currentIndex) {
      pill.classList.add('is-current');
    }
    // Flagged takes priority — applied last, overwrites answered
    if (flagSet.has(question.id)) {
      pill.classList.add('is-flagged');
    }
  });
}

// ---------------------------------------------------------------------------
// T027 — startTimer: drift-corrected countdown timer
// ---------------------------------------------------------------------------

function formatSeconds(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
}

function startTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
  }
  timerStartTime = Date.now();

  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - timerStartTime) / 1000);
    remainingSeconds = Math.max(0, timerTotalSeconds - elapsed); // U1 fix

    document.getElementById('timerDisplay').textContent = formatSeconds(remainingSeconds);

    const timerChip = document.getElementById('timerChip');
    if (remainingSeconds <= 300) {
      timerChip.classList.add('is-warning');
    } else {
      timerChip.classList.remove('is-warning');
    }

    if (remainingSeconds === 0 && !autoSubmitted) {
      clearInterval(timerInterval);
      timerInterval = null;
      autoSubmitted = true;
      submitExam(true);
    }
  }, 1000);
}

// ---------------------------------------------------------------------------
// T028 — resumeTimer: restart timer using remainingSeconds variable (U1 fix)
// ---------------------------------------------------------------------------

function resumeTimer() {
  // U1 fix: use remainingSeconds directly instead of parsing DOM text
  timerTotalSeconds = remainingSeconds;
  timerStartTime = Date.now();
  startTimer();
}

// ---------------------------------------------------------------------------
// T023 — submitExam: build payload, call bridge, handle result / error
// ---------------------------------------------------------------------------

async function submitExam(isAutoSubmit) {
  // Guard: prevent double-submit (including duplicate auto-submit)
  if (isSubmitting) return;

  isSubmitting = true;
  document.body.classList.add('exam-page-loading');
  document.getElementById('modalBackdrop').classList.add('hidden');
  document.getElementById('errorBanner').classList.add('hidden');

  // Pause timer
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  // Build answers payload (only answered questions)
  const answers = Object.entries(answerMap).map(([qId, cId]) => ({
    questionId: +qId,
    choiceId: cId,
  }));

  let result;
  try {
    result = await window.bridge.submitExam(answers);
  } catch {
    result = {
      ok: false,
      error: {
        code: 'BRIDGE_ERROR',
        message: 'Unable to reach the server. Please check your connection and try again.',
      },
    };
  }

  // I1 fix: guard against undefined result (UNAUTHORIZED — main.js navigated away)
  if (!result) return;

  if (result.ok) {
    // T023 + T030 — flush any in-progress clip before navigating away
    try {
      if (typeof forceFinalize === 'function') await forceFinalize();
    } catch (_) { /* non-blocking */ }
    if (typeof stopClipRecorder === 'function') stopClipRecorder();
    window.location.href = '../result/index.html';
    return;
  }

  // Error recovery
  isSubmitting = false;
  document.body.classList.remove('exam-page-loading');
  document.getElementById('errorBannerText').textContent =
    result.error?.message || 'An unexpected error occurred.';
  document.getElementById('errorBanner').classList.remove('hidden');

  if (!isAutoSubmit) {
    // Resume timer for manual-submit failures only
    resumeTimer();
  }
  // For auto-submit failures: leave timer frozen at 00:00:00
}

// ---------------------------------------------------------------------------
// T031 — initWebcam: request camera access and display stream
// ---------------------------------------------------------------------------

async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    activeStream = stream;
    document.getElementById('webcamFeed').srcObject = stream;
    startAiStreaming();

    // T019 — Start clip recorder, sharing the already-open AI stream.
    // Previously clip-recorder opened its own getUserMedia, creating a second
    // independent camera capture pipeline and a second hardware encoder session.
    // Both sessions competed for the same encoder hardware, causing I-frame
    // sequence interruptions (bitstream corruption at splice boundaries).
    // Passing the stream here lets clip-recorder clone it — same single camera
    // pipeline, no encoder contention.
    try {
      const cfgResult = await window.bridge.getUiConfig();
      const clipCfg = cfgResult?.ok ? cfgResult.ui?.clip_recording ?? null : null;
      if (typeof initClipRecorder === 'function') await initClipRecorder(clipCfg, examSession, stream);
    } catch (clipErr) {
      console.warn('[exam] clip recorder init failed:', clipErr.message);
    }
  } catch {
    document.getElementById('webcamFeed').classList.add('hidden');
    document.getElementById('cameraUnavailable').classList.remove('hidden');
  }
}

// ---------------------------------------------------------------------------
// AI streaming — capture frames and send predict to router
// ---------------------------------------------------------------------------

function startAiStreaming() {
  if (aiIntervalId || !window.bridge?.aiRpc) return;

  const video = document.getElementById('webcamFeed');
  if (!video) return;

  // ── Shared canvas kept only for legacy eye-gaze (high-frequency, fire-and-forget) ──
  aiCanvas = document.createElement('canvas');
  aiCanvas.width = 320;
  aiCanvas.height = 240;
  aiContext = aiCanvas.getContext('2d');

  // ── Helper: snapshot video into a private canvas and return a data-URL ────────────
  // Each call creates an isolated canvas so concurrent intervals never overwrite
  // each other's in-progress frame (root cause of the YOLO false-positive detections).
  function captureFrame(w = 320, h = 240, quality = 0.6) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    c.getContext('2d').drawImage(video, 0, 0, w, h);
    return c.toDataURL('image/jpeg', quality);
  }

  // ── Eye-gaze: shared canvas is fine here because these calls are fire-and-forget ──
  aiIntervalId = setInterval(async () => {
    if (!aiContext) return;
    if (video.readyState < 2) return;

    try {
      aiContext.drawImage(video, 0, 0, aiCanvas.width, aiCanvas.height);
      const frame = aiCanvas.toDataURL('image/jpeg', 0.6);
      const currentQuestion = examSession?.questions?.[currentIndex];
      const questionId = currentQuestion?.id ?? null;
      // DB column: IsAllowableToLookDown — API sends PascalCase (C# default).
      const isAllowableToLookDown = currentQuestion?.IsAllowableToLookDown
                                 ?? currentQuestion?.isAllowableToLookDown
                                 ?? false;

      // ── DEBUG: log question object once per unique question ─────────────
      if (!window._gazeDebugQuestions) window._gazeDebugQuestions = new Set();
      if (questionId !== null && !window._gazeDebugQuestions.has(questionId)) {
        window._gazeDebugQuestions.add(questionId);
        console.log(`[gaze-debug] Q${questionId} keys:`, currentQuestion ? Object.keys(currentQuestion) : 'null');
        console.log(`[gaze-debug] Q${questionId} IsAllowableToLookDown:`, currentQuestion?.IsAllowableToLookDown);
        console.log(`[gaze-debug] Q${questionId} resolved isAllowableToLookDown:`, isAllowableToLookDown);
      }
      // ── END DEBUG ────────────────────────────────────────────────────────

      window.bridge.aiRpc('predict', { service: 'eye-gaze', frame, questionId, isAllowableToLookDown }).catch(() => { });
    } catch {
      // Eye-gaze may be disabled; ignore polling errors.
    }
  }, EYE_GAZE_STREAM_INTERVAL_MS);

  // ── Face-recognition: independent interval and snapshot ──────────────────────────
  cloudVisionIntervalId = setInterval(async () => {
    try {
      if (video.readyState < 2) return;
      const frame = captureFrame();
      const questionId = examSession?.questions?.[currentIndex]?.id ?? null;
      await window.bridge.aiRpc('predict', { service: 'face-recognition', frame, questionId });
    } catch {
      // Service may be unconfigured/stopped; ignore transient router errors.
    }
  }, FACE_RECOGNITION_INTERVAL_MS);

  // ── Object-detection: independent interval and snapshot ───────────────────────────
  objectDetectIntervalId = setInterval(async () => {
    try {
      if (video.readyState < 2) return;
      const frame = captureFrame();
      const questionId = examSession?.questions?.[currentIndex]?.id ?? null;
      await window.bridge.aiRpc('predict', { service: 'object-detection', frame, questionId });
    } catch {
      // Service may be unconfigured/stopped; ignore transient router errors.
    }
  }, OBJECT_DETECT_INTERVAL_MS);

  // ── Face detection: independent snapshot ─────────────────────────────────────────
  faceDetectIntervalId = setInterval(async () => {
    try {
      if (video.readyState < 2) return;
      const frame = captureFrame();
      const questionId = examSession?.questions?.[currentIndex]?.id ?? null;
      await window.bridge.aiRpc('predict', { service: 'face-detection', frame, questionId });
    } catch {
      // Services may be unconfigured/stopped; ignore transient router errors.
    }
  }, FACE_DETECT_INTERVAL_MS);

  // ── Speech detection: no frame needed (mic-based) ────────────────────────────────
  speechPollIntervalId = setInterval(async () => {
    try {
      const questionId = examSession?.questions?.[currentIndex]?.id ?? null;
      await window.bridge.aiRpc('predict', { service: 'speech-detection', frame: 'MIC_POLL', questionId });
    } catch {
      // Ignore transient router errors; status polling handles recovery.
    }
  }, SPEECH_POLL_INTERVAL_MS);
}


function stopAiStreaming() {
  if (aiIntervalId) {
    clearInterval(aiIntervalId);
    aiIntervalId = null;
  }
  if (speechPollIntervalId) {
    clearInterval(speechPollIntervalId);
    speechPollIntervalId = null;
  }
  if (cloudVisionIntervalId) {
    clearInterval(cloudVisionIntervalId);
    cloudVisionIntervalId = null;
  }
  if (objectDetectIntervalId) {
    clearInterval(objectDetectIntervalId);
    objectDetectIntervalId = null;
  }
  if (faceDetectIntervalId) {
    clearInterval(faceDetectIntervalId);
    faceDetectIntervalId = null;
  }
  aiCanvas = null;
  aiContext = null;
}
