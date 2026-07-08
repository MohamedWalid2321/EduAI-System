'use strict';

/**
 * clip-recorder.js — Violation clip ring-buffer recorder for Lumina AI.
 *
 * Captures a rolling 10-second pre-violation buffer via the browser MediaRecorder
 * API. When a violation alert fires, it locks state, collects 10s of post-violation
 * footage, merges the buffers into a single Blob, converts it to an ArrayBuffer,
 * and forwards it (with metadata) to the Electron main process for encoding and
 * CDN upload.
 *
 * Security: No CDN credentials are handled here. The renderer only receives the
 * resulting evidenceUrl. The token for the backend POST is read by main.js from the
 * OS keychain and injected into the JSON-RPC params — never visible to this module.
 *
 * Public API:
 *   initClipRecorder(config, examSession)  — start ring buffer
 *   handleViolationEvent(alert)            — trigger or absorb a violation
 *   forceFinalize()                        — flush on exam end (async)
 *   stopClipRecorder()                     — release webcam + reset state
 */

// ---------------------------------------------------------------------------
// Module-scope state (T012)
// ---------------------------------------------------------------------------
/**
 * The very first chunk emitted by MediaRecorder contains the WebM initialization
 * segment (codec headers). Without it ffmpeg cannot decode any subsequent cluster.
 * We preserve it here so every merged clip blob is always decodable.
 * @type {Blob | null}
 */
let webmInitChunk = null;
/** @type {Blob | null} Header-only portion of webmInitChunk (EBML+Tracks, no T=0 cluster frames) */
let webmHeaderOnlyBlob = null;

/** @type {Array<{blob: Blob, timestamp: string}>} Rolling pre-violation buffer */
let ringBuffer = [];

/** @type {boolean} True while post-violation chunks are being collected */
let isCapturingPost = false;

/**
 * True for the ENTIRE video clip lifecycle — from the moment a violation fires
 * until the IPC upload call resolves (or rejects). Prevents a second recording
 * from starting while a previous clip is still being encoded or uploaded.
 * isCapturingPost resets early (so the ring buffer can refill); this flag stays
 * locked until the upload is fully done.
 * @type {boolean}
 */
let isVideoClipInFlight = false;

/**
 * Promise for the currently-running (or last-completed) video upload.
 * Stored so forceFinalize() can await it even after isCapturingPost has already
 * been reset to false (which happens before the first await in finalizeClip).
 * Initialised to Promise.resolve() so awaiting it is always safe.
 * @type {Promise<void>}
 */
let _videoUploadPromise = Promise.resolve();

/** @type {Array<{violationType: string, confidence: number, timestamp: string, description: string}>} */
let clipViolations = [];

/** @type {Array<{blob: Blob, timestamp: string}>} Post-violation chunks */
let postChunks = [];

/** @type {Array<{blob: Blob, timestamp: string}>} Snapshot of pre-violation buffer at trigger moment */
let preViolationChunksSnapshot = [];

/** @type {MediaRecorder | null} */
let mediaRecorder = null;

/** @type {MediaStream | null} */
let clipStream = null;

/** @type {object | null} clip_recording section from config.json */
let clipConfig = null;

/** @type {object | null} The active exam session */
let activeExamSession = null;

// ── Audio recorder state (speech violations only) ─────────────────────────────

/**
 * Alert codes produced by the speech-detection service.
 * When a violation's code matches one of these, the audio recorder is used
 * instead of the video recorder.
 */
const SPEECH_ALERT_CODES = new Set(['SPEECH_DETECTED']);
const SPEECH_NO_CLIP_CODES = new Set(['SPEECH_CHEATING_FLAGGED']);

/** @type {MediaRecorder | null} */
let audioRecorder = null;

/** @type {MediaStream | null} */
let audioStream = null;

/** @type {Array<{blob: Blob, timestamp: string}>} Rolling audio pre-violation buffer */
let audioRingBuffer = [];

/** @type {boolean} True while post-violation audio chunks are being collected */
let isCapturingAudioPost = false;

/**
 * True for the ENTIRE audio clip lifecycle — from violation fire to upload
 * completion. Mirrors isVideoClipInFlight for the speech recording path.
 * @type {boolean}
 */
let isAudioClipInFlight = false;

/**
 * Promise for the currently-running (or last-completed) audio upload.
 * Same pattern as _videoUploadPromise — allows forceFinalize() to await an
 * upload that started before isCapturingAudioPost was reset.
 * @type {Promise<void>}
 */
let _audioUploadPromise = Promise.resolve();

/** @type {Array<{blob: Blob, timestamp: string}>} Audio post-violation chunks */
let audioPostChunks = [];

/** @type {Array<{blob: Blob, timestamp: string}>} Snapshot of audio pre-buffer at trigger moment */
let audioPreSnapshot = [];

/** @type {Array<{violationType: string, confidence: number, timestamp: string, description: string}>} */
let audioViolations = [];

/**
 * The very first chunk from the audio MediaRecorder contains the WebM
 * initialization segment (Opus codec headers). Without it ffmpeg cannot
 * decode any subsequent audio cluster. We preserve it here so every merged
 * audio blob is always decodable — same pattern as webmInitChunk for video.
 * @type {Blob | null}
 */
let audioWebmInitChunk = null;
/** @type {Blob | null} Header-only portion of audioWebmInitChunk (EBML+Tracks, no T=0 cluster frames) */
let audioWebmHeaderOnlyBlob = null;

// ---------------------------------------------------------------------------
// WebM header extraction helper
// ---------------------------------------------------------------------------

/**
 * Extract the WebM container header bytes (EBML + SegmentInfo + Tracks)
 * without any Cluster data.
 *
 * MediaRecorder bundles codec headers AND the first video/audio cluster into
 * the very first ondataavailable chunk. When this chunk is later prepended to
 * a ring-buffer clip (violation fires after pre_buffer_seconds), ffmpeg sees a
 * frame at T=0ms as the very first PTS. setpts=PTS-STARTPTS then subtracts 0,
 * leaving ring-buffer frames at e.g. T=50,000ms unchanged → reported MP4
 * duration of ~70s even though only 20s of frames exist.
 *
 * The Cluster element always begins with EBML ID 0x1F 0x43 0xB6 0x75.
 * Slicing at that offset keeps only the header bytes so setpts=PTS-STARTPTS
 * anchors at the first ring-buffer frame instead.
 *
 * @param {Blob} initChunk - First MediaRecorder chunk (header + first cluster)
 * @returns {Promise<Blob|null>} Header-only Blob, or null if Cluster ID not found
 */
async function _extractWebmHeader(initChunk) {
  try {
    const buf = await initChunk.arrayBuffer();
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.length - 3; i++) {
      if (bytes[i] === 0x1F && bytes[i + 1] === 0x43 && bytes[i + 2] === 0xB6 && bytes[i + 3] === 0x75) {
        return new Blob([buf.slice(0, i)], { type: initChunk.type });
      }
    }
    return null; // Cluster not found — caller falls back to full init chunk
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// T013: Init — start webcam stream + ring buffer
// ---------------------------------------------------------------------------

/**
 * Initialise the clip recorder.
 * Must be called after the webcam stream is established on the exam page.
 *
 * @param {object}      config         The clip_recording block from config.json
 * @param {object}      examSession    The active ExamSession from main.js
 * @param {MediaStream} [existingStream] The AI webcam stream already opened by
 *   exam.js. When provided it is cloned (independent lifecycle, same underlying
 *   camera pipeline) so no second getUserMedia is needed. This eliminates the
 *   second hardware encoder session that was causing I-frame contention.
 *   If omitted, falls back to its own getUserMedia call.
 */
async function initClipRecorder(config, examSession, existingStream = null) {
  // Guard: stop any previously-running recorder before reinitialising.
  // Without this, a second call leaks the old MediaStream (webcam LED stays
  // on), leaves a stale ondataavailable handler writing to the ring buffer,
  // and silently discards the new recorder's init segment (webmInitChunk is
  // not null, so the new codec headers are never saved → corrupted clips).
  if (mediaRecorder || audioRecorder) {
    stopClipRecorder();
  }

  clipConfig = config || {};
  activeExamSession = examSession;

  // Guard: if browser doesn't support MediaRecorder, degrade gracefully.
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    console.warn('[clip-recorder] MediaRecorder not supported — clip recording disabled.');
    return;
  }

  // Approach A: reuse the existing AI stream via clone() so only one camera
  // capture pipeline is open. MediaStream.clone() gives an independent stream
  // object (its own lifecycle; stopClipRecorder stops it without affecting the
  // AI feed) while sharing the same underlying camera source.
  // Fall back to getUserMedia only if no stream was provided (e.g., unit tests).
  if (existingStream) {
    clipStream = existingStream.clone();
  } else {
    try {
      clipStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      console.warn('[clip-recorder] getUserMedia failed — clip recording disabled:', err.message);
      return;
    }
  }

  // Approach C: prefer H.264 (AVC Baseline) over VP9.
  // H.264 uses a much shorter GOP by default, so even if a splice point lands
  // on a P-frame the corruption affects only a few frames. VP9's variable-length
  // GOP can produce 60+ corrupted frames from a single missing keyframe.
  // The mimeType value from config is used first; the runtime checks below add
  // a final safeguard for environments that don't support the configured codec.
  const preferH264 = MediaRecorder.isTypeSupported('video/webm;codecs=h264');
  const preferVP9  = MediaRecorder.isTypeSupported('video/webm;codecs=vp9');
  const configuredMime = clipConfig.mime_type || 'video/webm;codecs=h264';
  const mimeType =
    MediaRecorder.isTypeSupported(configuredMime) ? configuredMime :
    preferH264 ? 'video/webm;codecs=h264' :
    preferVP9  ? 'video/webm;codecs=vp9'  : '';
  const chunkMs = clipConfig.chunk_duration_ms || 1000;

  // videoKeyFrameIntervalDuration forces an I-frame at every chunk boundary so
  // each ring-buffer chunk is independently decodable. This prevents the
  // "P-frame on top of missing base" corruption when chunks are spliced.
  try {
    mediaRecorder = new MediaRecorder(clipStream, { mimeType, videoKeyFrameIntervalDuration: chunkMs });
  } catch {
    try {
      mediaRecorder = new MediaRecorder(clipStream, { mimeType });
    } catch {
      try {
        mediaRecorder = new MediaRecorder(clipStream);
      } catch (err3) {
        console.warn('[clip-recorder] MediaRecorder construction failed:', err3.message);
        return;
      }
    }
  }

  mediaRecorder.ondataavailable = (e) => {
    if (!e.data || e.data.size === 0) return;
    // Save the very first chunk as the WebM init segment (contains codec headers)
    if (webmInitChunk === null) {
      webmInitChunk = e.data;
      // Asynchronously extract header-only bytes (no T=0 cluster) so
      // finalizeClip can prepend them without anchoring PTS-STARTPTS at T=0.
      // Resolves in microseconds — well before any violation can be 10s old.
      _extractWebmHeader(e.data).then(hdr => { webmHeaderOnlyBlob = hdr; });
    }
    const chunk = { blob: e.data, timestamp: new Date().toISOString() };

    // T013: Maintain rolling ring buffer (pre-violation) — time-based trimming.
    // Comparing chunk count to seconds breaks when the browser emits chunks at
    // irregular intervals (e.g. after GPU stalls or tab throttling). Using wall-
    // clock timestamps guarantees the buffer always holds exactly
    // pre_buffer_seconds of real elapsed time.
    const preBufferMs = (clipConfig.pre_buffer_seconds || 10) * 1000;
    if (!isCapturingPost) {
      ringBuffer.push(chunk);
      const nowMs = Date.now();
      while (ringBuffer.length > 1 &&
             nowMs - new Date(ringBuffer[0].timestamp).getTime() > preBufferMs) {
        ringBuffer.shift();
      }
    }

    // T015: Collect post-violation chunks when lock is active — time-based.
    if (isCapturingPost) {
      postChunks.push(chunk);
      const postBufferMs = (clipConfig.post_buffer_seconds || 10) * 1000;
      const postDurationMs = Date.now() - new Date(postChunks[0].timestamp).getTime();
      if (postDurationMs >= postBufferMs) {
        const preSnap = preViolationChunksSnapshot.slice();
        // Store the Promise so forceFinalize() can await it if exam ends
        // before the upload completes.
        _videoUploadPromise = finalizeClip(preSnap);
      }
    }
  };

  // Fix C: surface silent MediaRecorder stalls so the clip is finalized
  // even if the encoder crashes mid-capture (e.g. hardware encoder contention).
  mediaRecorder.onerror = (e) => {
    console.error('[clip-recorder] MediaRecorder error:', e.error?.message ?? e.error ?? 'unknown');
    if (isCapturingPost) {
      console.warn('[clip-recorder] Finalizing partial clip due to encoder error.');
      _videoUploadPromise = finalizeClip(preViolationChunksSnapshot.slice());
    } else {
      _resetCaptureState();
      isVideoClipInFlight = false;
    }
  };

  mediaRecorder.start(chunkMs);
  console.debug('[clip-recorder] Ring buffer started.');

  // Start audio ring buffer for speech-violation recording
  await _initAudioRecorder();
}

// ---------------------------------------------------------------------------
// Audio recorder init (speech violations)
// ---------------------------------------------------------------------------

/**
 * Initialise the audio-only MediaRecorder for speech violation clips.
 * Runs a separate getUserMedia({ audio: true }) so it is fully independent
 * from the video stream.
 */
async function _initAudioRecorder() {
  if (!navigator.mediaDevices || !window.MediaRecorder) return;

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    console.warn('[clip-recorder] audio getUserMedia failed — speech audio recording disabled:', err.message);
    return;
  }

  const chunkMs = clipConfig?.chunk_duration_ms || 1000;

  try {
    audioRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });
  } catch {
    try {
      audioRecorder = new MediaRecorder(audioStream);
    } catch (err2) {
      console.warn('[clip-recorder] audio MediaRecorder construction failed:', err2.message);
      return;
    }
  }

  audioRecorder.ondataavailable = (e) => {
    if (!e.data || e.data.size === 0) return;
    // Preserve the first chunk as the WebM init segment (Opus codec headers).
    // This mirrors the webmInitChunk pattern for video and is critical:
    // if the ring buffer trims the first chunk away, subsequent clusters
    // cannot be decoded by ffmpeg, producing ENCODE_FAILED.
    if (audioWebmInitChunk === null) {
      audioWebmInitChunk = e.data;
      // Same pattern as video: extract header-only bytes for finalizeAudioClip.
      _extractWebmHeader(e.data).then(hdr => { audioWebmHeaderOnlyBlob = hdr; });
    }
    const chunk = { blob: e.data, timestamp: new Date().toISOString() };

    // Time-based trimming — same rationale as the video ring buffer.
    const preBufferMs = (clipConfig?.pre_buffer_seconds || 10) * 1000;
    if (!isCapturingAudioPost) {
      audioRingBuffer.push(chunk);
      const nowMs = Date.now();
      while (audioRingBuffer.length > 1 &&
             nowMs - new Date(audioRingBuffer[0].timestamp).getTime() > preBufferMs) {
        audioRingBuffer.shift();
      }
    }

    if (isCapturingAudioPost) {
      audioPostChunks.push(chunk);
      const postBufferMs = (clipConfig?.post_buffer_seconds || 10) * 1000;
      const postDurationMs = Date.now() - new Date(audioPostChunks[0].timestamp).getTime();
      if (postDurationMs >= postBufferMs) {
        const preSnap = audioPreSnapshot.slice();
        // Store the Promise so forceFinalize() can await it if exam ends
        // before the upload completes.
        _audioUploadPromise = finalizeAudioClip(preSnap);
      }
    }
  };

  // Fix C (audio): surface silent audio MediaRecorder stalls.
  audioRecorder.onerror = (e) => {
    console.error('[clip-recorder] Audio MediaRecorder error:', e.error?.message ?? e.error ?? 'unknown');
    if (isCapturingAudioPost) {
      console.warn('[clip-recorder] Finalizing partial audio clip due to encoder error.');
      _audioUploadPromise = finalizeAudioClip(audioPreSnapshot.slice());
    } else {
      _resetAudioCaptureState();
      isAudioClipInFlight = false;
    }
  };

  audioRecorder.start(chunkMs);
  console.debug('[clip-recorder] Audio ring buffer started.');
}

// ---------------------------------------------------------------------------
// T014: Violation trigger handler
// ---------------------------------------------------------------------------

/**
 * Called by exam.js whenever the AI router emits an alert notification.
 *
 * @param {object} alert  Alert params from the JSON-RPC notification
 *   Expected fields: code, evidence?.confidence, timestamp, message
 */
function handleViolationEvent(alert) {
  // Speech violations → audio-only recording path
  if (SPEECH_ALERT_CODES.has(alert.code)) {
    _handleSpeechViolation(alert);
    return;
  }

  // SPEECH_CHEATING_FLAGGED is a summary event — no clip needed (clips were
  // already uploaded per-strike via SPEECH_DETECTED).
  if (SPEECH_NO_CLIP_CODES.has(alert.code)) {
    console.debug(`[clip-recorder] No clip for ${alert.code} — per-strike clips already uploaded.`);
    return;
  }

  // All other violations → video recording path
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    // T026: Webcam unavailable — emit a zero-blob record so the backend
    // still receives an event indicating a violation occurred.
    // Skip if a clip is already in-flight (recording or uploading).
    if (!isCapturingPost && !isVideoClipInFlight) {
      _handleWebcamUnavailable(alert);
    }
    return;
  }

  const entry = {
    violationType: alert.code || 'UNKNOWN',
    confidence: alert.evidence?.confidence ?? 0,
    timestamp: alert.timestamp || new Date().toISOString(),
    description: alert.message || alert.code || 'Violation detected',
  };

  if (isCapturingPost || isVideoClipInFlight) {
    // Lock is held (recording OR uploading) — absorb into the current clip.
    console.debug(`[clip-recorder] Violation absorbed into active clip: ${alert.code}`);
    clipViolations.push(entry);
    return;
  }

  // First violation — acquire full lifecycle lock and trigger capture
  isVideoClipInFlight = true;
  preViolationChunksSnapshot = ringBuffer.slice(); // snapshot
  ringBuffer = [];
  isCapturingPost = true;
  postChunks = [];
  clipViolations = [entry];
  console.debug(`[clip-recorder] Capture triggered by: ${alert.code}`);

  // Fix D: post-snap safety-net timer. If ondataavailable never fires for the
  // post window (e.g. stalled encoder), the 10-second check inside the handler
  // never triggers. This timeout guarantees finalization regardless.
  const postMs = (clipConfig?.post_buffer_seconds || 10) * 1000;
  setTimeout(() => {
    if (isCapturingPost) {
      console.warn('[clip-recorder] Post-snap timeout safety-net fired — finalizing clip.');
      _videoUploadPromise = finalizeClip(preViolationChunksSnapshot.slice());
    }
  }, postMs + 500);
}

// ---------------------------------------------------------------------------
// T016: Finalise — merge blobs, build metadata, invoke IPC
// ---------------------------------------------------------------------------

/**
 * Merge pre + post buffers into a single Blob and send to main.js for upload.
 * Resets all post-capture state after the IPC call returns.
 *
 * @param {Array<{blob: Blob, timestamp: string}>} preChunks  Snapshot of pre-violation buffer
 */
async function finalizeClip(preChunks) {
  if (postChunks.length === 0 && preChunks.length === 0) {
    console.warn('[clip-recorder] finalizeClip called with empty buffers — skipping.');
    _resetCaptureState();
    return;
  }

  // Local copies before async gap
  const localPost = postChunks.slice();
  const localViolations = clipViolations.slice();
  const localSession = activeExamSession;

  // Reset capture lock immediately so new violations are handled correctly
  _resetCaptureState();

  // Merge blobs
  const mimeType = clipConfig?.mime_type || 'video/webm';
  // Prepend WebM header bytes so ffmpeg can decode the stream regardless of
  // whether the ring buffer still contains the init chunk.
  // Use the header-only blob (no T=0 cluster frames) so setpts=PTS-STARTPTS
  // anchors at the first ring-buffer PTS rather than at T=0. Falls back to
  // the full init chunk if extraction failed or hasn't resolved yet.
  const firstDataBlob = preChunks[0]?.blob ?? localPost[0]?.blob ?? null;
  const _initBlobForHeader = webmHeaderOnlyBlob ?? webmInitChunk;
  const needsInitChunk = _initBlobForHeader && firstDataBlob !== webmInitChunk;
  const allBlobs = [
    ...(needsInitChunk ? [_initBlobForHeader] : []),
    ...preChunks.map(c => c.blob),
    ...localPost.map(c => c.blob),
  ];

  if (allBlobs.length === 0) {
    console.warn('[clip-recorder] No blobs to merge.');
    return;
  }

  const merged = new Blob(allBlobs, { type: mimeType });

  // Timestamps
  const captureWindowStart =
    preChunks[0]?.timestamp ?? localPost[0]?.timestamp ?? new Date().toISOString();
  const captureWindowEnd =
    localPost[localPost.length - 1]?.timestamp ?? new Date().toISOString();

  // T021: Multi-violation description
  const firstViolation = localViolations[0] || {};
  const description =
    localViolations.length === 1
      ? firstViolation.description || firstViolation.violationType || 'Violation detected'
      : `${firstViolation.description || firstViolation.violationType} + ${localViolations.length - 1} additional violation(s) in clip`;

  // Build ClipMetadata per ipc-save-and-upload-clip.json contract
  const metadata = {
    studentId: String(localSession?.studentId || localSession?.attemptId || 'unknown'),
    examAttemptId: String(localSession?.attemptId || ''),
    sessionId: String(localSession?.attemptId || ''),
    captureWindowStart,
    captureWindowEnd,
    primaryViolationType: firstViolation.violationType || 'VIOLATION',
    primaryConfidence: firstViolation.confidence ?? 0,
    description,
    allViolations: localViolations.map(v => ({
      violationType: v.violationType,
      confidence: v.confidence,
      timestamp: v.timestamp,
      description: v.description,
    })),
  };

  let blobArrayBuffer;
  try {
    blobArrayBuffer = await merged.arrayBuffer();
  } catch (err) {
    console.error('[clip-recorder] Failed to convert Blob to ArrayBuffer:', err.message);
    return;
  }

  // Forward to main process — credentials are never included here
  try {
    const response = await window.bridge.saveAndUploadClip({ blobArrayBuffer, metadata });
    if (response?.result?.evidenceUrl) {
      console.debug('[clip-recorder] Clip uploaded:', response.result.evidenceUrl);
    } else if (response?.result?.uploadStatus === 'upload_failed') {
      console.warn('[clip-recorder] Clip upload failed:', response.result.reasonCode);
    }
  } catch (err) {
    console.error('[clip-recorder] IPC saveAndUploadClip error:', err.message);
  } finally {
    // Release the lifecycle lock only after the upload resolves so no second
    // video clip can start while this one is still encoding or uploading.
    isVideoClipInFlight = false;
  }
}

// ---------------------------------------------------------------------------
// T022: Force-finalize on exam end
// ---------------------------------------------------------------------------

/**
 * Immediately flush any in-progress capture before exam teardown.
 * Safe to call even when no capture is active (no-op in that case).
 *
 * @returns {Promise<void>}
 */
async function forceFinalize() {
  const promises = [];

  // Case 1: still collecting post-violation chunks — flush them now.
  if (isCapturingPost) {
    if (postChunks.length > 0) {
      console.debug('[clip-recorder] forceFinalize: flushing partial video clip.');
      _videoUploadPromise = finalizeClip(preViolationChunksSnapshot.slice());
      promises.push(_videoUploadPromise);
    } else {
      console.warn('[clip-recorder] forceFinalize: video lock held but no post-chunks — clearing lock.');
      _resetCaptureState();
    }
  }

  // Case 2: recording is done but upload is still running (isCapturingPost
  // was already reset inside finalizeClip before its first await). Await the
  // saved Promise so teardown doesn't race the CDN PUT.
  if (isVideoClipInFlight) {
    console.debug('[clip-recorder] forceFinalize: video upload in progress — waiting.');
    promises.push(_videoUploadPromise);
  }

  // Same two cases for audio.
  if (isCapturingAudioPost) {
    if (audioPostChunks.length > 0) {
      console.debug('[clip-recorder] forceFinalize: flushing partial audio clip.');
      _audioUploadPromise = finalizeAudioClip(audioPreSnapshot.slice());
      promises.push(_audioUploadPromise);
    } else {
      console.warn('[clip-recorder] forceFinalize: audio lock held but no post-chunks — clearing lock.');
      _resetAudioCaptureState();
    }
  }

  if (isAudioClipInFlight) {
    console.debug('[clip-recorder] forceFinalize: audio upload in progress — waiting.');
    promises.push(_audioUploadPromise);
  }

  await Promise.all(promises);
}

// ---------------------------------------------------------------------------
// T029: Stop clip recorder
// ---------------------------------------------------------------------------

/**
 * Stop and release all MediaRecorder + webcam stream resources.
 * Reset all module-scope state. Safe to call multiple times.
 */
function stopClipRecorder() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try {
      mediaRecorder.stop();
    } catch (_) {
      // ignore errors during teardown
    }
  }
  if (clipStream) {
    clipStream.getTracks().forEach(t => t.stop());
  }
  mediaRecorder = null;
  clipStream = null;
  _resetCaptureState();
  ringBuffer = [];
  webmInitChunk = null;
  webmHeaderOnlyBlob = null;
  isVideoClipInFlight = false;
  _videoUploadPromise = Promise.resolve();

  // Stop audio recorder
  if (audioRecorder && audioRecorder.state !== 'inactive') {
    try {
      audioRecorder.stop();
    } catch (_) {
      // ignore errors during teardown
    }
  }
  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop());
  }
  audioRecorder = null;
  audioStream = null;
  _resetAudioCaptureState();
  audioRingBuffer = [];
  audioWebmInitChunk = null;
  audioWebmHeaderOnlyBlob = null;
  isAudioClipInFlight = false;
  _audioUploadPromise = Promise.resolve();

  clipConfig = null;
  activeExamSession = null;
  console.debug('[clip-recorder] Stopped and reset.');
}

// ---------------------------------------------------------------------------
// Speech violation handler (audio path)
// ---------------------------------------------------------------------------

/**
 * Trigger audio capture for a speech violation.
 * Mirrors handleViolationEvent but operates on the audio recorder state.
 *
 * @param {object} alert  Alert params from the JSON-RPC notification
 */
function _handleSpeechViolation(alert) {
  if (!audioRecorder || audioRecorder.state === 'inactive') {
    // Audio unavailable — send a zero-blob record so the backend still logs it.
    // Skip if an audio clip is already in-flight (recording or uploading).
    if (!isCapturingAudioPost && !isAudioClipInFlight) {
      _handleWebcamUnavailable(alert);
    }
    return;
  }

  const entry = {
    violationType: alert.code || 'SPEECH_DETECTED',
    confidence: alert.evidence?.confidence ?? 0,
    timestamp: alert.timestamp || new Date().toISOString(),
    description: alert.message || alert.code || 'Speech detected',
  };

  if (isCapturingAudioPost || isAudioClipInFlight) {
    // Lock is held (recording OR uploading) — absorb into the current audio clip.
    console.debug(`[clip-recorder] Speech violation absorbed into active audio clip: ${alert.code}`);
    audioViolations.push(entry);
    return;
  }

  // Acquire full lifecycle lock and trigger audio capture
  isAudioClipInFlight = true;
  audioPreSnapshot = audioRingBuffer.slice();
  audioRingBuffer = [];
  isCapturingAudioPost = true;
  audioPostChunks = [];
  audioViolations = [entry];
  console.debug(`[clip-recorder] Audio capture triggered by: ${alert.code}`);

  // Fix D (audio): post-snap safety-net timer. Mirrors the video path — ensures
  // finalization even if ondataavailable stops firing for the audio recorder.
  const postMs = (clipConfig?.post_buffer_seconds || 10) * 1000;
  setTimeout(() => {
    if (isCapturingAudioPost) {
      console.warn('[clip-recorder] Audio post-snap timeout safety-net fired — finalizing clip.');
      _audioUploadPromise = finalizeAudioClip(audioPreSnapshot.slice());
    }
  }, postMs + 500);
}

// ---------------------------------------------------------------------------
// Audio clip finalization
// ---------------------------------------------------------------------------

/**
 * Merge audio pre + post buffers into a single Blob and send to main.js for upload.
 * Adds mediaType: 'audio' to the metadata so the backend skips video re-encoding.
 *
 * @param {Array<{blob: Blob, timestamp: string}>} preChunks  Snapshot of audio pre-buffer
 */
async function finalizeAudioClip(preChunks) {
  if (audioPostChunks.length === 0 && preChunks.length === 0) {
    console.warn('[clip-recorder] finalizeAudioClip called with empty buffers — skipping.');
    _resetAudioCaptureState();
    return;
  }

  const localPost = audioPostChunks.slice();
  const localViolations = audioViolations.slice();
  const localSession = activeExamSession;

  _resetAudioCaptureState();

  const mimeType = (audioRecorder?.mimeType) || 'audio/webm';
  // Same header-only strategy as video: use header-without-cluster so
  // setpts=PTS-STARTPTS anchors at the first ring-buffer PTS, not T=0.
  const firstAudioBlob = preChunks[0]?.blob ?? localPost[0]?.blob ?? null;
  const _audioInitBlobForHeader = audioWebmHeaderOnlyBlob ?? audioWebmInitChunk;

  // Guard: if no WebM init segment was ever received (e.g. safety-net fired
  // before the first ondataavailable chunk), the merged blob will have no
  // codec headers and ffmpeg will fail with ENCODE_FAILED.  Send a
  // clip_unavailable record instead so the violation is still logged.
  if (!_audioInitBlobForHeader) {
    console.warn('[clip-recorder] finalizeAudioClip: no audio init segment available — sending unavailable record.');
    const fallbackAlert = localViolations[0]
      ? { code: localViolations[0].violationType, evidence: { confidence: localViolations[0].confidence }, timestamp: localViolations[0].timestamp, message: localViolations[0].description }
      : { code: 'SPEECH_DETECTED', evidence: { confidence: 0 }, timestamp: new Date().toISOString(), message: 'Speech detected' };
    _handleWebcamUnavailable(fallbackAlert);
    isAudioClipInFlight = false;
    return;
  }

  const needsAudioInit = _audioInitBlobForHeader && firstAudioBlob !== audioWebmInitChunk;
  const allBlobs = [
    ...(needsAudioInit ? [_audioInitBlobForHeader] : []),
    ...preChunks.map(c => c.blob),
    ...localPost.map(c => c.blob),
  ];

  if (allBlobs.length === 0) {
    console.warn('[clip-recorder] No audio blobs to merge.');
    return;
  }

  const merged = new Blob(allBlobs, { type: mimeType });

  const captureWindowStart =
    preChunks[0]?.timestamp ?? localPost[0]?.timestamp ?? new Date().toISOString();
  const captureWindowEnd =
    localPost[localPost.length - 1]?.timestamp ?? new Date().toISOString();

  const firstViolation = localViolations[0] || {};
  const description =
    localViolations.length === 1
      ? firstViolation.description || firstViolation.violationType || 'Speech detected'
      : `${firstViolation.description || firstViolation.violationType} + ${localViolations.length - 1} additional violation(s) in clip`;

  const metadata = {
    studentId: String(localSession?.studentId || localSession?.attemptId || 'unknown'),
    examAttemptId: String(localSession?.attemptId || ''),
    sessionId: String(localSession?.attemptId || ''),
    captureWindowStart,
    captureWindowEnd,
    primaryViolationType: firstViolation.violationType || 'SPEECH_DETECTED',
    primaryConfidence: firstViolation.confidence ?? 0,
    description,
    allViolations: localViolations.map(v => ({
      violationType: v.violationType,
      confidence: v.confidence,
      timestamp: v.timestamp,
      description: v.description,
    })),
    mediaType: 'audio',
  };

  let blobArrayBuffer;
  try {
    blobArrayBuffer = await merged.arrayBuffer();
  } catch (err) {
    console.error('[clip-recorder] Failed to convert audio Blob to ArrayBuffer:', err.message);
    return;
  }

  try {
    const response = await window.bridge.saveAndUploadClip({ blobArrayBuffer, metadata });
    if (response?.result?.evidenceUrl) {
      console.debug('[clip-recorder] Audio clip uploaded:', response.result.evidenceUrl);
    } else if (response?.result?.uploadStatus === 'upload_failed') {
      console.warn('[clip-recorder] Audio clip upload failed:', response.result.reasonCode);
    }
  } catch (err) {
    console.error('[clip-recorder] IPC saveAndUploadClip (audio) error:', err.message);
  } finally {
    // Release the audio lifecycle lock only after the upload resolves.
    isAudioClipInFlight = false;
  }
}

// ---------------------------------------------------------------------------
// T026: Webcam unavailable path
// ---------------------------------------------------------------------------

/**
 * When MediaRecorder is not available, send a clip_unavailable record to the
 * backend so the violation is still logged.
 *
 * @param {object} alert  The original alert params
 */
function _handleWebcamUnavailable(alert) {
  if (!window.bridge?.saveAndUploadClip) return;

  const session = activeExamSession;
  const minimalMetadata = {
    studentId: String(session?.studentId || session?.attemptId || 'unknown'),
    examAttemptId: String(session?.attemptId || ''),
    sessionId: String(session?.attemptId || ''),
    captureWindowStart: alert.timestamp || new Date().toISOString(),
    captureWindowEnd: alert.timestamp || new Date().toISOString(),
    primaryViolationType: alert.code || 'UNKNOWN',
    primaryConfidence: alert.evidence?.confidence ?? 0,
    description: alert.message || alert.code || 'Violation detected',
    allViolations: [{
      violationType: alert.code || 'UNKNOWN',
      confidence: alert.evidence?.confidence ?? 0,
      timestamp: alert.timestamp || new Date().toISOString(),
      description: alert.message || alert.code || 'Violation detected',
    }],
    uploadStatus: 'clip_unavailable',
    reasonCode: 'WEBCAM_UNAVAILABLE',
  };

  console.warn('[clip-recorder] Webcam unavailable — sending clip_unavailable event.');
  // Fire-and-forget: inform backend even without actual footage
  window.bridge.saveAndUploadClip({ blobArrayBuffer: null, metadata: minimalMetadata })
    .catch(() => {});
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function _resetCaptureState() {
  isCapturingPost = false;
  clipViolations = [];
  postChunks = [];
  preViolationChunksSnapshot = [];
  // NOTE: isVideoClipInFlight is intentionally NOT reset here.
  // It stays true until the IPC upload resolves in finalizeClip's finally block.
}

function _resetAudioCaptureState() {
  isCapturingAudioPost = false;
  audioViolations = [];
  audioPostChunks = [];
  audioPreSnapshot = [];
  // NOTE: isAudioClipInFlight is intentionally NOT reset here.
  // It stays true until the IPC upload resolves in finalizeAudioClip's finally block.
}

// ---------------------------------------------------------------------------
// Exports — exposed as window.ClipRecorder (no require(); nodeIntegration is false)
// ---------------------------------------------------------------------------

window.ClipRecorder = {
  initClipRecorder,
  handleViolationEvent,
  forceFinalize,
  stopClipRecorder,
};
