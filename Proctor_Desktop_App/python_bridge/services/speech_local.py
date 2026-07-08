# python_bridge/services/speech_local.py

import datetime
import threading
import time
from typing import Dict, Any, List, Optional

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_base import AIService


# ──────────────────────────────────────────────
# Configuration — tweak these without touching logic
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK = 512
# Human voice frequency range: 80 Hz (lowest male fundamental) to 3400 Hz
# (upper intelligibility limit used in telephony). Frequencies outside this
# band — keyboard clicks, fan noise, low-frequency rumble, high-pitched beeps
# — are attenuated before VAD so only genuine speech triggers detection.
VOICE_LOW_HZ  = 80    # Hz — lower edge of human fundamental frequency
VOICE_HIGH_HZ = 3400  # Hz — upper edge of speech intelligibility band
SPEECH_PROB_THRESHOLD = 0.5      # Silero VAD returns [0.0, 1.0] — was incorrectly 5.0
CHEATING_DURATION = 5.0          # cumulative speech seconds required to earn one strike
CHEATING_THRESHOLD = 5           # active strikes within STRIKE_DECAY_WINDOW = cheater
ALLOWED_PAUSE = 1.5              # silence gap that ends a speech segment
MIN_SPEECH_SEGMENT = 1.0         # minimum segment length to accumulate speech debt
VIOLATION_COOLDOWN = 1.0         # avoid duplicate alerts from jittery boundaries
STRIKE_DECAY_WINDOW = 300.0      # seconds — strikes older than this are forgotten


class SpeechDetectionService(AIService):
    """
    Listens to the local microphone using Silero VAD.
    Violations accumulate internally and are flushed on every predict() poll.

    Threading model:
      - sounddevice fires _audio_callback() from its own C-level thread (writes state)
      - predict() is called from the async server thread (reads state)
      - threading.Lock guards all shared mutable state between the two
    """

    def __init__(self, session_id: str, config: Dict[str, Any]):
        super().__init__("speech-detection", session_id, config)

        speech_cfg = (config or {}).get("services", {}).get("speech-detection", {})
        self._energy_threshold = float(speech_cfg.get("threshold", 0.02))
        self._use_energy_fallback = False

        # Pre-compute a 5th-order Butterworth bandpass filter for the human
        # voice range. Using second-order sections (SOS) for numerical stability.
        # Computed once here so _audio_callback stays cheap.
        nyq = SAMPLE_RATE / 2.0
        self._voice_sos = butter(
            N=5,
            Wn=[VOICE_LOW_HZ / nyq, VOICE_HIGH_HZ / nyq],
            btype='bandpass',
            output='sos',
        )

        # ── Load Silero VAD in a background thread to avoid blocking the constructor ──
        self._model_ready = threading.Event()
        self._log("[SpeechDetection] Loading Silero VAD model in background...")
        if not _TORCH_AVAILABLE:
            self._model = None
            self._use_energy_fallback = True
            self._model_ready.set()
            self._log("[SpeechDetection] torch not installed. Using RMS fallback detector.")
        else:
            def _load_model():
                try:
                    model, _ = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False,
                        trust_repo=True
                    )
                    model.eval()
                    self._model = model
                    self._log("[SpeechDetection] Model ready.")
                except Exception as e:
                    self._model = None
                    self._use_energy_fallback = True
                    self._log(f"[SpeechDetection] Silero unavailable ({e}). Using RMS fallback detector.")
                finally:
                    self._model_ready.set()

            threading.Thread(target=_load_model, daemon=True, name="silero-vad-loader").start()

        # ── Stream handle ──
        self._stream: Optional[sd.InputStream] = None

        # ── Shared state (guarded by _lock) ──
        # Written by sounddevice thread, read by async predict()
        self._lock = threading.Lock()
        self._violation_log: List[Dict[str, Any]] = []
        # Timestamps of every strike issued; entries outside STRIKE_DECAY_WINDOW are pruned.
        self._strike_timestamps: List[float] = []
        # Total strikes ever issued (never decrements) — used to compute per-poll delta.
        self._total_strikes_issued: int = 0
        # Value of _total_strikes_issued at the last predict() call.
        self._last_poll_total: int = 0

        # ── Speech timing state ──
        # Only touched inside _audio_callback, so no lock needed
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._last_violation_time: float = 0.0
        # Accumulated speech seconds that have not yet triggered a strike.
        # One strike is issued every CHEATING_DURATION seconds of cumulative speech.
        self._speech_debt: float = 0.0

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def start(self):
        """Open the microphone stream and begin VAD in the background."""
        if self.is_running:
            self._log("[SpeechDetection] Already running.")
            return

        # Block until the background model-loading thread finishes.
        # Timeout of 60 s covers a first-time hub download; on subsequent runs
        # the cached model loads in under a second.
        if not self._model_ready.wait(timeout=60):
            self._log("[SpeechDetection] Model loading timed out — falling back to RMS detector.")
            self._use_energy_fallback = True
            self._model_ready.set()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK,
            callback=self._audio_callback  # sounddevice calls this every ~32ms
        )
        self._stream.start()
        self.is_running = True
        self._log("[SpeechDetection] Microphone stream started.")

    async def stop(self):
        """Stop the microphone stream cleanly."""
        if not self.is_running:
            return

        self.is_running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._log("[SpeechDetection] Microphone stream stopped.")

    # ──────────────────────────────────────────────
    # Poll endpoint — called by frontend every ~2 seconds
    # ──────────────────────────────────────────────

    async def predict(self, frame: str) -> Dict[str, Any]:
        """
        Flush and return all violations recorded since the last poll.
        `frame` is intentionally unused — audio is captured locally via sounddevice.
        """
        now = time.time()
        with self._lock:
            flushed = self._violation_log.copy()
            self._violation_log.clear()
            # new_strikes: exact count of strikes issued since the last poll (never
            # affected by decay so a strike is never silently lost from the frontend).
            new_strikes = self._total_strikes_issued - self._last_poll_total
            self._last_poll_total = self._total_strikes_issued
            # active_count: strikes within the rolling decay window (drives is_cheater).
            active_count = self._active_strike_count(now)
            is_cheater = active_count >= CHEATING_THRESHOLD

        return self.create_detection_event(
            confidence=1.0,
            payload={
                "new_violations": flushed,
                "violation_count": len(flushed),
                "new_strikes": new_strikes,      # strikes issued since last poll
                "total_strikes": active_count,   # active strikes within decay window
                "is_cheater": is_cheater,
            }
        )

    # ──────────────────────────────────────────────
    # Audio callback — runs on sounddevice's internal C thread
    # ──────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags
    ):
        """Called every ~32ms. Must be fast and non-blocking."""
        if not self.is_running:
            return

        # 1. Prepare audio array and apply human-voice bandpass filter (80–3400 Hz).
        #    This removes low-frequency rumble, fan/HVAC noise, keyboard clicks,
        #    and high-frequency interference before VAD so only genuine speech
        #    in the human fundamental + formant range is evaluated.
        audio = indata[:, 0]
        try:
            audio = sosfilt(self._voice_sos, audio).astype(np.float32)
        except Exception:
            pass  # if filter fails, continue with unfiltered audio

        # 2. VAD inference
        try:
            if self._use_energy_fallback:
                # Scale RMS energy to a pseudo-probability in [0, 1].
                rms = float(np.sqrt(np.mean(np.square(audio))))
                speech_prob = min(1.0, max(0.0, rms / max(self._energy_threshold * 2.0, 1e-6)))
            else:
                tensor = torch.from_numpy(audio.copy())
                speech_prob = self._model(tensor, SAMPLE_RATE).item()
        except Exception as e:
            self._log(f"[SpeechDetection] VAD error: {e}")
            return

        current_time = time.time()

        # 3. State machine
        if speech_prob > SPEECH_PROB_THRESHOLD:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = current_time
                self._log("[SpeechDetection] Speech started.")

            self._last_speech_time = current_time

        else:
            if self._is_speaking:
                if current_time - self._last_speech_time > ALLOWED_PAUSE:
                    segment_duration = max(0.0, self._last_speech_time - self._speech_start_time)

                    if segment_duration >= MIN_SPEECH_SEGMENT:
                        # Accumulate speech debt. Issue one strike per CHEATING_DURATION
                        # seconds of cumulative speech, carrying the remainder forward.
                        self._speech_debt += segment_duration
                        while (self._speech_debt >= CHEATING_DURATION
                               and current_time - self._last_violation_time >= VIOLATION_COOLDOWN):
                            self._speech_debt -= CHEATING_DURATION
                            self._record_violation(segment_duration)
                            self._last_violation_time = current_time

                    with self._lock:
                        active = self._active_strike_count(current_time)
                    self._log(
                        f"\n[SpeechDetection] Speech ended "
                        f"(segment={segment_duration:.2f}s, debt={self._speech_debt:.2f}s). "
                        f"Active strikes: {active}/{CHEATING_THRESHOLD}"
                    )
                    self._is_speaking = False

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _record_violation(self, duration: float):
        """Append a violation to the log. Called from sounddevice thread → use lock."""
        with self._lock:
            now = time.time()
            self._strike_timestamps.append(now)
            self._total_strikes_issued += 1
            active = self._active_strike_count(now)
            is_cheater = active >= CHEATING_THRESHOLD

            self._violation_log.append({
                "type": "speech_violation",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "duration_seconds": round(duration, 2),
                "strike_number": self._total_strikes_issued,
                "active_strikes": active,
                "cheater_flagged": is_cheater,
            })

        flag = " — CHEATER FLAGGED" if is_cheater else ""
        self._log(
            f"[SpeechDetection] Strike #{self._total_strikes_issued} "
            f"(active: {active}/{CHEATING_THRESHOLD} within {STRIKE_DECAY_WINDOW:.0f}s){flag}"
        )

    def _active_strike_count(self, now: float) -> int:
        """Count strikes within the rolling STRIKE_DECAY_WINDOW; prunes expired entries.
        Must be called with self._lock held.
        """
        cutoff = now - STRIKE_DECAY_WINDOW
        self._strike_timestamps = [t for t in self._strike_timestamps if t >= cutoff]
        return len(self._strike_timestamps)

    def _log(self, message: str):
        """Write service logs to stderr to avoid corrupting JSON-RPC stdout."""
        print(message, file=sys.stderr, flush=True)