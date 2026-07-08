"""
speech_localMain.py
────────────────────────────────────────────────────────────────────────────
Stateless-per-request but session-aware speech processing using Silero VAD.

Responsibilities
────────────────
• Load the Silero VAD model once at module import time.
• Maintain a lightweight SpeechSession dict per session_id via SpeechSessionManager.
• Expose `process_audio_chunk(session_id, b64_audio, sample_rate)` as the single
  entry-point called by the FastAPI router via run_in_executor.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Configuration & Thresholds
# ---------------------------------------------------------------------------

SPEECH_PROB_THRESHOLD: float = 0.5   # Silero confidence required to count as speech
CHEATING_DURATION:     float = 5.0   # Continuous seconds of speech → SPEECH_VIOLATION
ALLOWED_PAUSE:         float = 1.5   # Seconds of silence before the timer resets

# Silero VAD requires EXACTLY 512 samples per forward pass at 16 kHz (256 at 8 kHz).
# Incoming chunks are longer (e.g. 1–2 s = 16000–32000 samples), so we split them
# into 512-sample windows and take the MAX probability as the chunk score.
VAD_WINDOW_SIZE: int = 512

# ---------------------------------------------------------------------------
# 2. Load Silero VAD model (once, at import time)
# ---------------------------------------------------------------------------

logger.info("Loading Silero VAD model…")
_vad_model, _vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
)
_vad_model.eval()
logger.info("Silero VAD model loaded successfully.")

# ---------------------------------------------------------------------------
# 3. Session state
# ---------------------------------------------------------------------------

class SpeechSession(TypedDict):
    """Per-session mutable state tracked across HTTP requests."""
    is_speaking:        bool
    speech_start_time:  float   # epoch seconds when current speech segment began
    last_speech_time:   float   # epoch seconds of the last chunk scored as speech


class SpeechSessionManager:
    """Thread-safe store for SpeechSession objects keyed by session_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, SpeechSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SpeechSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SpeechSession(
                    is_speaking=False,
                    speech_start_time=0.0,
                    last_speech_time=0.0,
                )
                logger.debug("[SpeechSessionManager] Created session '%s'", session_id)
            return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        with self._lock:
            removed = self._sessions.pop(session_id, None)
            if removed:
                logger.debug("[SpeechSessionManager] Cleared session '%s'", session_id)


# Module-level singleton — imported and used by the router's `clear` endpoint.
manager = SpeechSessionManager()

# ---------------------------------------------------------------------------
# 4. Core processing function
# ---------------------------------------------------------------------------

def process_audio_chunk(
    session_id:  str,
    b64_audio:   str,
    sample_rate: int = 16000,
) -> dict:
    """
    Decode one Base64 audio chunk, run Silero VAD, update session state,
    and return a verdict dictionary compatible with SpeechVerdict.

    Parameters
    ----------
    session_id  : Identifies which exam session this chunk belongs to.
    b64_audio   : Base64-encoded raw float32 PCM bytes (mono, 16 kHz).
    sample_rate : Must be 16000 for Silero VAD.

    Returns
    -------
    dict with keys:
        timestamp          – ISO-8601 UTC string
        flag               – "SILENCE" | "SPEAKING" | "SPEECH_VIOLATION"
        speech_probability – float [0.0, 1.0]
        speaking_duration  – accumulated continuous speech seconds
        evidence           – human-readable explanation
    """

    # ------------------------------------------------------------------
    # 4a. Decode Base64 → numpy float32 → torch tensor
    # ------------------------------------------------------------------
    try:
        raw_bytes   = base64.b64decode(b64_audio)
        audio_array = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        tensor      = torch.from_numpy(audio_array)
    except Exception as exc:
        raise ValueError(f"Failed to decode audio chunk: {exc}") from exc

    # ------------------------------------------------------------------
    # 4b. Run Silero VAD inference (windowed — Silero requires exactly 512 samples)
    # ------------------------------------------------------------------
    speech_prob: float = _infer_windowed(tensor, sample_rate)

    current_time = time.time()
    session      = manager.get_or_create(session_id)

    # ------------------------------------------------------------------
    # 4c. State-machine: update session and determine flag
    # ------------------------------------------------------------------
    flag              = "SILENCE"
    speaking_duration = 0.0

    if speech_prob > SPEECH_PROB_THRESHOLD:
        # ---- Speech detected in this chunk ----
        if not session["is_speaking"]:
            # Transition: silence → speaking
            session["is_speaking"]       = True
            session["speech_start_time"] = current_time
            logger.debug("[%s] Speech started.", session_id)

        session["last_speech_time"] = current_time
        speaking_duration            = current_time - session["speech_start_time"]

        if speaking_duration >= CHEATING_DURATION:
            flag = "SPEECH_VIOLATION"

            # Reset the start time so the *next* 5-second window begins immediately
            # if the user keeps talking without pausing.
            session["speech_start_time"] = current_time
            speaking_duration            = 0.0  # report 0 after reset so backend knows it fired

            logger.info(
                "[%s] SPEECH_VIOLATION — spoke continuously for %.1f s.",
                session_id, CHEATING_DURATION,
            )
        else:
            flag = "SPEAKING"

    else:
        # ---- Silence detected in this chunk ----
        if session["is_speaking"]:
            time_since_last_word = current_time - session["last_speech_time"]

            if time_since_last_word > ALLOWED_PAUSE:
                # Transition: speaking → silence (pause too long → reset timer)
                session["is_speaking"]       = False
                session["speech_start_time"] = 0.0
                logger.debug(
                    "[%s] Speech stopped after %.1f s silence — timer reset.",
                    session_id, time_since_last_word,
                )
            else:
                # Short gap — still considered speaking; keep the timer running
                speaking_duration = current_time - session["speech_start_time"]
                flag = "SPEAKING"

    # ------------------------------------------------------------------
    # 4d. Build and return the verdict event dict
    # ------------------------------------------------------------------
    evidence = _build_evidence(flag, speech_prob, speaking_duration)

    verdict = _build_event(
        timestamp          = datetime.now(timezone.utc).isoformat(),
        flag               = flag,
        speech_probability = round(speech_prob, 4),
        speaking_duration  = round(speaking_duration, 2),
        evidence           = evidence,
    )

    return verdict


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

def _infer_windowed(tensor: torch.Tensor, sample_rate: int) -> float:
    """
    Split `tensor` into VAD_WINDOW_SIZE-sample windows and run Silero on the
    entire batch in ONE forward pass instead of a per-window loop.

    Silero accepts shape [batch_size, num_samples], so we reshape the chunk
    into [N, 512] and get back N probabilities in a single call — reducing
    a 1.5 s chunk from ~46 sequential passes (~2 s) down to ~1 pass (~40 ms).

    Returns the MAX probability across all windows (if any window contains
    speech we want to know about it).

    Incomplete trailing samples (not divisible by VAD_WINDOW_SIZE) are dropped
    because Silero raises a ValueError on non-512-sample inputs.
    """
    num_windows = len(tensor) // VAD_WINDOW_SIZE
    if num_windows == 0:
        raise ValueError(
            f"Audio chunk too short: {len(tensor)} samples. "
            f"Need at least {VAD_WINDOW_SIZE} samples (32 ms at 16 kHz)."
        )

    # Trim to exact multiple of VAD_WINDOW_SIZE, then reshape → [N, 512]
    trimmed = tensor[: num_windows * VAD_WINDOW_SIZE]
    batch   = trimmed.reshape(num_windows, VAD_WINDOW_SIZE)   # [N, 512]

    with torch.no_grad():
        probs = _vad_model(batch, sample_rate)   # returns tensor of shape [N] or scalar

    # probs may be a scalar (N=1) or a 1-D tensor (N>1) — handle both
    if probs.dim() == 0:
        return probs.item()
    return probs.max().item()


def _build_event(
    timestamp:          str,
    flag:               str,
    speech_probability: float,
    speaking_duration:  float,
    evidence:           str,
) -> dict:
    """
    Construct the standardised verdict dictionary.
    Mirrors the `build_event` pattern used in the gaze detection module.
    """
    return {
        "timestamp":          timestamp,
        "flag":               flag,
        "speech_probability": speech_probability,
        "speaking_duration":  speaking_duration,
        "evidence":           evidence,
    }


def _build_evidence(flag: str, prob: float, duration: float) -> str:
    """Return a human-readable explanation for the current flag."""
    if flag == "SPEECH_VIOLATION":
        return (
            f"Continuous speech exceeded {CHEATING_DURATION}s threshold "
            f"(VAD confidence: {prob:.2f})."
        )
    if flag == "SPEAKING":
        return (
            f"Speech detected (VAD confidence: {prob:.2f}). "
            f"Continuous duration so far: {duration:.1f}s."
        )
    return f"No speech detected (VAD confidence: {prob:.2f})."