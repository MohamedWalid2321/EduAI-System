from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from Gaze import GazeDetector

# ─────────────────────────────────────────────
# logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── calibration ───────────────────────────────
SMOOTHING_BUFFER_SIZE = 5
ENVELOPE_WINDOW       = 90

# ── away timers ───────────────────────────────
AWAY_SHORT_SECONDS = 3.0
AWAY_LONG_SECONDS  = 5.0

# ── pitch threshold ───────────────────────────
# Offset from each person's calibrated neutral pitch.
# pitch > baseline + PITCH_DOWN_THRESHOLD → looking DOWN.
PITCH_DOWN_THRESHOLD = 18.0


# ─────────────────────────────────────────────
# Session manager
# ─────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        self.session: GazeSession | None = None
        self.session_lock: threading.Lock = threading.Lock()

    def get_or_create(self, session_id: str) -> GazeSession:
        if self.session is None:
            self.session = GazeSession(session_id)
            logger.info(f"[SessionManager] Session created: {session_id}")
        return self.session

    def clear(self, session_id: str) -> None:
        self.session = None
        logger.info(f"[SessionManager] Session cleared: {session_id}")

    @property
    def lock(self) -> threading.Lock:
        return self.session_lock


manager = SessionManager()


# ─────────────────────────────────────────────
# GazeSession  — same logic as main.py
# ─────────────────────────────────────────────
class GazeSession:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._detector  = GazeDetector()

        # ── horizontal gaze calibration ───────────────────────────────
        self._gaze_history_x    = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self._envelope_x        = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_center_x = None
        self._baseline_std_x    = None

        # ── head-pitch calibration ─────────────────────────────────────
        self._pitch_envelope = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_pitch = None

        self._initialized     = False
        self._attention_state = "INITIALIZING"
        self._away_start_time = None
        self._event_id        = 0

        logger.info(
            f"[GazeSession] Initialized: {session_id}"
        )

    def set_question_type(self, is_writing: bool) -> None:
        """No-op — writing mode is disabled; backend always sends False."""
        pass

    # ── main entry — call once per camera frame ────────────────────────
    def process_gaze_frame(self, frame) -> dict:
        current_time = datetime.now().timestamp()

        h_ratio, v_ratio, face_present = self._detector.get_gaze_ratio(frame)
        pitch_deg = self._detector.last_pitch_deg

        # ── no face ───────────────────────────────────────────────────
        if not face_present:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event("NO_FACE", 1.0, "NO_FACE")

        # ── calibration ───────────────────────────────────────────────
        if not self._initialized:
            self._envelope_x.append(h_ratio)
            if pitch_deg != 0.0:
                self._pitch_envelope.append(pitch_deg)

            if len(self._envelope_x) >= ENVELOPE_WINDOW:
                self._initialized = True
                logger.info(f"[GazeSession] Calibration complete — {ENVELOPE_WINDOW} frames")
            elif len(self._envelope_x) % 15 == 0:
                logger.info(f"[GazeSession] Calibrating... {len(self._envelope_x)}/{ENVELOPE_WINDOW}")

            return self._build_event("INITIALIZING", 0.0, "CALIBRATING")

        # ── compute baselines once ─────────────────────────────────────
        if self._baseline_center_x is None:
            self._baseline_center_x = float(np.median(self._envelope_x))
            self._baseline_std_x    = float(np.std(self._envelope_x))
            logger.info(
                f"[GazeSession] H baseline — center={self._baseline_center_x:.3f} "
                f"std={self._baseline_std_x:.3f}"
            )

        if self._baseline_pitch is None:
            if len(self._pitch_envelope) >= max(10, ENVELOPE_WINDOW // 3):
                self._baseline_pitch = float(np.max(self._pitch_envelope))
                logger.info(
                    f"[GazeSession] Pitch baseline={self._baseline_pitch:.1f}° | "
                    f"down threshold={self._baseline_pitch + PITCH_DOWN_THRESHOLD:.1f}°"
                )
            else:
                self._baseline_pitch = 0.0
                logger.warning("[GazeSession] Pitch baseline: insufficient samples, defaulting to 0°")

        # ── smooth ────────────────────────────────────────────────────
        self._gaze_history_x.append(h_ratio)
        avg_x = sum(self._gaze_history_x) / len(self._gaze_history_x)

        # ── horizontal check ──────────────────────────────────────────
        tolerance_x = max(0.10, self._baseline_std_x * 4.0)
        h_deviation = avg_x - self._baseline_center_x
        h_outside   = abs(h_deviation) > tolerance_x

        # ── down check (pitch threshold) ──────────────────────────────
        active_threshold = self._baseline_pitch + PITCH_DOWN_THRESHOLD
        looking_down     = pitch_deg > active_threshold

        inside_safe_zone = (not h_outside) and (not looking_down)

        # ── direction ─────────────────────────────────────────────────
        if h_outside and looking_down:
            direction = "BOTH"
        elif h_outside:
            direction = "LEFT" if h_deviation < 0 else "RIGHT"
        elif looking_down:
            direction = "DOWN"
        else:
            direction = "CENTER"

        # ── away timers ───────────────────────────────────────────────
        if inside_safe_zone:
            self._attention_state = "ON_SCREEN"
            self._away_start_time = None
        else:
            if self._away_start_time is None:
                self._away_start_time = current_time

            elapsed = current_time - self._away_start_time

            if elapsed >= AWAY_LONG_SECONDS:
                self._attention_state = "AWAY_LONG"
            elif elapsed >= AWAY_SHORT_SECONDS:
                self._attention_state = "AWAY_SHORT"
            else:
                self._attention_state = "ON_SCREEN"

        if self._attention_state == "AWAY_SHORT":
            logger.warning(f"[GazeSession] AWAY_SHORT | dir={direction}")
        elif self._attention_state == "AWAY_LONG":
            logger.warning(f"[GazeSession] AWAY_LONG  | dir={direction}")

        probability = {"ON_SCREEN": 0.0, "AWAY_SHORT": 0.5}.get(
            self._attention_state, 1.0
        )

        return self._build_event(
            self._attention_state, probability, direction,
            h_dev=h_deviation, tol_x=tolerance_x,
            pitch=pitch_deg, base_pitch=self._baseline_pitch,
            threshold=active_threshold,
        )

    def _build_event(self, state, probability, evidence, **extra):
        self._event_id += 1
        evt = {
            "id":              self._event_id,
            "timestamp":       datetime.now().isoformat(),
            "attention_state": state,
            "probability":     round(probability, 4),
            "evidence":        evidence,
        }
        evt.update(extra)
        return evt