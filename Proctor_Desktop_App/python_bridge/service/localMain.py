from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    from .Gaze import GazeDetector
except ImportError:  # pragma: no cover
    from Gaze import GazeDetector

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
        """
        Safety lock in case the backend sends two batches at the same time
        to the same instance.
        """
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


def process_frames_batch(
    session_id: str,
    frames_b64: list[str],
    fps: int = 30,
    is_writing: bool = False,   # kept for API compatibility; has no effect
) -> list[dict]:
    """Process a batch of base64-encoded frames for gaze detection.

    Parameters
    ----------
    session_id  : Unique ID for the exam attempt.
    frames_b64  : List of base64-encoded JPEG/PNG frames.
    fps         : Frame rate used to assign per-frame timestamps.
    is_writing  : Ignored — writing mode is not supported in this layer.
    """
    frame_interval = 1.0 / fps
    session = manager.get_or_create(session_id)

    with manager.lock:
        start = time.time()
        results = []
        batch_start_time = time.time()

        for index, frame_b64 in enumerate(frames_b64):
            frame_timestamp = batch_start_time + (index * frame_interval)

            try:
                img_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as exc:
                logger.warning(f"[process_frames_batch] Frame {index} decode error: {exc}")
                frame = None

            if frame is None:
                evt = session.process_gaze_frame(None, timestamp=frame_timestamp)
            else:
                t0 = time.time()
                evt = session.process_gaze_frame(frame, timestamp=frame_timestamp)
                t1 = time.time()
                diag = evt.get("gaze_diagnostics", {})
                print(
                    f"  Frame {index:02d} | "
                    f"mediapipe={t1 - t0:.3f}s | "
                    f"face={diag.get('face_present', '?')} | "
                    f"h={diag.get('h_ratio', 0.0):.3f} "
                    f"pitch={diag.get('pitch_deg', 0.0):.1f}°"
                )

            results.append(evt)

        elapsed = time.time() - start
        print(
            f"[GazeSession] {len(frames_b64)} frames processed locally "
            f"in {elapsed:.2f}s  (~{elapsed / len(frames_b64) * 1000:.1f}ms/frame)"
        )

    return results


# ─────────────────────────────────────────────
# GazeSession  — mirrors src/Server/localMain.py exactly
# ─────────────────────────────────────────────
class GazeSession:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._detector  = GazeDetector()

        # ── horizontal gaze calibration ───────────────────────────────
        self._gaze_history_x    = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self._envelope_x        = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_center_x: float | None = None
        self._baseline_std_x:    float | None = None

        # ── head-pitch calibration ─────────────────────────────────────
        self._pitch_envelope = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_pitch: float | None = None

        self._initialized     = False
        self._attention_state = "INITIALIZING"
        self._away_start_time: float | None = None
        self._event_id        = 0

        logger.info(f"[GazeSession] Initialized: {session_id}")

    # ── recalibration ─────────────────────────────────────────────────
    def recalibrate(self) -> None:
        """Reset all calibration state so the next ENVELOPE_WINDOW face-visible
        frames re-run calibration from scratch."""
        self._gaze_history_x.clear()
        self._envelope_x.clear()
        self._baseline_center_x = None
        self._baseline_std_x    = None
        self._pitch_envelope.clear()
        self._baseline_pitch    = None
        self._initialized       = False
        self._attention_state   = "INITIALIZING"
        self._away_start_time   = None
        self._event_id          = 0
        logger.info(f"[GazeSession] Recalibration triggered: {self.session_id}")

    # ── main entry ────────────────────────────────────────────────────
    def process_gaze_frame(
        self,
        frame,
        timestamp: float | None = None,
    ) -> dict:
        """Process a single BGR frame (or ``None`` for a decode failure).

        Returns a self-contained event dict with gaze diagnostics embedded.
        """
        current_time = timestamp if timestamp is not None else time.time()

        # ── decode failure / null frame ───────────────────────────────
        if frame is None:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event(
                "NO_FACE", 1.0, "NO_FACE",
                timestamp=current_time,
                face_present=False,
                h_ratio=0.0, avg_x=0.0,
                pitch_deg=0.0,
                h_deviation=0.0, tolerance_x=0.0,
                looking_down=False,
                direction="NO_FACE",
                baseline_center_x=self._baseline_center_x,
                baseline_std_x=self._baseline_std_x,
                baseline_pitch=self._baseline_pitch,
                active_threshold=None,
            )

        h_ratio, _v_ratio, face_present = self._detector.get_gaze_ratio(frame)
        # Cap pitch to ±25° so extreme head-tilt never inflates the down score.
        pitch_deg = float(min(self._detector.last_pitch_deg, 25.0))

        # ── no face ───────────────────────────────────────────────────
        if not face_present:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event(
                "NO_FACE", 1.0, "NO_FACE",
                timestamp=current_time,
                face_present=False,
                h_ratio=h_ratio, avg_x=h_ratio,
                pitch_deg=pitch_deg,
                h_deviation=0.0, tolerance_x=0.0,
                looking_down=False,
                direction="NO_FACE",
                baseline_center_x=self._baseline_center_x,
                baseline_std_x=self._baseline_std_x,
                baseline_pitch=self._baseline_pitch,
                active_threshold=None,
            )

        # ── calibration — face must be visible to count ────────────────
        if not self._initialized:
            self._envelope_x.append(h_ratio)
            if pitch_deg != 0.0:
                self._pitch_envelope.append(pitch_deg)

            n = len(self._envelope_x)
            if n >= ENVELOPE_WINDOW:
                self._initialized = True
                logger.info(f"[GazeSession] Calibration complete — {ENVELOPE_WINDOW} frames")
            elif n % 15 == 0 and n > 0:
                logger.info(f"[GazeSession] Calibrating... {n}/{ENVELOPE_WINDOW}")

            return self._build_event(
                "INITIALIZING", 0.0, "CALIBRATING",
                timestamp=current_time,
                face_present=True,
                h_ratio=h_ratio, avg_x=h_ratio,
                pitch_deg=pitch_deg,
                h_deviation=0.0, tolerance_x=0.0,
                looking_down=False,
                direction="CALIBRATING",
                baseline_center_x=self._baseline_center_x,
                baseline_std_x=self._baseline_std_x,
                baseline_pitch=self._baseline_pitch,
                active_threshold=None,
            )

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
            timestamp=current_time,
            face_present=True,
            h_ratio=round(h_ratio, 4),
            avg_x=round(avg_x, 4),
            pitch_deg=round(pitch_deg, 2),
            h_deviation=round(h_deviation, 4),
            tolerance_x=round(tolerance_x, 4),
            looking_down=looking_down,
            direction=direction,
            baseline_center_x=round(self._baseline_center_x, 4),
            baseline_std_x=round(self._baseline_std_x, 4),
            baseline_pitch=round(self._baseline_pitch, 2),
            active_threshold=round(active_threshold, 2),
        )

    # ── event builder ─────────────────────────────────────────────────
    def _build_event(
        self,
        state: str,
        probability: float,
        evidence: str,
        *,
        timestamp: float | None = None,
        face_present: bool = False,
        h_ratio: float = 0.0,
        avg_x: float = 0.0,
        pitch_deg: float = 0.0,
        h_deviation: float = 0.0,
        tolerance_x: float = 0.0,
        looking_down: bool = False,
        direction: str = "UNKNOWN",
        baseline_center_x: float | None = None,
        baseline_std_x: float | None = None,
        baseline_pitch: float | None = None,
        active_threshold: float | None = None,
    ) -> dict:
        """Build a self-contained event dict with all gaze diagnostics embedded.

        Parameters
        ----------
        state            : Attention state (``"ON_SCREEN"``, ``"AWAY_SHORT"``,
                           ``"AWAY_LONG"``, ``"NO_FACE"``, ``"INITIALIZING"``).
        probability      : Suspicion probability in [0, 1].
        evidence         : Human-readable direction / condition label.
        timestamp        : Unix epoch float; defaults to now.
        face_present     : Whether a face was detected in this frame.
        h_ratio          : Raw horizontal iris ratio from GazeDetector.
        avg_x            : Smoothed horizontal ratio after the moving average.
        pitch_deg        : Head pitch in degrees from FaceLandmarker.
        h_deviation      : ``avg_x - baseline_center_x``.
        tolerance_x      : Horizontal tolerance band (dynamic).
        looking_down     : True when pitch exceeds the active threshold.
        direction        : ``"CENTER"`` | ``"LEFT"`` | ``"RIGHT"`` | ``"DOWN"``
                           | ``"BOTH"`` | ``"NO_FACE"`` | ``"CALIBRATING"``.
        baseline_center_x: Calibrated horizontal neutral position.
        baseline_std_x   : Standard deviation of calibration samples.
        baseline_pitch   : Calibrated neutral head-pitch (degrees).
        active_threshold : ``baseline_pitch + PITCH_DOWN_THRESHOLD``.
        """
        self._event_id += 1
        ts = (
            datetime.fromtimestamp(timestamp).isoformat()
            if timestamp is not None else datetime.now().isoformat()
        )
        return {
            "id":               self._event_id,
            "timestamp":        ts,
            "attention_state":  state,
            "probability":      round(probability, 4),
            "evidence":         evidence,
            "gaze_diagnostics": {
                "face_present":       face_present,
                "direction":          direction,
                "h_ratio":            h_ratio,
                "avg_x":              avg_x,
                "h_deviation":        h_deviation,
                "tolerance_x":        tolerance_x,
                "pitch_deg":          pitch_deg,
                "looking_down":       looking_down,
                "baseline_center_x":  baseline_center_x,
                "baseline_std_x":     baseline_std_x,
                "baseline_pitch":     baseline_pitch,
                "active_threshold":   active_threshold,
            },
        }
