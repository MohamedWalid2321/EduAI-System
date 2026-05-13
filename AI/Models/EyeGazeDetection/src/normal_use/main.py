from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum

import cv2
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
SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW = 90

# ── away timers — horizontal ─────────────────
AWAY_SHORT_SECONDS_HORIZONTAL = 3.0
AWAY_LONG_SECONDS_HORIZONTAL = 5.0

# ── away timers — down (writing posture) ──────
# Moderate head tilt, eyes looking down at paper/desk.
# This is normal behaviour during math / written exams.
AWAY_SHORT_SECONDS_DOWN_WRITE = 5.0
AWAY_LONG_SECONDS_DOWN_WRITE = 7.0

# ── away timers — down (lap / extreme) ────────
# Head tilted far down — looking at lap, phone, or notes
# hidden below the desk.  Strict thresholds.
AWAY_SHORT_SECONDS_DOWN_LAP = 3.0
AWAY_LONG_SECONDS_DOWN_LAP = 5.0

# ── pitch thresholds ──────────────────────────
# MediaPipe pitch convention: positive = head tilts down.
# When pitch exceeds the active threshold the head is considered
# "lap-looking" (strict timers). Two values allow different
# sensitivity per question type:
#   writing  — lenient (22°): the head tilts down naturally when writing
#   normal   — stricter (18°): any notable head-down tilt is flagged sooner
PITCH_LAP_THRESHOLD_WRITING = 24.0   # degrees — writing mode (head-down is expected)
PITCH_LAP_THRESHOLD_NORMAL  = 18.0   # degrees — normal mode  (head-down triggers faster)

# ── long-term behavioral analysis ────────────
BEHAVIOR_WINDOW_SECONDS = 60.0
BEHAVIOR_MIN_EVENTS = 10
SUSTAINED_DOWN_MIN_SECONDS = 5.0
SUSTAINED_HORIZONTAL_MIN_SECONDS = 3.0
SUSTAINED_DOWN_SUSPICIOUS_RATIO = 0.30
SUSTAINED_HORIZONTAL_SUSPICIOUS_RATIO = 0.15
NO_FACE_SUSPICIOUS_RATIO = 0.15

# ── writing-pattern threshold ─────────────────
WRITING_PATTERN_AVG_EPISODE_MAX_SECONDS = 5.0

# ── tolerance cap ─────────────────────────────
# Prevents tolerance_y from growing so large that
# eye-only-down signals get swallowed.
TOLERANCE_Y_MAX = 0.08

# ── overlay colours ───────────────────────────
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)


class SuspicionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class BehaviorEvent:
    __slots__ = ("timestamp", "state", "direction")

    def __init__(self, timestamp: float, state: str, direction: str):
        self.timestamp = timestamp
        self.state = state
        self.direction = direction


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


class GazeSession:
    # ── question-type constants ────────────────────────────────────────
    QUESTION_TYPE_NORMAL  = "normal"   # Standard MCQ — looking down is NOT expected
    QUESTION_TYPE_WRITING = "writing"  # Writing MCQ — looking down to write is OK

    def __init__(self, session_id: str, question_type: str = "normal"):
        self.session_id = session_id
        self.detector = GazeDetector()

        # ── MCQ question type ──────────────────────────────────────────
        # "normal"  → person should not look down; DOWN treated with strict timers
        # "writing" → person is expected to look down to write; DOWN is ignored
        self._question_type: str = question_type

        self.gaze_history_x: deque[float] = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.envelope_x: deque[float] = deque(maxlen=ENVELOPE_WINDOW)
        self.baseline_center_x: float | None = None
        self.baseline_std_x: float | None = None

        self.gaze_history_y: deque[float] = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.envelope_y: deque[float] = deque(maxlen=ENVELOPE_WINDOW)
        self.baseline_center_y: float | None = None
        self.baseline_std_y: float | None = None

        # ── head-pitch calibration ─────────────────────────────────────
        # Collects raw pitch angles during the calibration window so we can
        # compute each person's neutral head angle.  All lap-detection
        # thresholds are then applied as offsets from this baseline.
        self.pitch_envelope: deque[float] = deque(maxlen=ENVELOPE_WINDOW)
        self.baseline_pitch: float | None = None   # median neutral pitch (degrees)

        self.initialized: bool = False
        self.attention_state: str = "INITIALIZING"

        self.away_start_time_horizontal: float | None = None
        self.away_start_time_vertical_write: float | None = None
        self.away_start_time_vertical_lap: float | None = None

        self.last_direction: str = "CENTER"

        self.behavior_history: deque[BehaviorEvent] = deque(maxlen=10000)

        self._continuous_down_start: float | None = None
        self._continuous_horizontal_start: float | None = None

        self._total_sustained_down_seconds: float = 0.0
        self._total_sustained_horizontal_seconds: float = 0.0
        self._total_no_face_seconds: float = 0.0
        self._session_start_time: float = time.time()

        self._last_process_time: float | None = None
        self.model_id: int = 0

        logger.info(
            f"[GazeSession] Initialized for session: {session_id} "
            f"| question_type={self._question_type}"
        )

    # ── question-type API ─────────────────────────────────────────────
    @property
    def question_type(self) -> str:
        return self._question_type

    def set_question_type(self, qtype: str) -> None:
        """Switch MCQ mode at runtime.

        Parameters
        ----------
        qtype : "normal" | "writing"
            "normal"  — person should NOT look down (strict timers on DOWN).
            "writing" — person is expected to look down to write (DOWN ignored).
        """
        if qtype not in (self.QUESTION_TYPE_NORMAL, self.QUESTION_TYPE_WRITING):
            raise ValueError(
                f"question_type must be 'normal' or 'writing', got {qtype!r}"
            )
        if qtype != self._question_type:
            logger.info(
                f"[GazeSession] question_type changed: "
                f"{self._question_type!r} → {qtype!r}"
            )
            self._question_type = qtype
            # Reset vertical timers so we don't carry stale state
            self.away_start_time_vertical_write = None
            self.away_start_time_vertical_lap = None

    # ─────────────────────────────────────────
    # Classify gaze — now pitch-aware
    # ─────────────────────────────────────────
    def _classify_gaze(
        self, avg_x: float, avg_y: float, pitch_deg: float,
        pitch_lap_threshold: float = PITCH_LAP_THRESHOLD_WRITING,
    ) -> tuple[str, str, dict]:
        """Classify gaze direction with pitch-aware down categorisation.

        Parameters
        ----------
        pitch_lap_threshold : Active lap-detection threshold in degrees.
            Use PITCH_LAP_THRESHOLD_WRITING for writing mode (lenient)
            or PITCH_LAP_THRESHOLD_NORMAL for normal mode (stricter).

        Returns
        -------
        direction    : "CENTER" | "LEFT_RIGHT" | "DOWN" | "DOWN_LAP"
        axis_outside : "NONE" | "HORIZONTAL" | "VERTICAL" | "BOTH"
        diag         : internal values for overlay

        DOWN     — iris down AND pitch < pitch_lap_threshold  (writing posture)
        DOWN_LAP — iris down AND pitch >= pitch_lap_threshold (looking at lap)
        """
        h_deviation = avg_x - self.baseline_center_x
        v_deviation = avg_y - self.baseline_center_y

        h_abs = abs(h_deviation)
        v_abs = abs(v_deviation)

        tolerance_x = max(0.07, self.baseline_std_x * 3.5)
        # Capped at TOLERANCE_Y_MAX so eye-only-down signals always cross
        tolerance_y = min(TOLERANCE_Y_MAX, max(0.04, self.baseline_std_y * 2.0))

        h_outside = h_abs > tolerance_x

        looking_down = v_deviation > 0
        v_outside = v_abs > tolerance_y and looking_down

        if h_outside and v_outside:
            axis_outside = "BOTH"
        elif h_outside:
            axis_outside = "HORIZONTAL"
        elif v_outside:
            axis_outside = "VERTICAL"
        else:
            axis_outside = "NONE"

        if axis_outside == "NONE":
            direction = "CENTER"
        elif axis_outside == "HORIZONTAL":
            direction = "LEFT_RIGHT"
        elif axis_outside == "VERTICAL":
            # Pitch decides: writing posture or lap-looking?
            if pitch_deg > pitch_lap_threshold:
                direction = "DOWN_LAP"
            else:
                direction = "DOWN"
        else:  # BOTH
            h_norm = h_abs / tolerance_x
            v_norm = v_abs / tolerance_y
            if h_norm >= v_norm:
                direction = "LEFT_RIGHT"
            else:
                if pitch_deg > pitch_lap_threshold:
                    direction = "DOWN_LAP"
                else:
                    direction = "DOWN"

        # Determine pitch zone for overlay
        if pitch_deg > pitch_lap_threshold:
            pitch_zone = "LAP"
        elif pitch_deg < pitch_lap_threshold * 0.9:   # ~90% of threshold = writing zone
            pitch_zone = "WRITING"
        else:
            pitch_zone = "LEVEL"

        diag = {
            "v_deviation": v_deviation,
            "tolerance_y": tolerance_y,
            "looking_down": looking_down,
            "v_outside": v_outside,
            "h_deviation": h_deviation,
            "tolerance_x": tolerance_x,
            "pitch_zone": pitch_zone,
        }

        return direction, axis_outside, diag

    def process_gaze_result(
        self,
        h_ratio: float,
        v_ratio: float,
        face_present: bool,
        pitch_deg: float = 0.0,
        timestamp: float | None = None,
    ) -> tuple[dict, dict]:
        _empty_diag: dict = {
            "raw_v": 0.0,
            "avg_y": 0.0,
            "v_deviation": 0.0,
            "tolerance_y": 0.0,
            "looking_down": False,
            "v_outside": False,
            "baseline_center_y": None,
            "baseline_std_y": None,
            "pitch_deg": 0.0,
            "pitch_zone": "LEVEL",
            "baseline_pitch": self.baseline_pitch,
            "active_pitch_threshold": None,
        }

        current_time = timestamp if timestamp is not None else time.time()

        if self._last_process_time is not None:
            dt = current_time - self._last_process_time
        else:
            dt = 1.0 / 30.0
        self._last_process_time = current_time

        # ── no face ───────────────────────────────────────────────────
        if not face_present:
            if self._continuous_down_start is not None:
                duration = current_time - self._continuous_down_start
                if duration >= SUSTAINED_DOWN_MIN_SECONDS:
                    self._total_sustained_down_seconds += duration
                    logger.info(
                        f"[GazeSession] Sustained DOWN closed (NO_FACE) | "
                        f"duration={duration:.1f}s | "
                        f"total={self._total_sustained_down_seconds:.1f}s"
                    )
            if self._continuous_horizontal_start is not None:
                duration = current_time - self._continuous_horizontal_start
                if duration >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                    self._total_sustained_horizontal_seconds += duration
                    logger.info(
                        f"[GazeSession] Sustained H closed (NO_FACE) | "
                        f"duration={duration:.1f}s | "
                        f"total={self._total_sustained_horizontal_seconds:.1f}s"
                    )

            self.attention_state = "NO_FACE"
            self.away_start_time_horizontal = None
            self.away_start_time_vertical_write = None
            self.away_start_time_vertical_lap = None
            self._continuous_down_start = None
            self._continuous_horizontal_start = None
            self._total_no_face_seconds += dt

            self.behavior_history.append(
                BehaviorEvent(current_time, "NO_FACE", "UNKNOWN")
            )

            suspicion = self._compute_behavioral_suspicion(current_time)
            logger.warning(
                f"[GazeSession] NO_FACE | total_no_face="
                f"{self._total_no_face_seconds:.1f}s | "
                f"suspicion={suspicion['level']} score={suspicion['score']:.3f}"
            )
            return (
                self._build_event(
                    "NO_FACE", 1.0, "NO_FACE", current_time, suspicion
                ),
                _empty_diag,
            )

        # ── calibrating ───────────────────────────────────────────────
        if not self.initialized:
            self.envelope_x.append(h_ratio)
            self.envelope_y.append(v_ratio)
            # Accumulate pitch to calibrate neutral head angle
            if pitch_deg != 0.0:
                self.pitch_envelope.append(pitch_deg)
            progress = len(self.envelope_x)

            if (
                len(self.envelope_x) >= ENVELOPE_WINDOW
                and len(self.envelope_y) >= ENVELOPE_WINDOW
            ):
                self.initialized = True
                logger.info(
                    f"[GazeSession] Calibration complete — "
                    f"{ENVELOPE_WINDOW} frames"
                )
            elif progress % 15 == 0:
                logger.info(
                    f"[GazeSession] Calibrating... {progress}/{ENVELOPE_WINDOW}"
                )

            self.behavior_history.append(
                BehaviorEvent(current_time, "INITIALIZING", "CENTER")
            )

            suspicion = self._compute_behavioral_suspicion(current_time)

            cal_diag = {
                "raw_v": v_ratio,
                "avg_y": v_ratio,
                "v_deviation": 0.0,
                "tolerance_y": 0.0,
                "looking_down": False,
                "v_outside": False,
                "baseline_center_y": None,
                "baseline_std_y": None,
                "pitch_deg": pitch_deg,
                "pitch_zone": "LEVEL",
                "baseline_pitch": self.baseline_pitch,
                "active_pitch_threshold": None,
            }
            return (
                self._build_event(
                    "INITIALIZING", 0.0, "CALIBRATING", current_time, suspicion
                ),
                cal_diag,
            )

        # ── compute baselines (once) ──────────────────────────────────
        if self.baseline_center_x is None:
            self.baseline_center_x = float(np.median(self.envelope_x))
            self.baseline_std_x = float(np.std(self.envelope_x))
            logger.info(
                f"[GazeSession] H baseline — center="
                f"{self.baseline_center_x:.3f} std={self.baseline_std_x:.3f} "
                f"tolerance={max(0.07, self.baseline_std_x * 3.5):.3f}"
            )

        if self.baseline_center_y is None:
            self.baseline_center_y = float(np.median(self.envelope_y))
            self.baseline_std_y = float(np.std(self.envelope_y))
            tol_y = min(TOLERANCE_Y_MAX, max(0.04, self.baseline_std_y * 2.0))
            logger.info(
                f"[GazeSession] V baseline — center="
                f"{self.baseline_center_y:.3f} std={self.baseline_std_y:.3f} "
                f"tolerance={tol_y:.3f}"
            )

        # ── pitch baseline (once) ─────────────────────────────────────
        if self.baseline_pitch is None:
            if len(self.pitch_envelope) >= max(10, ENVELOPE_WINDOW // 3):
                self.baseline_pitch = float(np.max(self.pitch_envelope))
                logger.info(
                    f"[GazeSession] Pitch baseline — neutral={self.baseline_pitch:.1f}° "
                    f"(from {len(self.pitch_envelope)} frames) | "
                    f"effective normal threshold={self.baseline_pitch + PITCH_LAP_THRESHOLD_NORMAL:.1f}° | "
                    f"effective writing threshold={self.baseline_pitch + PITCH_LAP_THRESHOLD_WRITING:.1f}°"
                )
            else:
                # Fallback: not enough pitch samples — assume neutral = 0°
                self.baseline_pitch = 0.0
                logger.warning(
                    "[GazeSession] Pitch baseline: insufficient samples, defaulting to 0°"
                )

        # ── smooth ────────────────────────────────────────────────────
        self.gaze_history_x.append(h_ratio)
        self.gaze_history_y.append(v_ratio)
        avg_x = sum(self.gaze_history_x) / len(self.gaze_history_x)
        avg_y = sum(self.gaze_history_y) / len(self.gaze_history_y)

        # ── classify (now pitch-aware) ────────────────────────────────
        # Active threshold = person's neutral pitch + mode offset.
        # This makes detection relative to their natural sitting posture.
        _pitch_offset = (
            PITCH_LAP_THRESHOLD_WRITING
            if self._question_type == self.QUESTION_TYPE_WRITING
            else PITCH_LAP_THRESHOLD_NORMAL
        )
        _neutral = self.baseline_pitch if self.baseline_pitch is not None else 0.0
        active_pitch_threshold = _neutral + _pitch_offset
        direction, axis_outside, _diag = self._classify_gaze(
            avg_x, avg_y, pitch_deg, pitch_lap_threshold=active_pitch_threshold
        )

        # ── question-type overrides ───────────────────────────────────
        # "writing" mode: looking down to write is completely expected.
        #   → Treat DOWN (writing posture) as CENTER — no timer, no suspicion.
        # "normal" mode: the pitch-based threshold already distinguishes
        #   DOWN (pitch < PITCH_LAP_THRESHOLD_NORMAL, mild tilt → moderate timers)
        #   from DOWN_LAP (pitch ≥ PITCH_LAP_THRESHOLD_NORMAL → strict timers).
        #   No additional override is needed; the classifier handles it.
        if direction == "DOWN" and self._question_type == self.QUESTION_TYPE_WRITING:
            # Silently ignore the downward gaze — same as looking at screen
            direction = "CENTER"
            axis_outside = "NONE"
            logger.debug(
                "[GazeSession] DOWN suppressed (writing mode)"
            )

        v_diag = {
            "raw_v": v_ratio,
            "avg_y": avg_y,
            "v_deviation": _diag["v_deviation"],
            "tolerance_y": _diag["tolerance_y"],
            "looking_down": _diag["looking_down"],
            "v_outside": _diag["v_outside"],
            "baseline_center_y": self.baseline_center_y,
            "baseline_std_y": self.baseline_std_y,
            "pitch_deg": pitch_deg,
            "pitch_zone": _diag["pitch_zone"],
            "baseline_pitch": self.baseline_pitch,
            "active_pitch_threshold": active_pitch_threshold,
        }

        if direction != self.last_direction:
            logger.info(
                f"[GazeSession] Direction: {self.last_direction} → "
                f"{direction} | axis={axis_outside} | "
                f"avg_x={avg_x:.3f} avg_y={avg_y:.3f} "
                f"pitch={pitch_deg:.1f}° zone={_diag['pitch_zone']}"
            )
            self.last_direction = direction

        self.behavior_history.append(
            BehaviorEvent(current_time, self.attention_state, direction)
        )

        # Both DOWN and DOWN_LAP count as "down" for sustained tracking
        is_down = direction in ("DOWN", "DOWN_LAP")
        inside_safe_zone = axis_outside == "NONE"
        self._update_sustained_trackers(
            direction, current_time, inside_safe_zone, is_down
        )

        # ── away-timer logic ──────────────────────────────────────────
        if inside_safe_zone:
            self.attention_state = "ON_SCREEN"
            self.away_start_time_horizontal = None
            self.away_start_time_vertical_write = None
            self.away_start_time_vertical_lap = None
        else:
            if direction == "LEFT_RIGHT":
                if self.away_start_time_horizontal is None:
                    self.away_start_time_horizontal = current_time
                    logger.debug(
                        f"[GazeSession] H timer started | avg_x={avg_x:.3f}"
                    )
                self.away_start_time_vertical_write = None
                self.away_start_time_vertical_lap = None
                elapsed = current_time - self.away_start_time_horizontal

                if elapsed >= AWAY_LONG_SECONDS_HORIZONTAL:
                    self.attention_state = "AWAY_LONG"
                elif elapsed >= AWAY_SHORT_SECONDS_HORIZONTAL:
                    self.attention_state = "AWAY_SHORT"
                else:
                    self.attention_state = "ON_SCREEN"

            elif direction == "DOWN":
                # Writing posture — lenient timers
                if self.away_start_time_vertical_write is None:
                    self.away_start_time_vertical_write = current_time
                    logger.debug(
                        f"[GazeSession] V-WRITE timer started | "
                        f"avg_y={avg_y:.3f} pitch={pitch_deg:.1f}°"
                    )
                self.away_start_time_horizontal = None
                self.away_start_time_vertical_lap = None
                elapsed = current_time - self.away_start_time_vertical_write

                if elapsed >= AWAY_LONG_SECONDS_DOWN_WRITE:
                    self.attention_state = "AWAY_LONG"
                elif elapsed >= AWAY_SHORT_SECONDS_DOWN_WRITE:
                    self.attention_state = "AWAY_SHORT"
                else:
                    self.attention_state = "ON_SCREEN"

            elif direction == "DOWN_LAP":
                # Lap-looking — strict timers
                if self.away_start_time_vertical_lap is None:
                    self.away_start_time_vertical_lap = current_time
                    logger.debug(
                        f"[GazeSession] V-LAP timer started | "
                        f"avg_y={avg_y:.3f} pitch={pitch_deg:.1f}°"
                    )
                self.away_start_time_horizontal = None
                self.away_start_time_vertical_write = None
                elapsed = current_time - self.away_start_time_vertical_lap

                if elapsed >= AWAY_LONG_SECONDS_DOWN_LAP:
                    self.attention_state = "AWAY_LONG"
                elif elapsed >= AWAY_SHORT_SECONDS_DOWN_LAP:
                    self.attention_state = "AWAY_SHORT"
                else:
                    self.attention_state = "ON_SCREEN"
            else:
                self.attention_state = "ON_SCREEN"

        probability = {
            "ON_SCREEN": 0.0,
            "AWAY_SHORT": 0.5,
        }.get(self.attention_state, 1.0)

        suspicion = self._compute_behavioral_suspicion(current_time)

        logger.debug(
            f"[Frame] state={self.attention_state:<12} dir={direction:<10} "
            f"axis={axis_outside:<10} h={avg_x:.3f} v={avg_y:.3f} "
            f"pitch={pitch_deg:+.1f}° zone={_diag['pitch_zone']:<8} "
            f"v_dev={_diag['v_deviation']:+.3f} "
            f"tol_y={_diag['tolerance_y']:.3f} "
            f"down={_diag['looking_down']} v_out={_diag['v_outside']}"
        )

        if self.attention_state == "AWAY_SHORT":
            logger.warning(
                f"[GazeSession] AWAY_SHORT | dir={direction} "
                f"sus={suspicion['level']}"
            )
        elif self.attention_state == "AWAY_LONG":
            logger.warning(
                f"[GazeSession] AWAY_LONG  | dir={direction} "
                f"sus={suspicion['level']} score={suspicion['score']:.3f} "
                f"| {suspicion['detail']}"
            )

        return (
            self._build_event(
                self.attention_state,
                probability,
                self.attention_state,
                current_time,
                suspicion,
            ),
            v_diag,
        )

    # ── sustained-gaze trackers ───────────────────────────────────────
    def _update_sustained_trackers(
        self,
        direction: str,
        current_time: float,
        inside_safe_zone: bool,
        is_down: bool,
    ) -> None:
        if inside_safe_zone:
            if self._continuous_down_start is not None:
                duration = current_time - self._continuous_down_start
                if duration >= SUSTAINED_DOWN_MIN_SECONDS:
                    self._total_sustained_down_seconds += duration
                    logger.info(
                        f"[GazeSession] Sustained DOWN closed | "
                        f"duration={duration:.1f}s | "
                        f"total={self._total_sustained_down_seconds:.1f}s"
                    )
                self._continuous_down_start = None

            if self._continuous_horizontal_start is not None:
                duration = current_time - self._continuous_horizontal_start
                if duration >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                    self._total_sustained_horizontal_seconds += duration
                    logger.info(
                        f"[GazeSession] Sustained H closed | "
                        f"duration={duration:.1f}s | "
                        f"total={self._total_sustained_horizontal_seconds:.1f}s"
                    )
                self._continuous_horizontal_start = None
        else:
            if is_down:
                if self._continuous_down_start is None:
                    self._continuous_down_start = current_time
                if self._continuous_horizontal_start is not None:
                    duration = current_time - self._continuous_horizontal_start
                    if duration >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                        self._total_sustained_horizontal_seconds += duration
                    self._continuous_horizontal_start = None

            elif direction == "LEFT_RIGHT":
                if self._continuous_horizontal_start is None:
                    self._continuous_horizontal_start = current_time
                if self._continuous_down_start is not None:
                    duration = current_time - self._continuous_down_start
                    if duration >= SUSTAINED_DOWN_MIN_SECONDS:
                        self._total_sustained_down_seconds += duration
                    self._continuous_down_start = None

    # ── behavioural suspicion ─────────────────────────────────────────
    def _compute_behavioral_suspicion(self, current_time: float) -> dict:
        window_start = current_time - BEHAVIOR_WINDOW_SECONDS

        recent: list[BehaviorEvent] = []
        for e in reversed(self.behavior_history):
            if e.timestamp >= window_start:
                recent.append(e)
            else:
                break
        recent.reverse()

        if len(recent) < BEHAVIOR_MIN_EVENTS:
            return {
                "level": SuspicionLevel.LOW,
                "score": 0.0,
                "down_ratio": 0.0,
                "down_lap_ratio": 0.0,
                "horizontal_ratio": 0.0,
                "no_face_ratio": 0.0,
                "transitions": 0,
                "detail": "insufficient data",
            }

        total = len(recent)
        down_write_count = sum(1 for e in recent if e.direction == "DOWN")
        down_lap_count = sum(1 for e in recent if e.direction == "DOWN_LAP")
        horizontal_count = sum(
            1 for e in recent if e.direction == "LEFT_RIGHT"
        )
        no_face_count = sum(1 for e in recent if e.state == "NO_FACE")

        down_write_ratio = down_write_count / total
        down_lap_ratio = down_lap_count / total
        horizontal_ratio = horizontal_count / total
        no_face_ratio = no_face_count / total

        transitions = sum(
            1
            for i in range(1, len(recent))
            if (recent[i - 1].state == "ON_SCREEN")
            != (recent[i].state == "ON_SCREEN")
        )

        # DOWN_LAP weighted almost as heavily as horizontal.
        # DOWN (writing) gets a low weight — it's often benign.
        raw_score = (
            horizontal_ratio * 0.40
            + down_lap_ratio * 0.35
            + down_write_ratio * 0.10
            + no_face_ratio * 0.15
        )

        # ── writing pattern detection ─────────────────────────────────
        down_total = down_write_count + down_lap_count
        down_ratio_total = down_total / total

        if down_ratio_total > 0.10 and down_total > 0:
            down_episodes = 0
            for i in range(len(recent)):
                if recent[i].direction in ("DOWN", "DOWN_LAP"):
                    if i == 0 or recent[i - 1].direction not in (
                        "DOWN",
                        "DOWN_LAP",
                    ):
                        down_episodes += 1

            if down_episodes > 0:
                avg_episode_frames = down_total / down_episodes
                avg_episode_seconds = avg_episode_frames / 30.0

                if avg_episode_seconds < WRITING_PATTERN_AVG_EPISODE_MAX_SECONDS:
                    raw_score *= 0.70
                    detail = (
                        f"writing pattern — avg down episode "
                        f"{avg_episode_seconds:.1f}s"
                    )
                else:
                    detail = (
                        f"sustained down gaze — avg down episode "
                        f"{avg_episode_seconds:.1f}s"
                    )
            else:
                detail = "sustained down gaze detected"
        elif horizontal_ratio > 0.25:
            detail = "frequent horizontal gaze detected"
        elif no_face_ratio > NO_FACE_SUSPICIOUS_RATIO:
            detail = "frequent face absence detected"
        else:
            detail = "gaze appears normal"

        # ── sustained session-level checks ────────────────────────────
        session_elapsed = max(1.0, current_time - self._session_start_time)

        ongoing_down = 0.0
        if self._continuous_down_start is not None:
            ongoing = current_time - self._continuous_down_start
            if ongoing >= SUSTAINED_DOWN_MIN_SECONDS:
                ongoing_down = ongoing

        ongoing_horizontal = 0.0
        if self._continuous_horizontal_start is not None:
            ongoing = current_time - self._continuous_horizontal_start
            if ongoing >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                ongoing_horizontal = ongoing

        sustained_down_ratio = (
            self._total_sustained_down_seconds + ongoing_down
        ) / session_elapsed
        sustained_horizontal_ratio = (
            self._total_sustained_horizontal_seconds + ongoing_horizontal
        ) / session_elapsed

        if sustained_horizontal_ratio > SUSTAINED_HORIZONTAL_SUSPICIOUS_RATIO:
            raw_score = min(1.0, raw_score + 0.20)
            detail += " | high sustained horizontal in session"

        if sustained_down_ratio > SUSTAINED_DOWN_SUSPICIOUS_RATIO:
            raw_score = min(1.0, raw_score + 0.10)
            detail += " | high sustained down in session"

        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        if raw_score >= 0.60:
            level = SuspicionLevel.HIGH
        elif raw_score >= 0.30:
            level = SuspicionLevel.MEDIUM
        else:
            level = SuspicionLevel.LOW

        return {
            "level": level,
            "score": round(raw_score, 4),
            "down_ratio": round(down_write_ratio, 4),
            "down_lap_ratio": round(down_lap_ratio, 4),
            "horizontal_ratio": round(horizontal_ratio, 4),
            "no_face_ratio": round(no_face_ratio, 4),
            "transitions": transitions,
            "detail": detail,
        }

    # ── event builder ─────────────────────────────────────────────────
    def _build_event(
        self,
        state: str,
        probability: float,
        evidence: str,
        timestamp: float | None = None,
        suspicion: dict | None = None,
    ) -> dict:
        self.model_id += 1
        ts = (
            datetime.fromtimestamp(timestamp).isoformat()
            if timestamp
            else datetime.now().isoformat()
        )
        return {
            "id": self.model_id,
            "timestamp": ts,
            "flag": state,
            "probability": round(probability, 4),
            "evidence": evidence,
            "suspicion": suspicion
            or {
                "level": SuspicionLevel.LOW,
                "score": 0.0,
                "down_ratio": 0.0,
                "down_lap_ratio": 0.0,
                "horizontal_ratio": 0.0,
                "no_face_ratio": 0.0,
                "transitions": 0,
                "detail": "no data",
            },
        }


# ─────────────────────────────────────────────
# camera loop + overlay
# ─────────────────────────────────────────────
def _get_state_color(state: str) -> tuple[int, int, int]:
    return {
        "ON_SCREEN": COLOR_GREEN,
        "AWAY_SHORT": COLOR_YELLOW,
        "AWAY_LONG": COLOR_RED,
        "NO_FACE": COLOR_RED,
        "INITIALIZING": COLOR_GRAY,
    }.get(state, COLOR_WHITE)


def _draw_overlay(
    frame: np.ndarray,
    verdict: dict,
    direction: str,
    avg_x: float,
    avg_y: float,
    raw_v: float,
    baseline_center_y: float | None,
    baseline_std_y: float | None,
    v_deviation: float,
    tolerance_y: float,
    looking_down_flag: bool,
    v_outside_flag: bool,
    pitch_deg: float,
    pitch_zone: str,
    fps: float,
    question_type: str = "normal",
    baseline_pitch: float | None = None,
    active_pitch_threshold: float | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    state = verdict["flag"]
    sus = verdict["suspicion"]
    color = _get_state_color(state)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (440, 460), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def put(text, row, col=10, scale=0.50, thickness=1, clr=COLOR_WHITE):
        cv2.putText(
            frame,
            text,
            (col, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            clr,
            thickness,
            cv2.LINE_AA,
        )

    # ── question mode badge ───────────────────
    qtype_label = "MODE: WRITING  [W=writing  N=normal]"
    qtype_clr   = COLOR_ORANGE
    if question_type == "normal":
        qtype_label = "MODE: NORMAL   [W=writing  N=normal]"
        qtype_clr   = COLOR_CYAN
    put(qtype_label, 18, scale=0.45, thickness=1, clr=qtype_clr)

    # ── state ─────────────────────────────────
    put(f"STATE: {state}", 40, scale=0.70, thickness=2, clr=color)

    # ── direction with colour coding ──────────
    dir_clr = COLOR_WHITE
    if direction == "DOWN_LAP":
        dir_clr = COLOR_RED
    elif direction == "DOWN":
        dir_clr = COLOR_ORANGE
    elif direction == "LEFT_RIGHT":
        dir_clr = COLOR_YELLOW
    elif direction == "CENTER":
        dir_clr = COLOR_GREEN
    put(f"Direction  : {direction}", 67, clr=dir_clr)

    put(f"Gaze H     : {avg_x:.4f}", 87)
    put(f"Gaze V     : {avg_y:.4f}", 107)

    # ── suspicion ─────────────────────────────
    sus_color = {
        "LOW": COLOR_GREEN,
        "MEDIUM": COLOR_YELLOW,
        "HIGH": COLOR_RED,
    }.get(str(sus["level"]), COLOR_WHITE)

    put(
        f"Suspicion  : {sus['level']}  ({sus['score']:.3f})",
        132,
        clr=sus_color,
    )
    put(f"H-ratio    : {sus['horizontal_ratio']:.4f}", 152)
    put(f"D-write    : {sus['down_ratio']:.4f}", 172)
    put(f"D-lap      : {sus['down_lap_ratio']:.4f}", 192, clr=COLOR_RED if sus['down_lap_ratio'] > 0.10 else COLOR_WHITE)

    # ── V diagnostics ─────────────────────────
    put("--- V DIAGNOSTICS -----------------", 217, clr=COLOR_GRAY)

    put(f"raw_v          : {raw_v:.4f}", 237)
    put(
        (
            f"baseline_ctr_y : {baseline_center_y:.4f}"
            if baseline_center_y is not None
            else "baseline_ctr_y : NOT SET YET"
        ),
        257,
    )
    put(
        (
            f"baseline_std_y : {baseline_std_y:.4f}"
            if baseline_std_y is not None
            else "baseline_std_y : NOT SET YET"
        ),
        277,
    )
    put(f"v_deviation    : {v_deviation:+.4f}", 297)
    put(f"tolerance_y    : {tolerance_y:.4f}", 317)

    # Pitch with zone colour
    zone_clr = {
        "LEVEL": COLOR_GREEN,
        "WRITING": COLOR_ORANGE,
        "LAP": COLOR_RED,
    }.get(pitch_zone, COLOR_WHITE)
    put(
        f"pitch          : {pitch_deg:+.1f}°  [{pitch_zone}]",
        337,
        clr=zone_clr,
    )
    # Pitch calibration row
    if baseline_pitch is not None:
        put(
            f"pitch baseline : {baseline_pitch:+.1f}°  (neutral)",
            357,
            clr=COLOR_GRAY,
        )
        eff_thresh = active_pitch_threshold if active_pitch_threshold is not None else 0.0
        put(
            f"lap threshold  : {eff_thresh:.1f}°  "
            f"(base{baseline_pitch:+.1f} + {'22' if question_type == 'writing' else '18'}°)",
            377,
            clr=COLOR_GRAY,
        )
    else:
        put(
            f"lap threshold  : calibrating...",
            357,
            clr=COLOR_GRAY,
        )

    put(
        f"looking_down   : {looking_down_flag}",
        397,
        clr=COLOR_GREEN if looking_down_flag else (200, 200, 200),
    )
    put(
        f"v_outside      : {v_outside_flag}",
        417,
        clr=COLOR_GREEN if v_outside_flag else COLOR_RED,
    )

    put(f"FPS: {fps:.1f}", 427, col=w - 100)

    detail = sus.get("detail", "")
    if detail and detail != "insufficient data":
        put(detail[:70], h - 15, scale=0.40, clr=COLOR_GRAY)

    return frame


def run_live_camera(
    session_id: str = "live_session",
    camera_index: int = 0,
    question_type: str = GazeSession.QUESTION_TYPE_NORMAL,
) -> None:
    """Run the live gaze-detection loop.

    Parameters
    ----------
    session_id    : Unique identifier for this proctoring session.
    camera_index  : OpenCV camera index (default 0).
    question_type : "normal" (default) — person must not look down;
                    "writing"          — person may look down to write.
    """
    logger.info(f"[Camera] Opening camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(f"[Camera] Could not open camera index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    session = manager.get_or_create(session_id)
    session.set_question_type(question_type)
    prev_time = time.time()
    frame_idx = 0
    last_avg_x = 0.5
    last_avg_y = 0.5

    last_v_diag: dict = {
        "raw_v": 0.0,
        "avg_y": 0.0,
        "v_deviation": 0.0,
        "tolerance_y": 0.0,
        "looking_down": False,
        "v_outside": False,
        "baseline_center_y": None,
        "baseline_std_y": None,
        "pitch_deg": 0.0,
        "pitch_zone": "LEVEL",
    }

    logger.info(
        "[Camera] Live gaze detection running. "
        "Press Q=quit | N=normal MCQ | W=writing MCQ"
    )
    logger.info(
        f"[Camera] Calibrating with {ENVELOPE_WINDOW} frames — "
        f"look at screen normally."
    )
    logger.info(
        f"[Camera] Pitch lap threshold — writing: {PITCH_LAP_THRESHOLD_WRITING}° | "
        f"normal: {PITCH_LAP_THRESHOLD_NORMAL}° | "
        f"DOWN_WRITE timers: {AWAY_SHORT_SECONDS_DOWN_WRITE}s/{AWAY_LONG_SECONDS_DOWN_WRITE}s | "
        f"DOWN_LAP timers: {AWAY_SHORT_SECONDS_DOWN_LAP}s/{AWAY_LONG_SECONDS_DOWN_LAP}s"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("[Camera] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - prev_time)
        prev_time = current_time

        t0 = time.time()
        h, v, face = session.detector.get_gaze_ratio(frame)
        pitch_deg = session.detector.last_pitch_deg
        t1 = time.time()

        if face and session.baseline_center_x is not None:
            tmp_x = list(session.gaze_history_x) + [h]
            tmp_y = list(session.gaze_history_y) + [v]
            tmp_x = tmp_x[-SMOOTHING_BUFFER_SIZE:]
            tmp_y = tmp_y[-SMOOTHING_BUFFER_SIZE:]
            last_avg_x = sum(tmp_x) / len(tmp_x)
            last_avg_y = sum(tmp_y) / len(tmp_y)

        verdict, v_diag = session.process_gaze_result(
            h, v, face, pitch_deg=pitch_deg, timestamp=current_time
        )
        last_v_diag = v_diag
        last_direction = session.last_direction

        logger.debug(
            f"[Frame {frame_idx:05d}] mp={t1 - t0:.3f}s face={face} "
            f"h={h:.3f} v={v:.3f} pitch={pitch_deg:+.1f}° "
            f"dir={last_direction} state={verdict['flag']}"
        )

        frame = _draw_overlay(
            frame=frame,
            verdict=verdict,
            direction=last_direction,
            avg_x=last_avg_x,
            avg_y=last_avg_y,
            raw_v=last_v_diag["raw_v"],
            baseline_center_y=last_v_diag["baseline_center_y"],
            baseline_std_y=last_v_diag["baseline_std_y"],
            v_deviation=last_v_diag["v_deviation"],
            tolerance_y=last_v_diag["tolerance_y"],
            looking_down_flag=last_v_diag["looking_down"],
            v_outside_flag=last_v_diag["v_outside"],
            pitch_deg=last_v_diag["pitch_deg"],
            pitch_zone=last_v_diag["pitch_zone"],
            fps=fps,
            question_type=session.question_type,
            baseline_pitch=last_v_diag.get("baseline_pitch"),
            active_pitch_threshold=last_v_diag.get("active_pitch_threshold"),
        )

        cv2.imshow("Gaze Detector  —  N=normal  W=writing  Q=quit", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("[Camera] Q pressed — shutting down.")
            break
        elif key == ord("n"):
            session.set_question_type(GazeSession.QUESTION_TYPE_NORMAL)
            logger.info("[Camera] Switched to NORMAL MCQ mode.")
        elif key == ord("w"):
            session.set_question_type(GazeSession.QUESTION_TYPE_WRITING)
            logger.info("[Camera] Switched to WRITING MCQ mode.")

    cap.release()
    cv2.destroyAllWindows()
    manager.clear(session_id)
    logger.info("[Camera] Released. Session cleared.")


if __name__ == "__main__":
    # Change question_type to "writing" when testing a writing-style MCQ question
    run_live_camera(
        session_id="dev_test",
        camera_index=0,
        question_type=GazeSession.QUESTION_TYPE_NORMAL,
    )