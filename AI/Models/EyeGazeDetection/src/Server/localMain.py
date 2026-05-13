from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum

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
AWAY_SHORT_SECONDS_DOWN_WRITE = 5.0
AWAY_LONG_SECONDS_DOWN_WRITE = 7.0

# ── away timers — down (lap / extreme) ────────
AWAY_SHORT_SECONDS_DOWN_LAP = 3.0
AWAY_LONG_SECONDS_DOWN_LAP = 5.0

# ── pitch thresholds ──────────────────────────
# Applied as offsets from each person's calibrated neutral pitch.
#   writing — lenient (+24°): head-down is expected
#   normal  — stricter (+18°): flags sooner
PITCH_LAP_THRESHOLD_WRITING = 24.0
PITCH_LAP_THRESHOLD_NORMAL  = 18.0

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
TOLERANCE_Y_MAX = 0.08


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
# GazeSession
# ─────────────────────────────────────────────
class GazeSession:
    # ── question-type constants ────────────────────────────────────────
    QUESTION_TYPE_NORMAL  = "normal"   # Standard MCQ — looking down is NOT expected
    QUESTION_TYPE_WRITING = "writing"  # Writing MCQ  — looking down to write is OK

    def __init__(self, session_id: str, question_type: str = "normal"):
        self.session_id = session_id
        self.detector = GazeDetector()

        # ── MCQ question type ──────────────────────────────────────────
        # Controlled by the backend via set_question_type(is_writing: bool)
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
        self.pitch_envelope: deque[float] = deque(maxlen=ENVELOPE_WINDOW)
        self.baseline_pitch: float | None = None

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
            f"[GazeSession] Initialized: {session_id} | question_type={self._question_type}"
        )

    # ── question-type API ─────────────────────────────────────────────
    @property
    def question_type(self) -> str:
        return self._question_type

    def set_question_type(self, is_writing: bool) -> None:
        """Switch MCQ mode when the backend sends a new question.

        Parameters
        ----------
        is_writing : bool
            False → normal MCQ  (looking down is flagged quickly).
            True  → writing MCQ (looking down to write is ignored).
        """
        qtype = self.QUESTION_TYPE_WRITING if is_writing else self.QUESTION_TYPE_NORMAL
        if qtype != self._question_type:
            logger.info(
                f"[GazeSession] question_type: {self._question_type!r} → {qtype!r}"
            )
            self._question_type = qtype
            self.away_start_time_vertical_write = None
            self.away_start_time_vertical_lap = None

    # ── classify gaze ─────────────────────────────────────────────────
    def _classify_gaze(
        self, avg_x: float, avg_y: float, pitch_deg: float,
        pitch_lap_threshold: float = PITCH_LAP_THRESHOLD_WRITING,
    ) -> tuple[str, str, dict]:
        h_deviation = avg_x - self.baseline_center_x
        v_deviation = avg_y - self.baseline_center_y
        h_abs = abs(h_deviation)
        v_abs = abs(v_deviation)

        tolerance_x = max(0.07, self.baseline_std_x * 3.5)
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
            direction = "DOWN_LAP" if pitch_deg > pitch_lap_threshold else "DOWN"
        else:
            h_norm = h_abs / tolerance_x
            v_norm = v_abs / tolerance_y
            if h_norm >= v_norm:
                direction = "LEFT_RIGHT"
            else:
                direction = "DOWN_LAP" if pitch_deg > pitch_lap_threshold else "DOWN"

        pitch_zone = (
            "LAP" if pitch_deg > pitch_lap_threshold
            else "WRITING" if pitch_deg < pitch_lap_threshold * 0.9
            else "LEVEL"
        )

        diag = {
            "v_deviation": v_deviation, "tolerance_y": tolerance_y,
            "looking_down": looking_down, "v_outside": v_outside,
            "h_deviation": h_deviation, "tolerance_x": tolerance_x,
            "pitch_zone": pitch_zone,
        }
        return direction, axis_outside, diag

    # ── main processing — call this once per camera frame ─────────────
    def process_gaze_result(
        self,
        h_ratio: float,
        v_ratio: float,
        face_present: bool,
        pitch_deg: float = 0.0,
        timestamp: float | None = None,
    ) -> tuple[dict, dict]:
        """Process one frame's gaze data and return the verdict.

        Parameters
        ----------
        h_ratio      : Horizontal gaze ratio (0 = far left, 1 = far right).
        v_ratio      : Vertical gaze ratio (0 = far down, 1 = far up).
        face_present : Whether a face was detected in this frame.
        pitch_deg    : Head pitch in degrees (positive = head tilting down).
        timestamp    : Frame timestamp; defaults to time.time().

        Returns
        -------
        verdict : Event dict with flag, probability, suspicion, etc.
        v_diag  : Vertical diagnostics dict (for debugging/overlay).
        """
        _empty_diag: dict = {
            "raw_v": 0.0, "avg_y": 0.0, "v_deviation": 0.0,
            "tolerance_y": 0.0, "looking_down": False, "v_outside": False,
            "baseline_center_y": None, "baseline_std_y": None,
            "pitch_deg": 0.0, "pitch_zone": "LEVEL",
            "baseline_pitch": self.baseline_pitch,
            "active_pitch_threshold": None,
        }

        current_time = timestamp if timestamp is not None else time.time()
        dt = (current_time - self._last_process_time) if self._last_process_time else 1.0 / 30.0
        self._last_process_time = current_time

        # ── no face ───────────────────────────────────────────────────
        if not face_present:
            if self._continuous_down_start is not None:
                dur = current_time - self._continuous_down_start
                if dur >= SUSTAINED_DOWN_MIN_SECONDS:
                    self._total_sustained_down_seconds += dur
            if self._continuous_horizontal_start is not None:
                dur = current_time - self._continuous_horizontal_start
                if dur >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                    self._total_sustained_horizontal_seconds += dur

            self.attention_state = "NO_FACE"
            self.away_start_time_horizontal = None
            self.away_start_time_vertical_write = None
            self.away_start_time_vertical_lap = None
            self._continuous_down_start = None
            self._continuous_horizontal_start = None
            self._total_no_face_seconds += dt

            self.behavior_history.append(BehaviorEvent(current_time, "NO_FACE", "UNKNOWN"))
            suspicion = self._compute_behavioral_suspicion(current_time)
            return self._build_event("NO_FACE", 1.0, "NO_FACE", current_time, suspicion), _empty_diag

        # ── calibrating ───────────────────────────────────────────────
        if not self.initialized:
            self.envelope_x.append(h_ratio)
            self.envelope_y.append(v_ratio)
            if pitch_deg != 0.0:
                self.pitch_envelope.append(pitch_deg)
            if len(self.envelope_x) >= ENVELOPE_WINDOW and len(self.envelope_y) >= ENVELOPE_WINDOW:
                self.initialized = True
                logger.info(f"[GazeSession] Calibration complete — {ENVELOPE_WINDOW} frames")

            self.behavior_history.append(BehaviorEvent(current_time, "INITIALIZING", "CENTER"))
            suspicion = self._compute_behavioral_suspicion(current_time)
            cal_diag = {
                **_empty_diag,
                "raw_v": v_ratio, "avg_y": v_ratio, "pitch_deg": pitch_deg,
                "baseline_pitch": self.baseline_pitch, "active_pitch_threshold": None,
            }
            return self._build_event("INITIALIZING", 0.0, "CALIBRATING", current_time, suspicion), cal_diag

        # ── baselines (set once after calibration) ────────────────────
        if self.baseline_center_x is None:
            self.baseline_center_x = float(np.median(self.envelope_x))
            self.baseline_std_x = float(np.std(self.envelope_x))
        if self.baseline_center_y is None:
            self.baseline_center_y = float(np.median(self.envelope_y))
            self.baseline_std_y = float(np.std(self.envelope_y))
        if self.baseline_pitch is None:
            if len(self.pitch_envelope) >= max(10, ENVELOPE_WINDOW // 3):
                self.baseline_pitch = float(np.max(self.pitch_envelope))
                logger.info(
                    f"[GazeSession] Pitch baseline — neutral={self.baseline_pitch:.1f}° | "
                    f"effective normal={self.baseline_pitch + PITCH_LAP_THRESHOLD_NORMAL:.1f}° | "
                    f"effective writing={self.baseline_pitch + PITCH_LAP_THRESHOLD_WRITING:.1f}°"
                )
            else:
                self.baseline_pitch = 0.0
                logger.warning("[GazeSession] Pitch baseline: insufficient samples, defaulting to 0°")

        # ── smooth ────────────────────────────────────────────────────
        self.gaze_history_x.append(h_ratio)
        self.gaze_history_y.append(v_ratio)
        avg_x = sum(self.gaze_history_x) / len(self.gaze_history_x)
        avg_y = sum(self.gaze_history_y) / len(self.gaze_history_y)

        # ── classify (pitch-aware, relative to calibrated baseline) ───
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

        # ── question-type override ────────────────────────────────────
        # writing mode: DOWN is expected — treat as CENTER, no timer.
        # normal mode:  the pitch threshold already splits DOWN / DOWN_LAP;
        #               no further override needed.
        if direction == "DOWN" and self._question_type == self.QUESTION_TYPE_WRITING:
            direction = "CENTER"
            axis_outside = "NONE"

        v_diag = {
            "raw_v": v_ratio, "avg_y": avg_y,
            "v_deviation": _diag["v_deviation"], "tolerance_y": _diag["tolerance_y"],
            "looking_down": _diag["looking_down"], "v_outside": _diag["v_outside"],
            "baseline_center_y": self.baseline_center_y, "baseline_std_y": self.baseline_std_y,
            "pitch_deg": pitch_deg, "pitch_zone": _diag["pitch_zone"],
            "baseline_pitch": self.baseline_pitch,
            "active_pitch_threshold": active_pitch_threshold,
        }

        if direction != self.last_direction:
            logger.info(
                f"[GazeSession] Direction: {self.last_direction} → {direction} | "
                f"avg_x={avg_x:.3f} avg_y={avg_y:.3f} pitch={pitch_deg:.1f}°"
            )
            self.last_direction = direction

        self.behavior_history.append(BehaviorEvent(current_time, self.attention_state, direction))

        is_down = direction in ("DOWN", "DOWN_LAP")
        inside_safe_zone = axis_outside == "NONE"
        self._update_sustained_trackers(direction, current_time, inside_safe_zone, is_down)

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
                self.away_start_time_vertical_write = None
                self.away_start_time_vertical_lap = None
                elapsed = current_time - self.away_start_time_horizontal
                self.attention_state = (
                    "AWAY_LONG" if elapsed >= AWAY_LONG_SECONDS_HORIZONTAL
                    else "AWAY_SHORT" if elapsed >= AWAY_SHORT_SECONDS_HORIZONTAL
                    else "ON_SCREEN"
                )

            elif direction == "DOWN":
                if self.away_start_time_vertical_write is None:
                    self.away_start_time_vertical_write = current_time
                self.away_start_time_horizontal = None
                self.away_start_time_vertical_lap = None
                elapsed = current_time - self.away_start_time_vertical_write
                self.attention_state = (
                    "AWAY_LONG" if elapsed >= AWAY_LONG_SECONDS_DOWN_WRITE
                    else "AWAY_SHORT" if elapsed >= AWAY_SHORT_SECONDS_DOWN_WRITE
                    else "ON_SCREEN"
                )

            elif direction == "DOWN_LAP":
                if self.away_start_time_vertical_lap is None:
                    self.away_start_time_vertical_lap = current_time
                self.away_start_time_horizontal = None
                self.away_start_time_vertical_write = None
                elapsed = current_time - self.away_start_time_vertical_lap
                self.attention_state = (
                    "AWAY_LONG" if elapsed >= AWAY_LONG_SECONDS_DOWN_LAP
                    else "AWAY_SHORT" if elapsed >= AWAY_SHORT_SECONDS_DOWN_LAP
                    else "ON_SCREEN"
                )
            else:
                self.attention_state = "ON_SCREEN"

        probability = {"ON_SCREEN": 0.0, "AWAY_SHORT": 0.5}.get(self.attention_state, 1.0)
        suspicion = self._compute_behavioral_suspicion(current_time)

        return self._build_event(
            self.attention_state, probability, self.attention_state, current_time, suspicion
        ), v_diag

    # ── sustained-gaze trackers ───────────────────────────────────────
    def _update_sustained_trackers(
        self, direction: str, current_time: float, inside_safe_zone: bool, is_down: bool
    ) -> None:
        if inside_safe_zone:
            if self._continuous_down_start is not None:
                dur = current_time - self._continuous_down_start
                if dur >= SUSTAINED_DOWN_MIN_SECONDS:
                    self._total_sustained_down_seconds += dur
                self._continuous_down_start = None
            if self._continuous_horizontal_start is not None:
                dur = current_time - self._continuous_horizontal_start
                if dur >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                    self._total_sustained_horizontal_seconds += dur
                self._continuous_horizontal_start = None
        else:
            if is_down:
                if self._continuous_down_start is None:
                    self._continuous_down_start = current_time
                if self._continuous_horizontal_start is not None:
                    dur = current_time - self._continuous_horizontal_start
                    if dur >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                        self._total_sustained_horizontal_seconds += dur
                    self._continuous_horizontal_start = None
            elif direction == "LEFT_RIGHT":
                if self._continuous_horizontal_start is None:
                    self._continuous_horizontal_start = current_time
                if self._continuous_down_start is not None:
                    dur = current_time - self._continuous_down_start
                    if dur >= SUSTAINED_DOWN_MIN_SECONDS:
                        self._total_sustained_down_seconds += dur
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
                "level": SuspicionLevel.LOW, "score": 0.0,
                "down_ratio": 0.0, "down_lap_ratio": 0.0,
                "horizontal_ratio": 0.0, "no_face_ratio": 0.0,
                "transitions": 0, "detail": "insufficient data",
            }

        total = len(recent)
        down_write_count  = sum(1 for e in recent if e.direction == "DOWN")
        down_lap_count    = sum(1 for e in recent if e.direction == "DOWN_LAP")
        horizontal_count  = sum(1 for e in recent if e.direction == "LEFT_RIGHT")
        no_face_count     = sum(1 for e in recent if e.state == "NO_FACE")

        down_write_ratio  = down_write_count / total
        down_lap_ratio    = down_lap_count / total
        horizontal_ratio  = horizontal_count / total
        no_face_ratio     = no_face_count / total

        transitions = sum(
            1 for i in range(1, len(recent))
            if (recent[i - 1].state == "ON_SCREEN") != (recent[i].state == "ON_SCREEN")
        )

        raw_score = (
            horizontal_ratio * 0.40
            + down_lap_ratio * 0.35
            + down_write_ratio * 0.10
            + no_face_ratio * 0.15
        )

        down_total = down_write_count + down_lap_count
        detail = "gaze appears normal"

        if (down_total / total) > 0.10 and down_total > 0:
            down_episodes = sum(
                1 for i in range(len(recent))
                if recent[i].direction in ("DOWN", "DOWN_LAP")
                and (i == 0 or recent[i - 1].direction not in ("DOWN", "DOWN_LAP"))
            )
            if down_episodes > 0:
                avg_ep_sec = (down_total / down_episodes) / 30.0
                if avg_ep_sec < WRITING_PATTERN_AVG_EPISODE_MAX_SECONDS:
                    raw_score *= 0.70
                    detail = f"writing pattern — avg down episode {avg_ep_sec:.1f}s"
                else:
                    detail = f"sustained down gaze — avg down episode {avg_ep_sec:.1f}s"
            else:
                detail = "sustained down gaze detected"
        elif horizontal_ratio > 0.25:
            detail = "frequent horizontal gaze detected"
        elif no_face_ratio > NO_FACE_SUSPICIOUS_RATIO:
            detail = "frequent face absence detected"

        session_elapsed = max(1.0, current_time - self._session_start_time)

        ongoing_down = 0.0
        if self._continuous_down_start is not None:
            o = current_time - self._continuous_down_start
            if o >= SUSTAINED_DOWN_MIN_SECONDS:
                ongoing_down = o

        ongoing_horizontal = 0.0
        if self._continuous_horizontal_start is not None:
            o = current_time - self._continuous_horizontal_start
            if o >= SUSTAINED_HORIZONTAL_MIN_SECONDS:
                ongoing_horizontal = o

        if (self._total_sustained_horizontal_seconds + ongoing_horizontal) / session_elapsed > SUSTAINED_HORIZONTAL_SUSPICIOUS_RATIO:
            raw_score = min(1.0, raw_score + 0.20)
            detail += " | high sustained horizontal in session"

        if (self._total_sustained_down_seconds + ongoing_down) / session_elapsed > SUSTAINED_DOWN_SUSPICIOUS_RATIO:
            raw_score = min(1.0, raw_score + 0.10)
            detail += " | high sustained down in session"

        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        level = (
            SuspicionLevel.HIGH   if raw_score >= 0.60
            else SuspicionLevel.MEDIUM if raw_score >= 0.30
            else SuspicionLevel.LOW
        )

        return {
            "level": level, "score": round(raw_score, 4),
            "down_ratio": round(down_write_ratio, 4),
            "down_lap_ratio": round(down_lap_ratio, 4),
            "horizontal_ratio": round(horizontal_ratio, 4),
            "no_face_ratio": round(no_face_ratio, 4),
            "transitions": transitions, "detail": detail,
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
            if timestamp else datetime.now().isoformat()
        )
        return {
            "id": self.model_id,
            "timestamp": ts,
            "flag": state,
            "probability": round(probability, 4),
            "evidence": evidence,
            "suspicion": suspicion or {
                "level": SuspicionLevel.LOW, "score": 0.0,
                "down_ratio": 0.0, "down_lap_ratio": 0.0,
                "horizontal_ratio": 0.0, "no_face_ratio": 0.0,
                "transitions": 0, "detail": "no data",
            },
        }