from __future__ import annotations
import cv2
import numpy as np
from collections import deque
from Gaze import GazeDetector
from datetime import datetime

import time
from collections import deque
from datetime import datetime
import logging
import sys
import threading
import time

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (180, 180, 180)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)

SMOOTHING_BUFFER_SIZE = 5
ENVELOPE_WINDOW       = 90

AWAY_SHORT_SECONDS = 3.0
AWAY_LONG_SECONDS  = 5.0



# Pitch threshold — offset from calibrated neutral pitch.
# If pitch > baseline_pitch + this value → looking DOWN.
PITCH_DOWN_THRESHOLD = 18.0

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

class GazeSession:

    def __init__(self):
        self._detector = GazeDetector()

        # Horizontal
        self._gaze_history_x    = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self._envelope_x        = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_center_x = None
        self._baseline_std_x    = None

        # Pitch calibration
        self._pitch_envelope    = deque(maxlen=ENVELOPE_WINDOW)
        self._baseline_pitch    = None

        self._initialized     = False
        self._attention_state = "INITIALIZING"
        self._away_start_time = None
        self._event_id        = 0

    def process_gaze_frame(self, frame):
        current_time = datetime.now().timestamp()

        h_ratio, v_ratio, face_present = self._detector.get_gaze_ratio(frame)
        pitch_deg = self._detector.last_pitch_deg

        if not face_present:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event("NO_FACE", 1.0, "NO_FACE")

        # ── calibration ──────────────────────────────────
        if not self._initialized:
            self._envelope_x.append(h_ratio)
            if pitch_deg != 0.0:
                self._pitch_envelope.append(pitch_deg)

            if len(self._envelope_x) >= ENVELOPE_WINDOW:
                self._initialized = True

            return self._build_event("INITIALIZING", 0.0, "CALIBRATING")

        # ── compute baselines once ───────────────────────
        if self._baseline_center_x is None:
            self._baseline_center_x = float(np.median(self._envelope_x))
            self._baseline_std_x    = float(np.std(self._envelope_x))

        if self._baseline_pitch is None:
            if len(self._pitch_envelope) >= max(10, ENVELOPE_WINDOW // 3):
                self._baseline_pitch = float(np.max(self._pitch_envelope))
            else:
                self._baseline_pitch = 0.0
            print(f"[Calibration] baseline_pitch={self._baseline_pitch:.1f} "
                  f"-> down threshold={self._baseline_pitch + PITCH_DOWN_THRESHOLD:.1f}")

        # ── smooth ───────────────────────────────────────
        self._gaze_history_x.append(h_ratio)
        avg_x = sum(self._gaze_history_x) / len(self._gaze_history_x)

        # ── horizontal check ─────────────────────────────
        tolerance_x   = max(0.07, self._baseline_std_x * 4.0)
        h_deviation   = avg_x - self._baseline_center_x
        h_outside     = abs(h_deviation) > tolerance_x

        # ── down check (pitch threshold) ─────────────────
        active_threshold = self._baseline_pitch + PITCH_DOWN_THRESHOLD
        looking_down     = pitch_deg > active_threshold

        inside_safe_zone = (not h_outside) and (not looking_down)

        # ── direction ────────────────────────────────────
        if h_outside and looking_down:
            direction = "BOTH"
        elif h_outside:
            direction = "LEFT" if h_deviation < 0 else "RIGHT"
        elif looking_down:
            direction = "DOWN"
        else:
            direction = "CENTER"

        # ── away timers ──────────────────────────────────
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


# ─────────────────────────────────────────────
# Live camera loop
# ─────────────────────────────────────────────
def run_live_camera(camera_index: int = 0) -> None:
    logger.info(f"[Camera] Opening camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(f"[Camera] Could not open camera index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    session = GazeSession()
    prev_time = time.time()
    frame_idx = 0

    logger.info("[Camera] Live gaze detection running. Press Q to quit.")
    logger.info(f"[Camera] Calibrating with {ENVELOPE_WINDOW} frames — look at screen normally.")
    logger.info(f"[Camera] Pitch down threshold offset: {PITCH_DOWN_THRESHOLD}°")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("[Camera] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - prev_time)
        prev_time = current_time

        event = session.process_gaze_frame(frame)
        state = event["attention_state"]
        direction = event["evidence"]

        # ── overlay ──────────────────────────────────
        color = COLOR_GREEN
        if state in ["AWAY_LONG", "NO_FACE"]:
            color = COLOR_RED
        elif state == "AWAY_SHORT":
            color = COLOR_YELLOW
        elif state == "INITIALIZING":
            color = COLOR_GRAY

        # Dark background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (520, 190), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # State + direction
        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        dir_clr = COLOR_GREEN if direction == "CENTER" else (
            COLOR_ORANGE if direction == "DOWN" else COLOR_YELLOW
        )
        cv2.putText(frame, f"Direction: {direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_clr, 1, cv2.LINE_AA)

        # ── Pitch row — always visible ────────────────
        live_pitch = session._detector.last_pitch_deg
        base_pitch = session._baseline_pitch
        active_threshold = (base_pitch + PITCH_DOWN_THRESHOLD) if base_pitch is not None else None

        if active_threshold is not None:
            above = live_pitch > active_threshold
            pitch_clr = COLOR_RED if above else COLOR_WHITE
            pitch_info = (f"pitch={live_pitch:+.1f}  "
                          f"base={base_pitch:+.1f}  "
                          f"thresh={active_threshold:.1f}  "
                          f"{'[DOWN]' if above else '[OK]'}")
        else:
            pitch_clr = COLOR_GRAY
            pitch_info = f"pitch={live_pitch:+.1f}  base=calibrating..."

        cv2.putText(frame, pitch_info, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_clr, 1, cv2.LINE_AA)

        # ── Horizontal + Yaw row ──────────────────────
        h_dev = event.get("h_dev")
        live_yaw = session._detector.last_yaw_deg
        if h_dev is not None:
            h_outside = abs(h_dev) > event.get("tol_x", 0.10)
            h_clr = COLOR_RED if h_outside else COLOR_WHITE
            info2 = f"h_dev={h_dev:+.3f}  tol_x={event['tol_x']:.3f}  yaw={live_yaw:+.1f}"
            cv2.putText(frame, info2, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, h_clr, 1, cv2.LINE_AA)
        else:
            info2 = f"yaw={live_yaw:+.1f}"
            cv2.putText(frame, info2, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

        # ── Calibration progress ──────────────────────
        if not session._initialized:
            progress = len(session._envelope_x)
            cal_text = f"Calibrating: {progress}/{ENVELOPE_WINDOW} frames"
            cv2.putText(frame, cal_text, (10, 143),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

        # FPS
        w_frame = frame.shape[1]
        cv2.putText(frame, f"FPS: {fps:.1f}", (w_frame - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.imshow("Gaze Detector — Q to quit", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("[Camera] Q pressed — shutting down.")
            break
    cap.release()
    cv2.destroyAllWindows()
    logger.info("[Camera] Released.")


if __name__ == "__main__":
    run_live_camera(camera_index=0)