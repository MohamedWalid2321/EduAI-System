import cv2 as cv
import numpy as np
from collections import deque
from Gaze import GazeDetector
from datetime import datetime

SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW       = 90
AWAY_SHORT_SECONDS = 3.0
AWAY_LONG_SECONDS  = 5.0

class GazeSession:

    def __init__(self):
        self._detector = GazeDetector()

        self._gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)

        self._envelope_x = deque(maxlen=ENVELOPE_WINDOW)

        self._initialized = False

        self._baseline_center_x = None
        self._baseline_std_x    = None

        self._attention_state = "INITIALIZING"
        self._away_start_time = None
        self._event_id        = 0

    def process_gaze_frame(self, frame):
        current_time = datetime.now().timestamp()

        h_ratio, _, face_present = self._detector.get_gaze_ratio(frame)

        if not face_present:
            self._attention_state = "NO_FACE"
            self._away_start_time = None
            return self._build_event("NO_FACE", 1.0, "NO_FACE")

        if not self._initialized:
            self._envelope_x.append(h_ratio)

            if len(self._envelope_x) >= ENVELOPE_WINDOW:
                self._initialized = True

            return self._build_event("INITIALIZING", 0.0, "CALIBRATING")

        if self._baseline_center_x is None:
            self._baseline_center_x = np.median(self._envelope_x)
            self._baseline_std_x    = np.std(self._envelope_x)

        self._gaze_history_x.append(h_ratio)
        avg_x = sum(self._gaze_history_x) / len(self._gaze_history_x)
        tolerance_x = max(0.07, self._baseline_std_x * 3.5)
        inside_safe_zone = abs(avg_x - self._baseline_center_x) <= tolerance_x

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

        if self._attention_state == "ON_SCREEN":
            probability = 0.0
        elif self._attention_state == "AWAY_SHORT":
            probability = 0.5
        else:   
            probability = 1.0

        return self._build_event(
            self._attention_state, probability, self._attention_state
        )

    def _build_event(self, state, probability, evidence):
        self._event_id = 1
        return {
            "id":              self._event_id,
            "timestamp":       datetime.now().isoformat(),
            "flag": state,
            "probability":     round(probability, 4),
            "evidence":        evidence,
        }