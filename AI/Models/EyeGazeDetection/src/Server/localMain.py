from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from Gaze import GazeDetector

logger = logging.getLogger(__name__)

SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW       = 90
AWAY_SHORT_SECONDS    = 3.0
AWAY_LONG_SECONDS     = 5.0

class SessionManager:

    def __init__(self):
        #variable to hold GazeSession object
        self.session: GazeSession | None = None
        """
        we are doing a safety procedure in case in the backend sends two batches at the same time to same instance
        we do not encounter any problem, but it most cases it will not happen
        """
        self.session_lock: threading.Lock = threading.Lock()

    def get_or_create(self, session_id: str) -> GazeSession:
        if self.session is None:
            self.session = GazeSession(session_id)
            logger.info(f"[GazeSession] Session created: {session_id}")
        return self.session

    def clear(self, session_id: str) -> None:
        self.session = None
        logger.info(f"[GazeSession] Session cleared: {session_id}")

    @property
    def lock(self) -> threading.Lock:
        return self.session_lock


manager = SessionManager()

def process_frames_batch(session_id: str,frames_b64: list[str],fps: int = 30,) -> list[dict]:
    frame_interval = 1.0 / fps
    session = manager.get_or_create(session_id)

    with manager.lock:
        start            = time.time()
        results          = []
        batch_start_time = time.time()

        for index, frame_b64 in enumerate(frames_b64):

            frame_timestamp = batch_start_time + (index * frame_interval)

            try:
                #decode the base64 to bytes
                img_bytes = base64.b64decode(frame_b64)
                #convert the compressed bytes to 1D numpy array
                np_arr    = np.frombuffer(img_bytes, np.uint8)
                #decode the numpy array to OpenCV image
                frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as exc:
                logger.warning(f"[process_frames_batch] Frame {index} decode error: {exc}")
                frame = None

            #check if frame is present or not 
            if frame is None:
                h, v, face = 0.0, 0.0, False
            else:
                t0         = time.time()
                h, v, face = session.detector.get_gaze_ratio(frame)
                t1         = time.time()
                print(
                    f"  Frame {index:02d} | "
                    f"mediapipe={t1 - t0:.3f}s | "
                    f"face={face} | h={h:.3f} v={v:.3f}"
                )

            verdict = session.process_gaze_result(h, v, face, timestamp=frame_timestamp)
            results.append(verdict)

        elapsed = time.time() - start
        print(
            f"[GazeSession] {len(frames_b64)} frames processed locally "
            f"in {elapsed:.2f}s  (~{elapsed / len(frames_b64) * 1000:.1f}ms/frame)"
        )

    return results


class GazeSession:

    def __init__(self, session_id: str):
        self.session_id        = session_id
        self.detector          = GazeDetector()
        self.gaze_history_x    = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.envelope_x        = deque(maxlen=ENVELOPE_WINDOW)
        self.initialized       = False
        self.baseline_center_x = None
        self.baseline_std_x    = None
        self.attention_state   = "INITIALIZING"
        self.away_start_time   = None
        self.model_id          = 0
        logger.info(f"[GazeSession] GazeDetector loaded for {session_id}")

    def process_gaze_result(self,h_ratio: float,v_ratio: float,face_present: bool,timestamp: float | None = None,) -> dict:
        current_time = timestamp if timestamp is not None else time.time()

        if not face_present:
            self.attention_state = "NO_FACE"
            self.away_start_time = None
            return self.build_event("NO_FACE", 1.0, "NO_FACE", current_time)

        if not self.initialized:
            self.envelope_x.append(h_ratio)
            if len(self.envelope_x) >= ENVELOPE_WINDOW:
                self.initialized = True
            return self.build_event("INITIALIZING", 0.0, "CALIBRATING", current_time)

        if self.baseline_center_x is None:
            self.baseline_center_x = float(np.median(self.envelope_x))
            self.baseline_std_x    = float(np.std(self.envelope_x))
            logger.info(
                f"[GazeSession] Baseline set — "
                f"center={self.baseline_center_x:.3f}  "
                f"std={self.baseline_std_x:.3f}"
            )

        self.gaze_history_x.append(h_ratio)
        avg_x       = sum(self.gaze_history_x) / len(self.gaze_history_x)
        tolerance_x = max(0.07, self.baseline_std_x * 3.5)

        inside_safe_zone = abs(avg_x - self.baseline_center_x) <= tolerance_x

        if inside_safe_zone:
            self.attention_state = "ON_SCREEN"
            self.away_start_time = None
        else:
            if self.away_start_time is None:
                self.away_start_time = current_time

            elapsed = current_time - self.away_start_time

            if elapsed >= AWAY_LONG_SECONDS:
                self.attention_state = "AWAY_LONG"
            elif elapsed >= AWAY_SHORT_SECONDS:
                self.attention_state = "AWAY_SHORT"
            else:
                self.attention_state = "ON_SCREEN"

        probability = {
            "ON_SCREEN":  0.0,
            "AWAY_SHORT": 0.5,
        }.get(self.attention_state, 1.0)

        return self.build_event(
            self.attention_state, probability, self.attention_state, current_time
        )

    def build_event(self,state: str,probability: float,evidence: str,timestamp: float | None = None,) -> dict:
        self.model_id = 1
        ts = datetime.fromtimestamp(timestamp).isoformat() if timestamp else datetime.now().isoformat()
        return {
            "id":          self.model_id,
            "timestamp":   ts,
            "flag":        state,
            "probability": round(probability, 4),
            "evidence":    evidence,
        }