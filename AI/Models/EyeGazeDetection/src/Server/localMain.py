import base64
import logging
import threading
import os
from dotenv import load_dotenv
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

_env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=_env_path)

MODAL_ENDPOINT        = os.getenv("MODAL_GAZE_ENDPOINT")
MODAL_CLEAR_ENDPOINT  = os.getenv("MODAL_GAZE_CLEAR_ENDPOINT")  # DELETE endpoint URL

SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW       = 90
AWAY_SHORT_SECONDS    = 3.0
AWAY_LONG_SECONDS     = 5.0

_sessions:      dict[str, "GazeSession"] = {}
_session_locks: dict[str, threading.Lock] = {}
_meta_lock =    threading.Lock()


def get_or_create_session(session_id: str) -> tuple["GazeSession", threading.Lock]:
    with _meta_lock:
        if session_id not in _sessions:
            _sessions[session_id]      = GazeSession(session_id)
            _session_locks[session_id] = threading.Lock()
            logger.info(f"[GazeSession] New session created: {session_id}")
        return _sessions[session_id], _session_locks[session_id]


def clear_session(session_id: str) -> None:
    with _meta_lock:
        _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)

    # clear the detector on Modal and Redis ownership
    try:
        if MODAL_CLEAR_ENDPOINT:
            requests.delete(
                MODAL_CLEAR_ENDPOINT,
                json={"session_id": session_id},
                timeout=5,
            )
    except Exception as exc:
        logger.warning(f"[clear_session] Could not clear Modal detector: {exc}")

    logger.info(f"[GazeSession] Session cleared: {session_id}")


def process_frames_batch(session_id: str, frames_b64: list[str]) -> list[dict]:
    session, lock = get_or_create_session(session_id)
    results = []

    with lock:
        for frame_b64 in frames_b64:
            frame = _decode_frame(frame_b64)
            if frame is None:
                logger.warning("[process_frames_batch] Could not decode a frame, skipping.")
                continue
            verdict = session.process_gaze_frame(frame)
            results.append(verdict)

    return results


def _decode_frame(frame_b64: str):
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.error(f"[_decode_frame] Failed to decode frame: {exc}")
        return None


class GazeSession:

    def __init__(self, session_id: str):
        self._session_id     = session_id       # passed to Modal every frame
        self._gaze_history_x = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self._envelope_x     = deque(maxlen=ENVELOPE_WINDOW)
        self._initialized    = False
        self._baseline_center_x = None
        self._baseline_std_x    = None
        self._attention_state   = "INITIALIZING"
        self._away_start_time   = None
        self._event_id          = 0

    def process_gaze_frame(self, frame) -> dict:
        current_time = datetime.now().timestamp()

        h_ratio, _v_ratio, face_present = self._call_modal(frame)

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
            self._baseline_center_x = float(np.median(self._envelope_x))
            self._baseline_std_x    = float(np.std(self._envelope_x))
            logger.info(
                f"[GazeSession] Baseline set — "
                f"center={self._baseline_center_x:.3f}  "
                f"std={self._baseline_std_x:.3f}"
            )

        self._gaze_history_x.append(h_ratio)
        avg_x       = sum(self._gaze_history_x) / len(self._gaze_history_x)
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

        probability = {
            "ON_SCREEN":  0.0,
            "AWAY_SHORT": 0.5,
        }.get(self._attention_state, 1.0)

        return self._build_event(
            self._attention_state, probability, self._attention_state
        )

    def _call_modal(self, frame) -> tuple[float, float, bool]:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64    = base64.b64encode(buf).decode("utf-8")

            response = requests.post(
                MODAL_ENDPOINT,
                json={
                    "session_id": self._session_id,   # Modal uses this to route to correct detector
                    "frame":      b64,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            # Modal said this user belongs to a different container — retry once
            if data.get("redirect"):
                logger.info(f"[GazeSession] Redirect received for {self._session_id}, retrying...")
                response = requests.post(
                    MODAL_ENDPOINT,
                    json={"session_id": self._session_id, "frame": b64},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

            return (
                float(data["h_ratio"]),
                float(data["v_ratio"]),
                bool(data["face_present"]),
            )

        except requests.exceptions.Timeout:
            logger.error("[GazeSession] Modal request timed out.")
            return 0.0, 0.0, False
        except requests.exceptions.RequestException as exc:
            logger.error(f"[GazeSession] Modal request failed: {exc}")
            return 0.0, 0.0, False
        except (KeyError, ValueError) as exc:
            logger.error(f"[GazeSession] Bad response from Modal: {exc}")
            return 0.0, 0.0, False

    def _build_event(self, state: str, probability: float, evidence: str) -> dict:
        self._event_id = 1    # fixed — was = 1
        return {
            "id":          self._event_id,
            "timestamp":   datetime.now().isoformat(),
            "flag":        state,
            "probability": round(probability, 4),
            "evidence":    evidence,
        }