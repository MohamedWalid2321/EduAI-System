from __future__ import annotations

import base64
import logging
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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
MODAL_CLEAR_ENDPOINT  = os.getenv("MODAL_GAZE_CLEAR_ENDPOINT")

SMOOTHING_BUFFER_SIZE = 3
ENVELOPE_WINDOW       = 90
AWAY_SHORT_SECONDS    = 3.0
AWAY_LONG_SECONDS     = 5.0

# Max concurrent Modal calls
MAX_WORKERS = 30

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

    with lock:
        # Step 1 — send ALL frames to Modal concurrently
        start         = time.time()
        modal_results = _call_modal_concurrent(session_id, frames_b64)
        elapsed       = time.time() - start
        print(f"[GazeSession] {len(frames_b64)} concurrent Modal calls finished in {elapsed:.2f}s")

        # Step 2 — feed each result into the stateful GazeSession in ORDER
        results = []
        for r in modal_results:
            if r is not None:
                verdict = session.process_gaze_result(
                    float(r["h_ratio"]),
                    float(r["v_ratio"]),
                    bool(r["face_present"]),
                )
            else:
                verdict = session.process_gaze_result(0.0, 0.0, False)
            results.append(verdict)

    return results


def _call_single_frame(session_id: str, index: int, frame_b64: str) -> tuple[int, Optional[dict]]:
    """Send one frame to Modal. Returns (index, result) to preserve order."""
    try:
        # time to send the request , modal is not included
        # upload frame to Modal      (network)
        # Modal queues the request   (waiting)
        # Modal decodes frame        (compute)
        # Modal runs MediaPipe       (compute)
        # Modal sends result back    (network)
        request_start = time.time()
        response = requests.post(
            MODAL_ENDPOINT,
            json={
                "session_id": session_id,
                "frame":      frame_b64,
            },
            timeout=30,
        )
        request_end = time.time()
        
        response.raise_for_status()

        data = response.json()


        
        print(
            f"  Frame {index:02d} | "
            f"request={request_end - request_start:.2f}s | "
        )

        return index, {
            "h_ratio":      float(data["h_ratio"]),
            "v_ratio":      float(data["v_ratio"]),
            "face_present": bool(data["face_present"]),
        }
    except requests.exceptions.Timeout:
        logger.error(f"[_call_single_frame] Frame {index} timed out.")
        return index, None
    except requests.exceptions.RequestException as exc:
        logger.error(f"[_call_single_frame] Frame {index} request failed: {exc}")
        return index, None
    except (KeyError, ValueError) as exc:
        logger.error(f"[_call_single_frame] Frame {index} bad response: {exc}")
        return index, None


def _call_modal_concurrent(session_id: str, frames_b64: list[str]) -> list[Optional[dict]]:
    """
    Fire all frames to Modal simultaneously using a thread pool.
    Returns results in the original frame order.
    """
    num_frames = len(frames_b64)
    ordered    = [None] * num_frames

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, num_frames)) as executor:
        futures = {
            executor.submit(_call_single_frame, session_id, i, frame_b64): i
            for i, frame_b64 in enumerate(frames_b64)
        }
        for future in as_completed(futures):
            index, result = future.result()
            ordered[index] = result

    return ordered




class GazeSession:

    def __init__(self, session_id: str):
        self._session_id        = session_id
        self._gaze_history_x    = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self._envelope_x        = deque(maxlen=ENVELOPE_WINDOW)
        self._initialized       = False
        self._baseline_center_x = None
        self._baseline_std_x    = None
        self._attention_state   = "INITIALIZING"
        self._away_start_time   = None
        self._event_id          = 0

    def process_gaze_result(self, h_ratio: float, v_ratio: float, face_present: bool) -> dict:
        """Apply calibration and attention logic to a single gaze result."""
        current_time = datetime.now().timestamp()

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

    def _build_event(self, state: str, probability: float, evidence: str) -> dict:
        self._event_id = 1
        return {
            "id":          self._event_id,
            "timestamp":   datetime.now().isoformat(),
            "flag":        state,
            "probability": round(probability, 4),
            "evidence":    evidence,
        }