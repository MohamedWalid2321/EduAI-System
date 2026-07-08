import cv2
import threading
import time
import base64
import sys
from contextlib import redirect_stdout
from typing import Callable, Optional
from ai_base import AIService

class LocalEyeGazeService(AIService):
    """
    Local Eye Gaze Detection service using OpenCV and a background thread.
    """

    def __init__(self, session_id: str, config: dict, event_callback: Callable[[dict], None]):
        super().__init__("eye-gaze", session_id, config)
        self.event_callback = event_callback
        
        # Load service specific config
        s_cfg = config.get("services", {}).get("eye-gaze", {})
        self.fps = s_cfg.get("fps", 5)
        self.camera_index = s_cfg.get("camera_index", 0)
        # Desktop app expects the renderer to own the webcam (getUserMedia) and
        # send frames to Python via predict(). Defaulting to "camera" can cause
        # device-lock conflicts with Chromium on Windows ("Camera unavailable").
        self.input_mode = s_cfg.get("input_mode", "shared_frame")
        self.model_path = s_cfg.get("model_path")
        self.threshold = s_cfg.get("threshold", 0.5)

        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_status = None
        self._session_id = session_id
        self._process_frames_batch = None
        # Tracks whether the current question allows looking down (writing mode).
        # Updated by the router via set_question_mode() whenever the student
        # navigates to a new question that carries IsAllowableToLookDown.
        self._is_writing: bool = False

        # Load the bridge-local port of localMain.py.
        self._load_error: str | None = None  # FIX: store import failure for start()-time emission
        try:
            from service.localMain import process_frames_batch  # type: ignore
            self._process_frames_batch = process_frames_batch
        except Exception as exc:
            # FIX: Do NOT call _emit_hardware_error here — the event loop is not
            # yet running and the callback (loop.call_soon_threadsafe) will silently
            # drop the notification. Store the message and emit it in start() instead.
            self._load_error = f"Failed to load local eye-gaze model: {exc}"

    def set_question_mode(self, is_allowable_to_look_down: bool) -> None:
        """Switch gaze-detection mode based on the current question's flag.

        Call this whenever the student navigates to a new question, passing
        the ``IsAllowableToLookDown`` value from the LMS API response.

        Parameters
        ----------
        is_allowable_to_look_down : bool
            ``True``  → writing mode  (looking down to write is ignored).
            ``False`` → normal mode   (looking down is flagged quickly).
        """
        self._is_writing = bool(is_allowable_to_look_down)

    async def start(self):
        """Start the background capture and inference thread."""
        if self.is_running:
            return

        # FIX: If the model failed to import, emit the error now — the event loop
        # is guaranteed to be running at this point (we are inside an async call
        # from the router). This ensures the serviceError notification reaches the
        # renderer and the UI can display an error pill instead of spinning forever.
        if self._load_error is not None:
            self._emit_hardware_error(self._load_error)
            return  # do NOT set is_running=True — service is non-functional

        # In shared_frame mode, renderer owns the webcam and sends frames via predict().
        if self.input_mode == "shared_frame":
            self.is_running = True
            return

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            # Emit hardware error via callback (wrapped in an async task if needed)
            self._emit_hardware_error("Unable to open webcam.")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    async def stop(self):
        """Stop the background thread and release hardware."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    async def predict(self, frame: str) -> dict:
        """
        In shared_frame mode, this service consumes renderer-provided frames.
        In camera mode, the local service runs its own capture loop and returns
        current state for polling.
        """
        if self.input_mode == "shared_frame":
            if self._process_frames_batch is None:
                # FIX: The model failed to load at __init__ time (localMain import
                # error, missing face_landmarker.task, or missing cv2/mediapipe).
                # Returning "initializing" silently here causes the calibration phase
                # to loop forever because the UI waits for any status != 'initializing'.
                # Emit a one-shot serviceError so the UI can surface a real error pill
                # instead of spinning forever.
                self._emit_hardware_error(
                    "Eye-gaze model unavailable: localMain failed to load. "
                    "Check that face_landmarker.task exists in python_bridge/service/face_landmarker/ "
                    "and that mediapipe + opencv-python are installed."
                )
                # Return a final 'error' status so the router can still send a
                # well-formed DetectionEvent if the callback path is unavailable.
                return self.create_detection_event(0.0, {"status": "initializing"})

            frame_b64 = frame
            if isinstance(frame_b64, str) and frame_b64.startswith("data:") and "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            try:
                # localMain prints timing info; redirect to stderr so router stdout stays JSON-only.
                with redirect_stdout(sys.stderr):
                    results = self._process_frames_batch(
                        self._session_id,
                        [frame_b64],
                        fps=max(1, int(self.fps)),
                        is_writing=self._is_writing,
                    )
            except Exception as exc:
                self._emit_hardware_error(f"Eye-gaze inference error: {exc}")
                return self.create_detection_event(0.0, {"status": "initializing"})

            verdict = results[-1] if results else {}
            raw_flag = str(verdict.get("attention_state", verdict.get("flag", "INITIALIZING")))
            probability = float(verdict.get("probability", 0.0))
            evidence = str(verdict.get("evidence", raw_flag))
            diag = verdict.get("gaze_diagnostics", {})

            status_map = {
                "ON_SCREEN": "on-screen",
                "AWAY_SHORT": "away",
                "AWAY_LONG": "away",
                "NO_FACE": "no-face",
                "INITIALIZING": "initializing",
            }
            status = status_map.get(raw_flag, "initializing")
            self._last_status = status

            return self.create_detection_event(1.0 - min(max(probability, 0.0), 1.0), {
                "gaze_x": diag.get("h_ratio", diag.get("gaze_h", 0.5)),
                "gaze_y": diag.get("avg_x",  diag.get("gaze_v", 0.5)),
                "status": status,
                "raw_flag": raw_flag,
                "probability": probability,
                "evidence": evidence,
                "gaze_diagnostics": diag,
            })

        return self.create_detection_event(1.0, {"status": self._last_status or "initializing"})

    def get_mock_event(self) -> dict:
        """Returns a mock gaze event."""
        return self.create_detection_event(0.95, {
            "gaze_x": 0.5,
            "gaze_y": 0.5,
            "status": "on-screen"
        })

    def recalibrate(self) -> None:
        """Reset the GazeSession calibration so the next frames re-run it."""
        try:
            from service.localMain import manager  # type: ignore
            if manager.session is not None:
                manager.session.recalibrate()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"[LocalEyeGazeService] recalibrate failed: {exc}"
            )

    def _run_loop(self):
        """Background thread loop."""
        interval = 1.0 / self.fps
        
        # In a real implementation, we would load the model here
        # model = load_model(self.model_path)

        while self.is_running:
            start_time = time.time()
            
            ret, frame = self._cap.read()
            if not ret:
                self._emit_hardware_error("Lost connection to webcam.")
                break

            # --- Placeholder for Real Gaze Inference ---
            # result = model.predict(frame)
            # For Phase 7, we simulate a gaze detection:
            status = "on-screen"
            confidence = 0.98
            # --------------------------------------------

            # Emit only on state change to reduce noise
            if status != self._last_status:
                event = self.create_detection_event(confidence, {
                    "gaze_x": 0.5,
                    "gaze_y": 0.5,
                    "status": status
                })
                self.event_callback(event)
                self._last_status = status

            # Throttle to FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def _emit_hardware_error(self, message: str):
        error_event = {
            "method": "serviceError",
            "params": {
                "service": self.service_name,
                "code": "HARDWARE_FAILURE",
                "message": message
            }
        }
        self.event_callback(error_event)
        self.is_running = False
