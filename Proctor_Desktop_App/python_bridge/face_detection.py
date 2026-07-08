"""
FaceDetectionService — face presence check via Modal /analysis/face-detection-file.

Used internally by FaceRecognitionService as Step 1 of the dual-frequency pipeline,
and available as an independent AIService if needed in the future.
"""

import datetime
from ai_base import AIService
from modal_client import ModalClient


class FaceDetectionService(AIService):
    """
    Face Detection service — checks for face presence in a single frame.

    Wraps the Modal /analysis/face-detection-file endpoint.

    Threshold-based decision
    ~~~~~~~~~~~~~~~~~~~~~~~~
    When num_faces == 0 the API returns a probability (~0.85) representing its
    confidence that no face is present. If detect_prob < face_detect_threshold the
    reading is treated as uncertain and face_detected is set to True (benefit of the
    doubt — avoids false alerts from brief occlusions or blurry frames).
    When num_faces > 0, face_detected is always True regardless of probability.

    Config keys read from ``services.face-recognition``:
        face_detect_endpoint_url  — Modal URL for /analysis/face-detection-file
        face_detect_threshold     — minimum confidence to accept "no face" verdict (default 0.7)
        timeout_seconds           — HTTP timeout (default 15.0)
    """

    def __init__(self, session_id: str, config: dict):
        super().__init__("face-detection", session_id, config)
        # Face detection shares the face-recognition config block (same Modal deployment).
        service_config = config.get("services", {}).get("face-recognition", {})
        self.face_detect_endpoint_url: str = service_config.get("face_detect_endpoint_url", "")
        # Minimum confidence required to accept a "no face" verdict.
        # The API returns a fixed probability of 0.85 for no-face readings.
        # Default 0.7 (below 0.85) accepts all such results — same as the old binary logic.
        # Raise above 0.85 to require higher certainty before treating a frame as face-absent.
        self.face_detect_threshold: float = float(
            service_config.get("face_detect_threshold", 0.7)
        )
        self.timeout: float = float(service_config.get("timeout_seconds", 15.0))
        modal_config = config.get("modal", {})
        self.token: str = modal_config.get("token_id", "test-token")
        self.client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not self.face_detect_endpoint_url:
            raise ValueError("Face Detection face_detect_endpoint_url not configured")
        self.client = ModalClient(
            self.face_detect_endpoint_url,
            self.token,
            timeout=self.timeout,
            face_detect_url=self.face_detect_endpoint_url,
        )
        self.is_running = True

    async def stop(self):
        self.is_running = False
        if self.client is not None:
            await self.client.aclose()
        self.client = None

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    async def detect(self, frame: str) -> dict:
        """
        Run face detection on a single frame and return a plain result dict.

        This is the primary method used by FaceRecognitionService as its Step 1
        gate. It returns a plain dict (not a full DetectionEvent) so the caller
        can merge the result into its own event payload.

        Return shape::

            {
                "ok":                True,
                "face_detected":     bool,   # threshold-gated presence flag
                "faces_count":       int,    # raw num_faces from the API
                "detection_message": str,    # API evidence string (for logging)
                "detect_prob":       float,  # raw probability from the API [0, 1]
            }

        On hard error::

            {
                "ok":    False,
                "error": {"code": str, "message": str},
                # safe defaults for all other fields:
                "face_detected": False, "faces_count": 0,
                "detection_message": "", "detect_prob": 0.0,
            }
        """
        if not self.is_running or self.client is None:
            return {
                "ok": False,
                "face_detected": False,
                "faces_count": 0,
                "detection_message": "",
                "detect_prob": 0.0,
                "error": {"code": "SERVICE_NOT_RUNNING",
                          "message": "Face detection service is not running."},
            }

        raw = await self.client.face_detect(self.session_id, frame)

        # Hard error from the client (non-200 response, timeout, unknown exception).
        if raw.get("ok") is False:
            return {
                "ok": False,
                "face_detected": False,
                "faces_count": 0,
                "detection_message": "",
                "detect_prob": 0.0,
                "error": raw.get("error", {"code": "FACE_DETECT_ERROR",
                                           "message": "Face detection request failed."}),
            }

        num_faces: int      = int(raw.get("num_faces", 0) or 0)
        detect_prob: float  = float(raw.get("probability") or 0.0)
        # "One face detected" / "no_face_detected" / "multiple_faces" — from the API.
        detection_message: str = (raw.get("evidence") or "").strip()

        if num_faces > 0:
            face_detected = True
        else:
            # Accept "no face" only when the detection confidence meets the threshold.
            # Below threshold → uncertain reading → give student benefit of the doubt.
            face_detected = detect_prob < self.face_detect_threshold

        return {
            "ok": True,
            "face_detected": face_detected,
            "faces_count": num_faces,
            "detection_message": detection_message,
            "detect_prob": detect_prob,
        }

    # ------------------------------------------------------------------
    # AIService interface (standalone use)
    # ------------------------------------------------------------------

    async def predict(self, frame: str) -> dict:
        """
        AIService interface — wraps detect() into a full DetectionEvent.
        Used when FaceDetectionService is operated as an independent service.
        """
        result = await self.detect(frame)
        if not result.get("ok", True):
            return self._create_error_event(
                result.get("error", {}).get("code", "FACE_DETECT_ERROR"),
                result.get("error", {}).get("message", "Face detection failed."),
            )
        return self.create_detection_event(result["detect_prob"], {
            "face_detected":     result["face_detected"],
            "faces_count":       result["faces_count"],
            "detection_message": result["detection_message"],
        })

    def get_mock_event(self) -> dict:
        return self.create_detection_event(0.85, {
            "face_detected":     True,
            "faces_count":       1,
            "detection_message": "One face detected",
        })

    def _create_error_event(self, code: str, message: str) -> dict:
        return {
            "service":    self.service_name,
            "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "confidence": 0.0,
            "sessionId":  self.session_id,
            "payload": {
                "status":  "error",
                "code":    code,
                "message": message,
            },
        }
