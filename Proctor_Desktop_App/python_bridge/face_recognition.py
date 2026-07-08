import time
import datetime
import httpx
from ai_base import AIService
from modal_client import ModalClient
from modal_response_adapters import adapt_face_modal_json, is_bridge_detection_event

# How often (in seconds) full identity recognition runs.
# Face detection runs independently via FaceDetectionService (face_detection.py)
# at the frontend's 1 s polling cadence.
FACE_RECOGNITION_INTERVAL = 5.0


class FaceRecognitionService(AIService):
    """
    Face Recognition service (hosted on Modal).

    Single-responsibility: identity verification via /analysis/verify-file.
    Runs at most once every FACE_RECOGNITION_INTERVAL seconds (5 s).

    Face *presence* detection is handled independently by FaceDetectionService
    (face_detection.py), which the frontend polls every 1 second separately.
    Both services send events to the orchestrator; the orchestrator applies
    deduplication when they overlap (every 5th second).
    """

    def __init__(self, session_id: str, config: dict):
        super().__init__("face-recognition", session_id, config)
        service_config = config.get("services", {}).get("face-recognition", {})
        self.endpoint_url          = service_config.get("endpoint_url")
        self.enroll_endpoint_url   = service_config.get("enroll_endpoint_url", "")
        self.unenroll_endpoint_url = service_config.get("unenroll_endpoint_url", "")
        self.face_detect_endpoint_url = service_config.get("face_detect_endpoint_url", "")
        self.face_frame_endpoint_url  = service_config.get("face_frame_endpoint_url", "")
        # Project-side probability threshold for the match/mismatch decision during
        # live proctoring (verify). Applies only to 'Authorised person verified' and
        # 'Face does not match' outcomes. Default 0.5 matches the server-side threshold.
        self.probability_threshold: float = float(
            service_config.get("probability_threshold", 0.5)
        )
        # Separate threshold used ONLY during enrollment identity confirmation.
        # Compared against the similarity returned by /analysis/face-frame.
        self.enrollment_similarity_threshold: float = float(
            service_config.get("enrollment_similarity_threshold", 0.5)
        )
        self.timeout: float = float(service_config.get("timeout_seconds", 15.0))
        self.enroll_timeout: float = float(service_config.get("enroll_timeout_seconds", 60.0))
        modal_config = config.get("modal", {})
        self.token = modal_config.get("token_id", "test-token")
        self.client = None

        # Recognition state
        self._last_recognition_time: float = 0.0
        self._last_recognition_payload: dict = {"is_matched": False, "faces_count": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not self.endpoint_url:
            raise ValueError("Face Recognition endpoint_url not configured")
        self.client = ModalClient(
            self.endpoint_url,
            self.token,
            timeout=self.timeout,
            enroll_url=self.enroll_endpoint_url,
            unenroll_url=self.unenroll_endpoint_url,
            face_detect_url=self.face_detect_endpoint_url,
            face_frame_url=self.face_frame_endpoint_url,
        )
        self.is_running = True

        # Pre-warm Modal container in the background so it doesn't block startup
        import asyncio
        asyncio.create_task(self.client.warmup())

    async def stop(self):
        self.is_running = False
        if self.client is not None:
            await self.client.aclose()
        self.client = None

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    async def enroll(self, frame: str, profile_picture_url: str | None = None) -> dict:
        """
        Enroll a reference image for this session.

        Identity-confirmation gate (replaces the old face-detection gate):
          1. If ``profile_picture_url`` is provided, fetch the official image and
             call POST /analysis/face-frame to compare the live capture against it.
             The similarity must exceed ``enrollment_similarity_threshold`` (from
             config.json) and the API must report an authorised match.
          2. Only when identity is confirmed, store the embedding via
             POST /analysis/enroll-file.

        Returns: {ok: True} on success or {ok: False, error: {code, message}}.
        """
        if not self.is_running or self.client is None:
            return {"ok": False, "error": {"code": "SERVICE_NOT_RUNNING",
                    "message": "Face recognition service is not running."}}

        # ------------------------------------------------------------------
        # Step 1 — Identity confirmation via /analysis/face-frame
        # ------------------------------------------------------------------
        probability = None
        evidence = None
        
        if profile_picture_url:
            # Fetch the reference image bytes from the CDN/LMS URL.
            ref_bytes: bytes | None = None
            ref_mime = "image/jpeg"
            try:
                async with httpx.AsyncClient(timeout=self.enroll_timeout) as http:
                    ref_response = await http.get(profile_picture_url)
                    if ref_response.status_code == 200:
                        ref_bytes = ref_response.content
                        content_type = ref_response.headers.get("content-type", "image/jpeg")
                        ref_mime = content_type.split(";")[0].strip() or "image/jpeg"
                    else:
                        return {
                            "ok": False,
                            "error": {
                                "code": "REFERENCE_FETCH_FAILED",
                                "message": (
                                    f"Could not download official profile image "
                                    f"(HTTP {ref_response.status_code})."
                                ),
                            },
                        }
            except httpx.TimeoutException:
                return {"ok": False, "error": {"code": "TIMEOUT",
                        "message": "Timed out while fetching official profile image."}}
            except Exception as exc:
                return {"ok": False, "error": {"code": "REFERENCE_FETCH_FAILED",
                        "message": str(exc)}}

            # Compare live frame vs official reference image.
            compare_result = await self.client.face_frame_compare(
                session_id=self.session_id,
                live_frame=frame,
                reference_image_bytes=ref_bytes,
                reference_mime=ref_mime,
                timeout=self.enroll_timeout,
            )

            if not compare_result.get("ok"):
                # Network/HTTP error during comparison
                return {"ok": False, "error": compare_result.get("error",
                        {"code": "FACE_FRAME_ERROR",
                         "message": "Failed to compare face against official record."})}

            probability = compare_result.get("probability", 0.0)
            evidence    = compare_result.get("evidence", "")

            # Evidence-based checks (no-face / multiple-faces / mismatch / spoof)
            if "No face detected" in evidence:
                return {"ok": False, "error": {
                    "code":    "NO_FACE_DETECTED",
                    "message": "No face detected — adjust your position and lighting.",
                }}
            if "Multiple faces" in evidence:
                return {"ok": False, "error": {
                    "code":    "MULTIPLE_FACES",
                    "message": "Multiple faces detected — ensure only you are visible.",
                }}
            if "Spoof detected" in evidence:
                return {"ok": False, "error": {
                    "code":    "SPOOF_DETECTED",
                    "message": "Liveness check failed — please use a live webcam.",
                }}

            # Threshold check against enrollment_similarity_threshold from config
            if probability < self.enrollment_similarity_threshold:
                return {"ok": False, "error": {
                    "code":    "IDENTITY_MISMATCH",
                    "message": (
                        "Your live photo does not match your official student record. "
                        f"Similarity: {probability * 100:.1f}% "
                        f"(required ≥ {self.enrollment_similarity_threshold * 100:.0f}%)."
                    ),
                }}

        # ------------------------------------------------------------------
        # Step 2 — Persist the embedding via /analysis/enroll-file
        # ------------------------------------------------------------------
        enroll_result = await self.client.enroll(self.session_id, frame, timeout=self.enroll_timeout)
        # Network/HTTP-level failure (non-200 status code).
        if enroll_result.get("ok") is False:
            return {"ok": False, "error": enroll_result.get("error",
                    {"code": "ENROLLMENT_FAILED", "message": "Enrollment failed."})}
        # API-level failure: 200 OK but success:false in the response body.
        if not enroll_result.get("success", True):
            return {"ok": False, "error": {"code": "ENROLLMENT_FAILED",
                    "message": "Enrollment rejected by the AI service."}}

        # Reset recognition timer so the first verify() after enrollment runs immediately.
        self._last_recognition_time = 0.0
        self._last_recognition_payload = {"is_matched": False, "faces_count": 0}
        
        out = {"ok": True}
        if probability is not None:
            out["probability"] = probability
        if evidence is not None:
            out["evidence"] = evidence
            
        return out

    async def unenroll(self) -> None:
        """Remove the stored embedding for this session (fire-and-forget)."""
        if self.client is not None:
            try:
                await self.client.unenroll(self.session_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Identity recognition (5 s gate)
    # ------------------------------------------------------------------

    async def predict(self, frame: str) -> dict:
        """
        Run identity recognition at most once every FACE_RECOGNITION_INTERVAL seconds.

        Between recognition passes, returns the cached result so the orchestrator
        always receives a complete payload without waiting for the full 5 s interval
        to expire.

        The face-detection step is intentionally NOT embedded here — that is handled
        independently by FaceDetectionService. The orchestrator deduplicates risk
        contributions when both events arrive in the same window.
        """
        if not self.is_running or self.client is None:
            return self._create_error_event("SERVICE_NOT_RUNNING", "Service is not running")

        now = time.monotonic()

        if (now - self._last_recognition_time) >= FACE_RECOGNITION_INTERVAL:
            raw = await self.client.predict(self.service_name, self.session_id, frame)

            if is_bridge_detection_event(raw):
                payload = raw.get("payload", {})
                self._last_recognition_payload = {
                    "is_matched":  payload.get("is_matched", False),
                    "faces_count": payload.get("faces_count", 0),
                }
                self._last_recognition_time = now
                raw["payload"]["recognition_ran"] = True
                return raw

            try:
                event = adapt_face_modal_json(
                    raw,
                    self.create_detection_event,
                    threshold=self.probability_threshold,
                )
                self._last_recognition_payload = {
                    "is_matched":  event["payload"].get("is_matched", False),
                    "faces_count": event["payload"].get("faces_count", 0),
                }
                self._last_recognition_time = now
                event["payload"]["recognition_ran"] = True
                return event
            except (TypeError, ValueError, KeyError):
                return self._create_error_event(
                    "BRIDGE_ERROR",
                    "Unexpected Modal face response shape",
                )

        # Between recognition passes — return cached result.
        cached = self._last_recognition_payload
        confidence = 0.9 if cached.get("is_matched") else 0.1
        return self.create_detection_event(confidence, {
            "is_matched":      cached.get("is_matched", False),
            "faces_count":     cached.get("faces_count", 0),
            "recognition_ran": False,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_mock_event(self) -> dict:
        return self.create_detection_event(0.99, {
            "is_matched":      True,
            "student_id":      "a3f1c2d4-0000-0000-0000-000000000000",
            "faces_count":     1,
            "recognition_ran": True,
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
