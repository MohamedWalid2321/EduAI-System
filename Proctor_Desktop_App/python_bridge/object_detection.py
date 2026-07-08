from ai_base import AIService
from modal_client import ModalClient
from modal_response_adapters import adapt_object_yolo_json, is_bridge_detection_event

class ObjectDetectionService(AIService):
    """Object Detection service (hosted on Modal)."""

    def __init__(self, session_id: str, config: dict):
        super().__init__("object-detection", session_id, config)
        service_config = config.get("services", {}).get("object-detection", {})
        self.endpoint_url = service_config.get("endpoint_url")
        # Project-side confidence threshold for the suspicious flag.
        # Detections below this probability are logged but do not trigger alerts.
        self.probability_threshold: float = float(
            service_config.get("probability_threshold", 0.3)
        )
        self.timeout: float = float(service_config.get("timeout_seconds", 10.0))
        modal_config = config.get("modal", {})
        self.token = modal_config.get("token_id", "test-token")
        self.client = None

    async def start(self):
        if not self.endpoint_url:
            raise ValueError("Object Detection endpoint_url not configured")
        self.client = ModalClient(self.endpoint_url, self.token, timeout=self.timeout)
        self.is_running = True

    async def stop(self):
        self.is_running = False
        if self.client is not None:
            await self.client.aclose()
        self.client = None

    async def predict(self, frame: str) -> dict:
        """Process a frame using Modal YOLO cloud inference (/analysis/object-frame)."""
        if not self.is_running:
            return self._create_error_event("SERVICE_NOT_RUNNING", "Service is not running")
        raw = await self.client.predict(self.service_name, self.session_id, frame)
        if is_bridge_detection_event(raw):
            return raw
        try:
            result = adapt_object_yolo_json(
                raw,
                self.create_detection_event,
                threshold=self.probability_threshold,
            )
            # ── DEBUG: save frame whenever YOLO fires a positive ─────────────────
            # Remove this block once the false-positive issue is diagnosed.
            if raw.get("flag") and getattr(self, "_debug_saved", 0) < 3:
                try:
                    import os, base64 as _b64
                    debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug_frames")
                    os.makedirs(debug_dir, exist_ok=True)
                    frame_data = frame.split(",", 1)[1] if (frame.startswith("data:") and ";base64," in frame) else frame
                    img_bytes = _b64.b64decode(frame_data)
                    prob = raw.get("propability", raw.get("probability", 0))
                    idx = getattr(self, "_debug_saved", 0)
                    fname = os.path.join(debug_dir, f"yolo_fp_{idx}_{prob:.2f}.jpg")
                    with open(fname, "wb") as fh:
                        fh.write(img_bytes)
                    self._debug_saved = idx + 1
                    import sys
                    print(f"[DEBUG] Saved YOLO false-positive frame -> {fname}", file=sys.stderr, flush=True)
                except Exception as _e:
                    import sys
                    print(f"[DEBUG] Frame save error: {_e}", file=sys.stderr, flush=True)
            # ── END DEBUG ─────────────────────────────────────────────────────────
            return result
        except (TypeError, ValueError, KeyError):
            return self._create_error_event(
                "BRIDGE_ERROR",
                "Unexpected Modal object-detection response shape",
            )

    def get_mock_event(self) -> dict:
        """Returns a mock object detection event."""
        return self.create_detection_event(0.88, {
            "objects": ["cell phone"],
            "count": 1,
            "suspicious": True
        })

    def _create_error_event(self, code: str, message: str) -> dict:
        import datetime
        return {
            "service": self.service_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "confidence": 0.0,
            "sessionId": self.session_id,
            "payload": {
                "status": "error",
                "code": code,
                "message": message
            }
        }
