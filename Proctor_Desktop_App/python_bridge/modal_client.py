import httpx
import datetime
import base64
import binascii
import json
import os
from typing import Dict, Any, Tuple

class ModalClient:
    """Async client for calling Modal web endpoints."""

    def __init__(self, endpoint_url: str, token: str, timeout: float = 30.0,
                 enroll_url: str = "", unenroll_url: str = "",
                 face_detect_url: str = "", face_frame_url: str = ""):
        self.endpoint_url = endpoint_url
        self.token = token
        self.timeout = timeout
        self._enroll_url = enroll_url or endpoint_url
        self._unenroll_url = unenroll_url or endpoint_url
        self._face_detect_url = face_detect_url or endpoint_url
        self._face_frame_url = face_frame_url or endpoint_url
        # Persistent HTTP client — reused across all requests to avoid the
        # TCP + TLS handshake overhead of creating a new connection per call.
        # Lazily initialised on the first request; closed explicitly via aclose().
        self._http_client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared persistent HTTP client, creating it on first call."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                # Keep connections alive for up to 60 s of idle time.
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=60,
                ),
            )
        return self._http_client

    async def aclose(self) -> None:
        """Close the persistent HTTP client and release the underlying TCP connection.
        Call this from the owning service's stop() method."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None

    async def warmup(self) -> None:
        """
        Send a lightweight GET request to the Modal app's health or root endpoint to wake it up.
        This triggers container cold start without running heavy ML inference.
        """
        try:
            base_url = "/".join(self.endpoint_url.split("/")[:3])
            health_url = f"{base_url}/health"
            client = self._get_client()
            response = await client.get(health_url)
            if response.status_code == 404:
                # Fallback to root if /health is not defined
                await client.get(base_url)
        except Exception:
            pass  # Fire-and-forget


    async def predict(self, service_name: str, session_id: str, frame: str) -> Dict[str, Any]:
        """
        Sends a frame to the Modal endpoint and returns a DetectionEvent.
        Handles cold starts and timeouts.
        """
        try:
            client = self._get_client()
            response = await self._send_request(client, service_name, session_id, frame)

            if response.status_code == 200:
                body = response.json()
                # ── TEMP DIAGNOSTIC LOG ──────────────────────────────────────
                # Logs the raw Modal JSON for object-detection to
                # python_bridge/tmp/modal_raw.ndjson so you can inspect
                # exactly what the server returns before any adapter runs.
                # Safe to delete: does not affect sessions/ or any production path.
                if service_name == "object-detection":
                    self._log_raw_response(body)
                # ── END TEMP LOG ─────────────────────────────────────────────
                return body
            elif response.status_code == 503:
                return self._create_error_event(
                    service_name, session_id, "SERVICE_UNAVAILABLE",
                    "Modal container is warming up. Retrying..."
                )
            else:
                return self._create_error_event(
                    service_name, session_id, "BRIDGE_ERROR",
                    f"Modal returned unexpected status: {response.status_code}"
                )
        except httpx.TimeoutException:
            return self._create_error_event(
                service_name, session_id, "TIMEOUT",
                "Request to Modal timed out (cold start threshold exceeded)."
            )
        except Exception as e:
            return self._create_error_event(
                service_name, session_id, "UNKNOWN_ERROR", str(e)
            )

    def _log_raw_response(self, body: Dict[str, Any]) -> None:
        """
        Append the raw Modal JSON response to a temp NDJSON log.
        File: python_bridge/tmp/modal_raw.ndjson
        Each line is a JSON object with a 'logged_at' timestamp + the full body.
        TEMP DIAGNOSTIC ONLY — delete this method and its call once resolved.
        """
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "tmp")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "modal_raw.ndjson")
            entry = {"logged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **body}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # never crash the bridge over a debug log

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        service_name: str,
        session_id: str,
        frame: str,
    ) -> httpx.Response:
        endpoint_lower = self.endpoint_url.lower()

        if "/analysis/detect_objects" in endpoint_lower:
            image_bytes, mime = self._decode_frame_to_image(frame)
            files = {"file": ("frame.jpg", image_bytes, mime)}
            return await client.post(self.endpoint_url, files=files)

        if "/analysis/object-frame" in endpoint_lower:
            # YOLO single-frame endpoint — multipart field name is 'image'
            image_bytes, mime = self._decode_frame_to_image(frame)
            files = {"image": ("frame.jpg", image_bytes, mime)}
            return await client.post(self.endpoint_url, files=files)

        if "/analysis/verify-file" in endpoint_lower:
            image_bytes, mime = self._decode_frame_to_image(frame)
            # data = {"session_id": "test_session_001"}
            data = {"session_id": session_id}
            files = {"frame": ("frame.jpg", image_bytes, mime)}
            return await client.post(self.endpoint_url, data=data, files=files)

        # No known endpoint pattern matched — fail loudly rather than sending a
        # malformed payload that no active Modal route accepts.
        raise ValueError(
            f"No matching request format for endpoint: {self.endpoint_url!r}. "
            "Verify that config endpoint_url contains a recognised Modal route "
            "(/analysis/object-frame, /analysis/detect_objects, or /analysis/verify-file)."
        )

    def _decode_frame_to_image(self, frame: str) -> Tuple[bytes, str]:
        """Decode a base64 frame (or data URL) into image bytes for multipart upload."""
        if frame == "WARMUP":
            # 1x1 JPEG used for endpoint warmup on multipart-only routes.
            return (
                base64.b64decode(
                    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBUQEA8QFRUVFRUVFRUVFRUVFRUXFxUWFhUV"
                    "FRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGhAQGi0fHyUtLS0tLS0tLS0tLS0t"
                    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAXAAEBAQEA"
                    "AAAAAAAAAAAAAAABAgME/8QAFhEBAQEAAAAAAAAAAAAAAAAAAAER/8QAFQEBAQAAAAAAAAAAAAAAAAAA"
                    "AgP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCjAAH/2Q=="
                ),
                "image/jpeg",
            )

        frame_data = frame
        mime = "image/jpeg"

        if frame.startswith("data:") and ";base64," in frame:
            header, frame_data = frame.split(",", 1)
            mime = header[5:].split(";", 1)[0] or "image/jpeg"

        try:
            return base64.b64decode(frame_data, validate=True), mime
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Frame must be a valid base64 image payload") from exc

    async def face_frame_compare(
        self,
        session_id: str,
        live_frame: str,
        reference_image_bytes: bytes,
        reference_mime: str = "image/jpeg",
        timeout: float = None,
    ) -> Dict[str, Any]:
        """
        Compare a live webcam frame against an official reference image via
        POST /analysis/face-frame (multipart/form-data).

        Parameters
        ----------
        session_id : str
            Optional session_id forwarded for server-side embedding caching.
        live_frame : str
            Base64 data-URL or raw base64 JPEG of the live webcam capture.
        reference_image_bytes : bytes
            Raw bytes of the authorised reference image (e.g. fetched from
            profilePictureUrl).
        reference_mime : str
            MIME type of the reference image, default 'image/jpeg'.

        Returns
        -------
        dict with keys:
          ok : bool
          probability : float   (normalised 0.0–1.0)
          evidence : str        (raw 'evidence' string from Modal)
          error : dict | None   (code + message when ok is False)
        """
        try:
            req_timeout = timeout if timeout is not None else self.timeout
            async with httpx.AsyncClient(timeout=req_timeout) as client:
                live_bytes, live_mime = self._decode_frame_to_image(live_frame)
                data = {"session_id": session_id}
                files = {
                    "frame":     ("frame.jpg",     live_bytes,            live_mime),
                    "reference": ("reference.jpg", reference_image_bytes, reference_mime),
                }
                response = await client.post(self._face_frame_url, data=data, files=files)

                if response.status_code == 200:
                    body = response.json()
                    # The /analysis/face-frame response is wrapped in face_recognition key
                    face_data = body.get("face_recognition", body)
                    raw_prob = face_data.get("probability", "0.0%")
                    # probability comes back as a percentage string, e.g. "98.21%"
                    try:
                        prob_float = float(str(raw_prob).replace("%", "").strip()) / 100.0
                    except (ValueError, TypeError):
                        prob_float = 0.0
                    evidence = face_data.get("evidence", "")
                    return {
                        "ok":          True,
                        "probability": prob_float,
                        "evidence":    evidence,
                    }

                # Non-200 — try to extract a readable detail
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                return {
                    "ok":    False,
                    "error": {
                        "code":    "FACE_FRAME_ERROR",
                        "message": detail or f"face-frame returned {response.status_code}",
                    },
                }

        except httpx.TimeoutException:
            return {"ok": False, "error": {"code": "TIMEOUT",
                    "message": "face-frame comparison request timed out."}}
        except ValueError as e:
            return {"ok": False, "error": {"code": "INVALID_FRAME", "message": str(e)}}
        except Exception as e:
            return {"ok": False, "error": {"code": "UNKNOWN_ERROR", "message": str(e)}}

    async def face_detect(self, session_id: str, frame: str) -> Dict[str, Any]:
        """
        Check for a face in a single frame via /analysis/face-detection-file.
        Returns the raw Modal JSON response.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                image_bytes, mime = self._decode_frame_to_image(frame)
                data = {"session_id": session_id}
                files = {"frame": ("frame.jpg", image_bytes, mime)}
                response = await client.post(self._face_detect_url, data=data, files=files)
                if response.status_code == 200:
                    return response.json()
                # Extract the human-readable detail from 422 bodies (e.g. "No face detected").
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                return {"ok": False, "error": {"code": "FACE_DETECT_ERROR",
                        "message": detail or f"Modal face-detect returned {response.status_code}"}}
        except httpx.TimeoutException:
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Face-detect request timed out."}}
        except Exception as e:
            return {"ok": False, "error": {"code": "UNKNOWN_ERROR", "message": str(e)}}

    async def enroll(self, session_id: str, frame: str, timeout: float = None) -> Dict[str, Any]:
        """
        Enroll a reference image via /analysis/enroll-file.
        'frame' is a base64 data URL or raw base64 JPEG.
        Returns the raw Modal JSON response.
        """
        try:
            req_timeout = timeout if timeout is not None else self.timeout
            async with httpx.AsyncClient(timeout=req_timeout) as client:
                image_bytes, mime = self._decode_frame_to_image(frame)
                data = {"session_id": session_id}
                files = {"references": ("reference.jpg", image_bytes, mime)}
                response = await client.post(self._enroll_url, data=data, files=files)
                if response.status_code == 200:
                    return response.json()
                # Extract the human-readable detail from 422 bodies
                # (e.g. "No face detected in reference image 0", "Multiple faces in reference image 0").
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                return {"ok": False, "error": {"code": "ENROLLMENT_FAILED",
                        "message": detail or f"Modal enroll returned {response.status_code}"}}
        except httpx.TimeoutException:
            return {"ok": False, "error": {"code": "TIMEOUT", "message": "Enrollment request timed out."}}
        except Exception as e:
            return {"ok": False, "error": {"code": "UNKNOWN_ERROR", "message": str(e)}}

    async def unenroll(self, session_id: str) -> None:
        """
        Remove the stored embedding via /analysis/unenroll (fire-and-forget).
        Errors are swallowed — callers must not await a meaningful result.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"session_id": session_id}
                await client.post(self._unenroll_url, data=data)
        except Exception:
            pass  # fire-and-forget — swallow all errors

    def _create_error_event(self, service: str, session_id: str, code: str, message: str) -> Dict[str, Any]:
        """Creates a normalized error detection event."""
        return {
            "service": service,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "confidence": 0.0,
            "sessionId": session_id,
            "payload": {
                "status": "error",
                "code": code,
                "message": message
            }
        }
