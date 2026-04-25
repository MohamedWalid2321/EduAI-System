"""
Face Recognition Module — Hybrid Approach
==========================================
Detection:   Local SCRFD ONNX (shared FaceDetectionService)
Recognition: ArcFace ONNX (w600k_r50.onnx) loaded directly with onnxruntime

Why hybrid?
    - Local SCRFD handles box-decoding + 5-point landmark extraction.
  - Raw ONNX gives us full control over ArcFace inference, enables batching,
        and keeps recognition independent from detection internals.

Designed to run as a Modal-hosted FastAPI service (consumed via Postman / .NET backend).

API endpoints (defined in routes/modal_route.py):
    POST /analysis/face-frame       — two image files  (frame + reference)
    POST /analysis/face-base64      — JSON with two base64 strings

Programmatic usage:
    from Models.Face_Recognition_Service import FaceRecognition

    fr = FaceRecognition()
    result = fr.compare_faces(frame_bgr, reference_bgr)
    result = fr.compare_faces_base64(frame_b64, ref_b64)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort

try:
    from upstash_redis import Redis as UpstashRedis
except ImportError:
    UpstashRedis = None

from Models.FaceAntiSpoofing import FASModel
from Models.FaceDetection import FaceDetectionService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_RECOGNITION_MODULE_ID = 3          # matches schema ModuleResult.id
DEFAULT_SIMILARITY_THRESHOLD = 0.5      # strict — suitable for exams
DEFAULT_DETECTION_THRESHOLD = 0.5

ARCFACE_INPUT_SIZE = (112, 112)         # ArcFace expects 112×112 aligned faces

# Liveness thresholds
LIVENESS_HEALTHY_THRESHOLD = 0.7        # passive mode — score >= this is healthy
LIVENESS_RISK_THRESHOLD = 0.5           # active challenge trigger threshold
LIVENESS_RISK_WINDOW = 3                # consecutive low frames to trigger active

# Legacy compatibility: constructor still accepts recognition_interval,
# but FR now runs on every verify call (no interval gate).
DEFAULT_RECOGNITION_INTERVAL = 5.0      # seconds
DEFAULT_REDIS_KEY_PREFIX = "face:enrollment"
DEFAULT_REDIS_TTL_ENV = "FACE_ENROLLMENT_TTL_SECONDS"
DEFAULT_REDIS_TTL_SECONDS = 10_800      # 3 hours (recommended exam default)

# Quality normalisation — Laplacian variance is unbounded, so we cap & scale
_LAPLACIAN_CAP = 500.0                  # empirical cap for sharpness

# Standard ArcFace 5-point destination landmarks (for 112×112)
_ARCFACE_DST_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Resolve the ONNX weights bundled alongside this file
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCFACE_ONNX_PATH = os.path.join(_SERVICE_DIR, "w600k_r50.onnx")


def _face_quality(face_bgr: np.ndarray) -> float:
    """Return a normalised [0, 1] sharpness score via Laplacian variance."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(variance / _LAPLACIAN_CAP, 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ort_providers() -> list[str]:
    """Return the best available ONNX-Runtime execution providers."""
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _resolve_fas_device(requested_device: str) -> str:
    """Resolve requested FAS device, preferring CUDA when available."""
    device = (requested_device or "").strip().lower()
    if device in {"", "auto"}:
        device = "cuda"

    if device == "cuda":
        try:
            import torch
        except Exception:
            logger.warning("PyTorch import failed; falling back to CPU for FAS.")
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        logger.warning(
            "FAS requested CUDA but no CUDA device is available; falling back to CPU."
        )
        return "cpu"

    if device == "cpu":
        return "cpu"

    logger.warning("Unsupported fas_device '%s'; falling back to CPU.", requested_device)
    return "cpu"


class FaceRecognition:
    """
    Hybrid Face Recognition for AI Proctoring.

    Pipeline
    --------
     1. **Detection** — local SCRFD detector (shared service)
       returns bounding boxes + 5-point keypoints.
    2. **Alignment** — Affine warp to 112×112 using the 5-point landmarks.
    3. **Recognition** — ArcFace ONNX (``w600k_r50.onnx``) produces a
       512-D L2-normalised embedding via ``onnxruntime``.
    4. **Verification** — Cosine similarity against the reference embedding.

    Cheating flags
    ~~~~~~~~~~~~~~
    - Multiple faces detected  → unauthorised person in frame
    - No face detected         → student left the frame
    - Face doesn't match       → different person sitting the exam

    Functions
    ~~~~~~~~~~~~~~
    compare_faces(frame: np.ndarray, reference_image: np.ndarray) -> dict: 
        Compare a webcam frame against the reference ID photo.
    
    compare_faces_base64(frame_base64: str, reference_base64: str) -> dict:
        Same as compare_faces but accepts base64-encoded images (for .NET).
    
    detect_faces(image: np.ndarray) -> list:
        Run SCRFD to detect faces and return bounding boxes + landmarks.
    
    extract_embedding(image: np.ndarray, face) -> np.ndarray:
        Align and extract a 512-D ArcFace embedding from a detected face.
    
    align_face(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        Affine-warp a face to 112×112 using 5-point landmarks.
    
    cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        Compute cosine similarity between two L2-normalised vectors.
    
    _build_response(timestamp: str, liveness_score: float, num_faces: int,
                    quality: float, probability: float, evidence: str,
                    session_id: Optional[str]) -> dict:
        Build a standardised face-recognition response payload.
    
    decode_base64(b64_string: str) -> Optional[np.ndarray]:
        Decode a base64 (or data-URI) string to a BGR numpy array.
            
"""

    
    # ------------------------------------------------------------------
    # Initialisation — called once, models stay warm in Modal container
    # ------------------------------------------------------------------
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        arcface_onnx_path: str = DEFAULT_ARCFACE_ONNX_PATH,
        det_size: Tuple[int, int] = (640, 640),
        fas_weights_path: str | None = None,
        fas_device: str = "auto",
        recognition_interval: float = DEFAULT_RECOGNITION_INTERVAL,
        face_detector: FaceDetectionService | None = None,
    ):
        """
        Args:
            similarity_threshold: Min cosine similarity to accept a match (0–1).
            detection_threshold:  Min confidence for the SCRFD detector.
            arcface_onnx_path:    Path to the ArcFace ``.onnx`` weights file.
            det_size:             Detection input resolution ``(width, height)``.
            fas_weights_path:     Path to MiniFASNetV2 weights. ``None`` = auto-resolve.
            fas_device:           Device for FAS inference (``'auto'``, ``'cpu'``, ``'cuda'``).
            recognition_interval: Legacy arg kept for backward compatibility.
            face_detector:        Optional shared face-detection service instance.
        """
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold

        providers = _ort_providers()

        # --- 1. DETECTION (standalone service) ---------------------------
        logger.info("Initializing FaceDetectionService...")
        self._face_detector = face_detector or FaceDetectionService(
            detection_threshold=detection_threshold,
            det_size=det_size,
            providers=providers,
        )
        logger.info("FaceDetectionService ready")

        # --- 2. RECOGNITION (raw ONNX via onnxruntime) -------------------
        #  Full control: we align + preprocess ourselves, then call the
        #  ONNX model directly.  This also lets us batch if needed later.
        if not os.path.isfile(arcface_onnx_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at: {arcface_onnx_path}"
            )

        logger.info("Loading ArcFace ONNX recogniser …")
        self._rec_session = ort.InferenceSession(arcface_onnx_path, providers=providers)
        self._rec_input_name = self._rec_session.get_inputs()[0].name
        logger.info("ArcFace ONNX recogniser ready  (%s)", arcface_onnx_path)

        # --- 3. FACE ANTI-SPOOFING (MiniFASNetV2) -----------------------
        if fas_weights_path is None:
            fas_weights_path = os.path.join(
                os.path.dirname(_SERVICE_DIR), "FaceAntiSpoofing",
                "2.7_80x80_MiniFASNetV2.pth",
            )
        logger.info("Loading FAS model (MiniFASNetV2) …")
        resolved_fas_device = _resolve_fas_device(fas_device)
        self._fas = FASModel(weights_path=fas_weights_path, device=resolved_fas_device)
        logger.info("FAS model ready  (%s)", fas_weights_path)

        # Rolling window of recent liveness scores for active-challenge logic
        self._liveness_history: deque[float] = deque(maxlen=LIVENESS_RISK_WINDOW)

        # Session-local cache to avoid a Redis fetch on every verify call.
        self._enrollment_store: Dict[str, np.ndarray] = {}
        self._redis_key_prefix = os.getenv(
            "FACE_ENROLLMENT_REDIS_PREFIX", DEFAULT_REDIS_KEY_PREFIX
        )
        self._redis_ttl_seconds = self._load_redis_ttl_seconds()
        self._redis = self._init_redis_client()

        # Keep legacy argument accepted; FR is no longer interval-gated.
        _ = recognition_interval

    def _init_redis_client(self):
        """Create Upstash Redis client from env vars if configured."""
        if UpstashRedis is None:
            logger.warning(
                "upstash-redis is not installed; enrollment will use in-memory cache only."
            )
            return None

        redis_url = self._normalise_redis_url(os.getenv("UPSTASH_REDIS_REST_URL", ""))
        redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip().strip('"\'')

        if not redis_url or not redis_token:
            logger.warning(
                "UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN is missing; "
                "enrollment will use in-memory cache only."
            )
            return None

        try:
            client = UpstashRedis(url=redis_url, token=redis_token)
            logger.info(
                "Upstash Redis enrollment store enabled (prefix='%s', ttl=%s).",
                self._redis_key_prefix,
                (
                    f"{self._redis_ttl_seconds}s"
                    if self._redis_ttl_seconds is not None
                    else "disabled"
                ),
            )
            return client
        except Exception:
            logger.exception("Failed to initialize Upstash Redis client.")
            return None


    @staticmethod
    #no need for this function now since we are using the upstash-redis client which handles the URL parsing and validation internally. However, if you want to keep it for additional validation or logging, you can uncomment the code and adjust it as needed.
    def _normalise_redis_url(raw_url: str) -> str:
        """Return a clean Upstash URL; auto-prepend https:// for host-only input."""
        url=raw_url
        # url = (raw_url or "").strip().strip('"\'')
        # if not url:
        #     return ""

        # if "://" not in url:
        #     logger.warning(
        #         "UPSTASH_REDIS_REST_URL is missing protocol. Prepending 'https://'."
        #     )
        #     url = f"https://{url}"

        # parsed = urlparse(url)
        # # Basic validation: must have http/https scheme and a network location (host)
        # if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        #     logger.error(
        #         "UPSTASH_REDIS_REST_URL is invalid ('%s'). "
        #         "Expected a full URL like 'https://<host>'.",
        #         raw_url,
        #     )
        #     return ""

        return url

    @staticmethod
    def _load_redis_ttl_seconds() -> Optional[int]:
        """Read enrollment TTL (seconds) from env with 3-hour default fallback."""
        raw = os.getenv(DEFAULT_REDIS_TTL_ENV, "").strip()
        if not raw:
            return DEFAULT_REDIS_TTL_SECONDS

        try:
            ttl_seconds = int(raw)
        except ValueError:
            logger.warning(
                "Invalid %s value '%s'; expected integer seconds. Using default %s.",
                DEFAULT_REDIS_TTL_ENV,
                raw,
                DEFAULT_REDIS_TTL_SECONDS,
            )
            return DEFAULT_REDIS_TTL_SECONDS

        if ttl_seconds <= 0:
            logger.warning(
                "%s must be > 0. Got %s; TTL disabled.",
                DEFAULT_REDIS_TTL_ENV,
                ttl_seconds,
            )
            return None

        return ttl_seconds

    def _redis_enrollment_key(self, session_id: str) -> str:
        """Build Redis key for a session enrollment vector."""
        return f"{self._redis_key_prefix}:{session_id}"

    #serialize to base64 string to store in Redis, and deserialize back to numpy array on retrieval
    #why base64? Redis stores strings, and embeddings are binary data. Base64 encodes the binary embedding into ASCII string format for safe storage and retrieval from Redis.
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> str:
        """Encode embedding as JSON payload with base64 float32 bytes."""
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        payload = {
            "dtype": "float32",
            "dim": int(vector.shape[0]),
            "vector_b64": base64.b64encode(vector.tobytes()).decode("ascii"),
        }
        return json.dumps(payload)

    @staticmethod
    def _deserialize_embedding(payload: Any) -> Optional[np.ndarray]:
        """Decode embedding payload from Redis into a normalised float32 vector."""
        try:
            if payload is None:
                return None

            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")

            if isinstance(payload, str):
                payload = json.loads(payload)

            if not isinstance(payload, dict):
                return None

            vector_b64 = payload.get("vector_b64")
            dim = int(payload.get("dim", 0))
            if not vector_b64 or dim <= 0:
                return None

            vector_bytes = base64.b64decode(vector_b64)
            embedding = np.frombuffer(vector_bytes, dtype=np.float32)
            if embedding.size != dim:
                return None

            embedding = embedding.copy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception:
            return None

    def _load_enrollment_embedding(self, session_id: str) -> Optional[np.ndarray]:
        """Load enrollment embedding from local cache first, then Upstash Redis."""
        cached = self._enrollment_store.get(session_id)
        if cached is not None:
            return cached

        if self._redis is None:
            return None

        try:
            payload = self._redis.get(self._redis_enrollment_key(session_id))
        except Exception:
            logger.exception(
                "Failed to fetch enrollment vector from Redis for session '%s'.",
                session_id,
            )
            return None

        embedding = self._deserialize_embedding(payload)
        if embedding is None:
            if payload is not None:
                logger.warning(
                    "Invalid enrollment vector payload in Redis for session '%s'.",
                    session_id,
                )
            return None

        self._enrollment_store[session_id] = embedding
        return embedding

    def _persist_enrollment_embedding(self, session_id: str, embedding: np.ndarray) -> bool:
        """Persist enrollment embedding to Redis (if configured) and local cache."""
        self._enrollment_store[session_id] = embedding

        if self._redis is None:
            return True

        try:
            payload = self._serialize_embedding(embedding)
            key = self._redis_enrollment_key(session_id)
            if self._redis_ttl_seconds is not None:
                self._redis.set(key, payload, ex=self._redis_ttl_seconds)
            else:
                self._redis.set(key, payload)
            return True
        except Exception:
            logger.exception(
                "Failed to persist enrollment vector to Redis for session '%s'.",
                session_id,
            )
            return False

    def _delete_enrollment_embedding(self, session_id: str) -> None:
        """Delete enrollment embedding from local cache and Redis."""
        self._enrollment_store.pop(session_id, None)

        if self._redis is None:
            return

        try:
            self._redis.delete(self._redis_enrollment_key(session_id))
        except Exception:
            logger.exception(
                "Failed to delete enrollment vector from Redis for session '%s'.",
                session_id,
            )

    # ==================================================================
    # Public API  (matches the response schema used by all modules)
    # ==================================================================
    # ------------------------------------------------------------------
    # Liveness helpers
    # ------------------------------------------------------------------
    def _crop_face(self, image: np.ndarray, face, pad_ratio: float = 0.3) -> np.ndarray:
        """Extract a BGR face crop with padding around the bounding box."""
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = image.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return image[y1:y2, x1:x2]

    @property
    def active_challenge_required(self) -> bool:
        """True when the last N liveness scores all fell below the risk threshold."""
        if len(self._liveness_history) < LIVENESS_RISK_WINDOW:
            return False
        return all(s < LIVENESS_RISK_THRESHOLD for s in self._liveness_history)

    # ------------------------------------------------------------------
    # Edit 2 — Enrollment
    # ------------------------------------------------------------------
    def enroll(
        self,
        session_id: str,
        reference_images: list[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute and persist averaged reference embedding for a session.

        Call once at exam start with one or more reference photos.
        The averaged L2-normalised embedding is stored in Upstash Redis
        (if configured) and cached in-memory for hot-path verification.
        """
        embeddings: list[np.ndarray] = []
        for i, img in enumerate(reference_images):
            faces = self._detect_faces(img)
            if len(faces) == 0:
                return {"success": False, "error": f"No face detected in reference image {i}"}
            if len(faces) > 1:
                return {"success": False, "error": f"Multiple faces in reference image {i}"}
            embeddings.append(self._extract_embedding(img, faces[0]))

        # Average embeddings and L2-normalise
        avg = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm

        # Persist the enrollment embedding; if Redis persistence fails, we consider the enrollment a failure since it won't survive beyond the current process.
        if not self._persist_enrollment_embedding(session_id, avg):
            self._enrollment_store.pop(session_id, None)
            return {
                "success": False,
                "error": "Failed to persist enrollment embedding in Redis.",
            }

        logger.info(
            "Enrolled session '%s' with %d image(s) (redis=%s)",
            session_id,
            len(embeddings),
            self._redis is not None,
        )
        return {"success": True, "session_id": session_id, "num_images": len(embeddings)}

    def unenroll(self, session_id: str) -> None:
        """Remove enrollment data for a session."""
        self._delete_enrollment_embedding(session_id)

    # ------------------------------------------------------------------
    # Verify (detection + FAS + FR on every frame)
    # ------------------------------------------------------------------
    def verify(self, session_id: str, frame: np.ndarray) -> Dict[str, Any]:
        """Per-frame verification against an enrolled template.

        * Detection + FAS run on **every** call.
        * FR embedding extraction + cosine comparison also run on **every** call.
        """
        timestamp = datetime.now().isoformat()

        ref_embedding = self._load_enrollment_embedding(session_id)
        if ref_embedding is None:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.0,
                evidence=f"No enrollment found for session '{session_id}'",
                session_id=session_id,
            )

        # --- Detection (every frame) ---
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)

        if num_faces > 1:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=num_faces,
                quality=0.0,
                probability=0.95,
                evidence=f"Multiple faces detected: {num_faces} faces in frame",
                session_id=session_id,
            )
        if num_faces == 0:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.85,
                evidence="No face detected in frame",
                session_id=session_id,
            )

        face = frame_faces[0]
        face_crop = self._crop_face(frame, face)
        quality = _face_quality(face_crop)

        # --- FAS (every frame) ---
        label, liveness_score = self._fas.predict(frame, face.bbox)
        self._liveness_history.append(liveness_score)

        if label == "Spoof":
            return self._build_response(
                timestamp,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces,
                quality=round(quality, 4),
                probability=-1.0,
                evidence=(
                    f"Spoof detected \u2014 liveness score {liveness_score:.4f} "
                    f"(threshold {LIVENESS_HEALTHY_THRESHOLD})"
                ),
                session_id=session_id,
            )

        # Full FR pass on every verify call.
        frame_embedding = self._extract_embedding(frame, face)
        similarity = self._cosine_similarity(frame_embedding, ref_embedding)

        if similarity < self.similarity_threshold:
            return self._build_response(
                timestamp,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces,
                quality=round(quality, 4),
                probability=round(similarity, 4),
                evidence=f"Face does not match reference identity (similarity: {similarity:.4f})",
                session_id=session_id,
            )

        return self._build_response(
            timestamp,
            liveness_score=round(liveness_score, 4),
            num_faces=num_faces,
            quality=round(quality, 4),
            probability=round(similarity, 4),
            evidence=f"Authorised person verified (similarity: {similarity:.4f})",
            session_id=session_id,
        )

    # ==================================================================
    # Public API  (matches the response schema used by all modules)
    # ==================================================================
    def compare_faces(
        self,
        frame: np.ndarray,
        reference_image: np.ndarray,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Compare the face in *frame* against *reference_image*.

        If *session_id* is provided the reference embedding is computed once
        and cached (Edit 1).  Subsequent calls with the same *session_id*
        skip reference re-computation and delegate to :meth:`verify`.

        Pipeline per frame:
            Face Detect → FAS check → (if Real) FR Embedding → Match

        Args:
            frame:           BGR image (current exam webcam frame).
            reference_image: BGR image of the authorised student (ID photo).
            session_id:      Optional session key for embedding caching.
        """
        timestamp = datetime.now().isoformat()

        # --- Edit 1: if enrolled for this session, delegate to verify ---
        if session_id is not None:
            if self._load_enrollment_embedding(session_id) is None:
                # Auto-enroll on first call with this session_id
                enroll_result = self.enroll(session_id, [reference_image])
                if not enroll_result["success"]:
                    return self._build_response(
                        timestamp,
                        liveness_score=0.0,
                        num_faces=0,
                        quality=0.0,
                        probability=0.0,
                        evidence=enroll_result["error"],
                        session_id=session_id,
                    )
            return self.verify(session_id, frame)

        # ---- Reference validation (legacy path — no session) ----
        ref_faces = self._detect_faces(reference_image)

        if len(ref_faces) == 0:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.0,
                evidence="No face detected in reference image",
                session_id=session_id,
            )
        if len(ref_faces) > 1:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=len(ref_faces),
                quality=0.0,
                probability=0.0,
                evidence=(
                    f"Multiple faces ({len(ref_faces)}) in reference image. "
                    "Please provide a single-face photo."
                ),
                session_id=session_id,
            )

        ref_embedding = self._extract_embedding(reference_image, ref_faces[0])

        # ---- Current frame analysis ----
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)

        # Case 1 — Multiple faces → CHEATING
        if num_faces > 1:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=num_faces,
                quality=0.0,
                probability=0.95,
                evidence=f"Multiple faces detected: {num_faces} faces in frame",
                session_id=session_id,
            )

        # Case 2 — No face → CHEATING
        if num_faces == 0:
            return self._build_response(
                timestamp,
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.85,
                evidence="No face detected in frame",
                session_id=session_id,
            )

        # ---- Single face: FAS → FR ----
        face = frame_faces[0]
        face_crop = self._crop_face(frame, face)
        quality = _face_quality(face_crop)

        # -- Face Anti-Spoofing (passive, every frame) --
        label, liveness_score = self._fas.predict(frame, face.bbox)
        self._liveness_history.append(liveness_score)

        if label == "Spoof":
            return self._build_response(
                timestamp,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces,
                quality=round(quality, 4),
                probability=round(1.0 - liveness_score, 4),
                evidence=(
                    f"Spoof detected — liveness score {liveness_score:.4f} "
                    f"(threshold {LIVENESS_HEALTHY_THRESHOLD})"
                ),
                session_id=session_id,
            )

        # -- Face Recognition (embedding + match) --
        frame_embedding = self._extract_embedding(frame, face)
        similarity = self._cosine_similarity(frame_embedding, ref_embedding)

        if similarity < self.similarity_threshold:
            return self._build_response(
                timestamp,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces,
                quality=round(quality, 4),
                probability=round(1.0 - similarity, 4),
                evidence=f"Face does not match reference identity (similarity: {similarity:.4f})",
                session_id=session_id,
            )

        # All checks passed — NO CHEATING
        return self._build_response(
            timestamp,
            liveness_score=round(liveness_score, 4),
            num_faces=num_faces,
            quality=round(quality, 4),
            probability=round(similarity, 4), #probability: confidence of being the authorised student based on similarity
            evidence=f"Authorised person verified (similarity: {similarity:.4f})",
            session_id=session_id,
        )

    def compare_faces_base64(
        self,
        frame_base64: str,
        reference_base64: str,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Same as :meth:`compare_faces` but accepts base64-encoded images."""
        frame = self._decode_base64(frame_base64)
        reference = self._decode_base64(reference_base64)

        if frame is None:
            return self._build_response(
                datetime.now().isoformat(),
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.0,
                evidence="Failed to decode frame image from base64",
                session_id=session_id,
            )
        if reference is None:
            return self._build_response(
                datetime.now().isoformat(),
                liveness_score=0.0,
                num_faces=0,
                quality=0.0,
                probability=0.0,
                evidence="Failed to decode reference image from base64",
                session_id=session_id,
            )

        return self.compare_faces(frame, reference, session_id=session_id)

    # ==================================================================
    # Internal — Detection
    # ==================================================================
    def _detect_faces(self, image: np.ndarray) -> List:
        """
        Run SCRFD on *image*.

        Returns a list of face objects that carry ``.bbox``,
        ``.kps`` (5-point landmarks), and ``.det_score``.
        """
        return self._face_detector.detect_faces(image)

    # ==================================================================
    # Internal — Alignment + Embedding
    # ==================================================================
    def _extract_embedding(self, image: np.ndarray, face) -> np.ndarray:
        """
        Align a detected face and extract its 512-D ArcFace embedding.

        Steps:
            1. Affine-warp to 112×112 using the 5-point keypoints.
            2. Preprocess to NCHW float32 blob normalised to [-1, 1].
            3. Run ArcFace ONNX inference.
            4. L2-normalise the resulting vector.
        """
        # 1. Align
        aligned = self._align_face(image, face.kps)

        # 2. Preprocess → (1, 3, 112, 112) blob normalised to [-1, 1]
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / 127.5,
            size=ARCFACE_INPUT_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )

        # 3. ONNX inference
        embedding = self._rec_session.run(
            None, {self._rec_input_name: blob}
        )[0].flatten()

        # 4. L2-normalise so cosine similarity = dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _align_face(
        image: np.ndarray,
        keypoints: np.ndarray | None,
        output_size: Tuple[int, int] = ARCFACE_INPUT_SIZE,
    ) -> np.ndarray:
        """Affine-warp a face to 112×112 using 5-point landmarks."""
        if keypoints is not None and keypoints.shape[0] >= 5:
            src = keypoints[:5].astype(np.float32)
            M = cv2.estimateAffinePartial2D(src, _ARCFACE_DST_LANDMARKS, method=cv2.RANSAC)[0]
            return cv2.warpAffine(image, M, output_size, borderValue=0.0)

        # Fallback: centre-crop & resize (lower accuracy)
        return cv2.resize(image, output_size)

    # ==================================================================
    # Internal — Similarity
    # ==================================================================
    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised 512-D vectors."""
        sim = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, sim))        # clamp into [0, 1]

    # ==================================================================
    # Internal — Result builder (consistent across all proctoring modules)
    # ==================================================================
    @staticmethod
    def _build_response(
        timestamp: str,
        *,
        liveness_score: float,
        num_faces: int,
        quality: float,
        probability: float,
        evidence: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Convert probability-like values from [0, 1] to percentage [0, 100]
        liveness_score_pct = round(liveness_score * 100.0, 2)
        quality_pct = round(quality * 100.0, 2)
        probability_pct = round(probability * 100.0, 2)
        return {
            "session_id": session_id,
            "liveness_score": f"{liveness_score_pct}%",
            "num_faces": num_faces,
            "quality": f"{quality_pct}%",
            "id": FACE_RECOGNITION_MODULE_ID,
            "timestamp": timestamp,
            "probability": f"{probability_pct}%",
            "evidence": evidence,
        }

    # ==================================================================
    # Internal — Base64 decoder
    # ==================================================================
    @staticmethod
    def _decode_base64(b64_string: str) -> Optional[np.ndarray]:
        """Decode a base64 (or data-URI) string to a BGR numpy array."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            raw = base64.b64decode(b64_string)
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

