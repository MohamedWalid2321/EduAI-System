"""
Face Recognition Module — Hybrid Approach
==========================================
Detection:   InsightFace RetinaFace (via FaceAnalysis, detection-only mode)
Recognition: ArcFace ONNX (w600k_r50.onnx) loaded directly with onnxruntime

Why hybrid?
  - InsightFace handles the complex RetinaFace box-decoding + landmark extraction.
  - Raw ONNX gives us full control over ArcFace inference, enables batching,
    and avoids pulling the full buffalo_l recognition weights.

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
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "InsightFace is required for face detection. "
        "Install with:  pip install insightface"
    )

from Models.FaceAntiSpoofing import FASModel

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

# Edit 3 — configurable interval between full FR embedding passes
DEFAULT_RECOGNITION_INTERVAL = 5.0      # seconds

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


class FaceRecognition:
    """
    Hybrid Face Recognition for AI Proctoring.

    Pipeline
    --------
    1. **Detection** — InsightFace ``FaceAnalysis`` (detection-only mode)
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
        Run RetinaFace to detect faces and return bounding boxes + landmarks.
    
    extract_embedding(image: np.ndarray, face) -> np.ndarray:
        Align and extract a 512-D ArcFace embedding from a detected face.
    
    align_face(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        Affine-warp a face to 112×112 using 5-point landmarks.
    
    cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        Compute cosine similarity between two L2-normalised vectors.
    
    result(timestamp: str, flag: bool, probability: float, evidence: str) -> dict:
        Build a standardised result dict matching the proctoring schema.
    
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
        fas_device: str = "cpu",
        recognition_interval: float = DEFAULT_RECOGNITION_INTERVAL,
    ):
        """
        Args:
            similarity_threshold: Min cosine similarity to accept a match (0–1).
            detection_threshold:  Min confidence for the RetinaFace detector.
            arcface_onnx_path:    Path to the ArcFace ``.onnx`` weights file.
            det_size:             Detection input resolution ``(width, height)``.
            fas_weights_path:     Path to MiniFASNetV2 weights. ``None`` = auto-resolve.
            fas_device:           Device for FAS inference (``'cpu'`` or ``'cuda'``).
            recognition_interval: Seconds between full FR embedding passes (Edit 3).
        """
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold

        providers = _ort_providers()

        # --- 1. DETECTION (InsightFace library, detection-only) ----------
        #  We let the library handle RetinaFace anchor decoding + NMS.
        #  'buffalo_l' ships with the detection model; allowed_modules
        #  restricts it so the heavy recognition model is NOT loaded.
        logger.info("Loading RetinaFace detector …")
        self._detector = FaceAnalysis(
            name="buffalo_l", #change it to 'buffalo_s' to get a smaller model 10GFP->2.5GFP (less accurate but faster)
            allowed_modules=["detection"],
            providers=providers,
        )
        self._detector.prepare(ctx_id=0, det_size=det_size) #ctx_id=0 forces GPU if available
        logger.info("RetinaFace detector ready  (det_size=%s)", det_size)

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
        self._fas = FASModel(weights_path=fas_weights_path, device=fas_device)
        logger.info("FAS model ready  (%s)", fas_weights_path)

        # Rolling window of recent liveness scores for active-challenge logic
        self._liveness_history: deque[float] = deque(maxlen=LIVENESS_RISK_WINDOW)

        # --- Edit 1 & 2: Enrollment cache (session_id → L2-normalised embedding) ---
        self._enrollment_store: Dict[str, np.ndarray] = {}

        # --- Edit 3: Recognition frequency gate (per-session) ---
        self.recognition_interval = recognition_interval
        self._last_fr_time: Dict[str, float] = {}          # session_id → monotonic timestamp
        self._last_fr_similarity: Dict[str, float] = {}    # session_id → last cosine sim

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
        """Compute and cache averaged reference embedding for a session.

        Call once at exam start with one or more reference photos.
        The averaged L2-normalised embedding is stored in-memory keyed by
        *session_id* and reused by :meth:`verify` / :meth:`compare_faces`.
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

        self._enrollment_store[session_id] = avg
        # Reset FR gate so first verify() triggers immediately
        self._last_fr_time.pop(session_id, None)
        self._last_fr_similarity.pop(session_id, None)

        logger.info("Enrolled session '%s' with %d image(s)", session_id, len(embeddings))
        return {"success": True, "session_id": session_id, "num_images": len(embeddings)}

    def unenroll(self, session_id: str) -> None:
        """Remove enrollment data for a session."""
        self._enrollment_store.pop(session_id, None)
        self._last_fr_time.pop(session_id, None)
        self._last_fr_similarity.pop(session_id, None)

    # ------------------------------------------------------------------
    # Edit 2 + 3 — Verify (detection + FAS every frame, FR gated)
    # ------------------------------------------------------------------
    def verify(self, session_id: str, frame: np.ndarray) -> Dict[str, Any]:
        """Per-frame verification against an enrolled template.

        * Detection + FAS run on **every** call.
        * FR embedding extraction + cosine comparison run only when
          ``recognition_interval`` seconds have elapsed since the last
          FR pass.  Between passes the last similarity is reused.
        """
        timestamp = datetime.now().isoformat()

        ref_embedding = self._enrollment_store.get(session_id)
        if ref_embedding is None:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.0,
                evidence=f"No enrollment found for session '{session_id}'",
            )

        # --- Detection (every frame) ---
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)

        if num_faces > 1:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=num_faces, quality=0.0,
                flag=True, probability=0.95,
                evidence=f"Multiple faces detected: {num_faces} faces in frame",
            )
        if num_faces == 0:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.85,
                evidence="No face detected in frame",
            )

        face = frame_faces[0]
        face_crop = self._crop_face(frame, face)
        quality = _face_quality(face_crop)

        # --- FAS (every frame) ---
        label, liveness_score = self._fas.predict(frame, face.bbox)
        self._liveness_history.append(liveness_score)
        challenge = self.active_challenge_required

        if label == "Spoof":
            return self._build_response(
                timestamp, match_similarity=0.0,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces, quality=round(quality, 4),
                flag=True, probability=round(1.0 - liveness_score, 4),
                evidence=(
                    f"Spoof detected \u2014 liveness score {liveness_score:.4f} "
                    f"(threshold {LIVENESS_HEALTHY_THRESHOLD})"
                ),
                active_challenge=challenge,
            )

        # --- FR gated by recognition_interval (Edit 3) ---
        now = time.monotonic() # use monotonic time for intervals to avoid issues with system clock changes
        last_time = self._last_fr_time.get(session_id, 0.0)

        if (now - last_time) >= self.recognition_interval:
            # Full FR pass — extract embedding and compare
            frame_embedding = self._extract_embedding(frame, face)
            similarity = self._cosine_similarity(frame_embedding, ref_embedding)
            self._last_fr_time[session_id] = now
            self._last_fr_similarity[session_id] = similarity
        else:
            # Between FR passes — reuse last known similarity
            similarity = self._last_fr_similarity.get(session_id, 0.0)

        if similarity < self.similarity_threshold:
            return self._build_response(
                timestamp, match_similarity=round(similarity, 4),
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces, quality=round(quality, 4),
                flag=True, probability=round(1.0 - similarity, 4),
                evidence=f"Face does not match reference identity (similarity: {similarity:.4f})",
                active_challenge=challenge,
            )

        return self._build_response(
            timestamp, match_similarity=round(similarity, 4),
            liveness_score=round(liveness_score, 4),
            num_faces=num_faces, quality=round(quality, 4),
            flag=False, probability=round(similarity, 4),
            evidence=f"Authorised person verified (similarity: {similarity:.4f})",
            active_challenge=challenge,
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
            if session_id not in self._enrollment_store:
                # Auto-enroll on first call with this session_id
                enroll_result = self.enroll(session_id, [reference_image])
                if not enroll_result["success"]:
                    return self._build_response(
                        timestamp, match_similarity=0.0, liveness_score=0.0,
                        num_faces=0, quality=0.0,
                        flag=True, probability=0.0,
                        evidence=enroll_result["error"],
                    )
            return self.verify(session_id, frame)

        # ---- Reference validation (legacy path — no session) ----
        ref_faces = self._detect_faces(reference_image)

        if len(ref_faces) == 0:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.0,
                evidence="No face detected in reference image",
            )
        if len(ref_faces) > 1:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=len(ref_faces), quality=0.0,
                flag=True, probability=0.0,
                evidence=(
                    f"Multiple faces ({len(ref_faces)}) in reference image. "
                    "Please provide a single-face photo."
                ),
            )

        ref_embedding = self._extract_embedding(reference_image, ref_faces[0])

        # ---- Current frame analysis ----
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)

        # Case 1 — Multiple faces → CHEATING
        if num_faces > 1:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=num_faces, quality=0.0,
                flag=True, probability=0.95,
                evidence=f"Multiple faces detected: {num_faces} faces in frame",
            )

        # Case 2 — No face → CHEATING
        if num_faces == 0:
            return self._build_response(
                timestamp, match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.85,
                evidence="No face detected in frame",
            )

        # ---- Single face: FAS → FR ----
        face = frame_faces[0]
        face_crop = self._crop_face(frame, face)
        quality = _face_quality(face_crop)

        # -- Face Anti-Spoofing (passive, every frame) --
        label, liveness_score = self._fas.predict(frame, face.bbox)
        self._liveness_history.append(liveness_score)
        challenge = self.active_challenge_required

        if label == "Spoof":
            return self._build_response(
                timestamp, match_similarity=0.0,
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces, quality=round(quality, 4),
                flag=True, probability=round(1.0 - liveness_score, 4),
                evidence=(
                    f"Spoof detected — liveness score {liveness_score:.4f} "
                    f"(threshold {LIVENESS_HEALTHY_THRESHOLD})"
                ),
                active_challenge=challenge,
            )

        # -- Face Recognition (embedding + match) --
        frame_embedding = self._extract_embedding(frame, face)
        similarity = self._cosine_similarity(frame_embedding, ref_embedding)

        if similarity < self.similarity_threshold:
            return self._build_response(
                timestamp, match_similarity=round(similarity, 4),
                liveness_score=round(liveness_score, 4),
                num_faces=num_faces, quality=round(quality, 4),
                flag=True, probability=round(1.0 - similarity, 4),
                evidence=f"Face does not match reference identity (similarity: {similarity:.4f})",
                active_challenge=challenge,
            )

        # All checks passed — NO CHEATING
        return self._build_response(
            timestamp, match_similarity=round(similarity, 4),
            liveness_score=round(liveness_score, 4),
            num_faces=num_faces, quality=round(quality, 4),
            flag=False, probability=round(similarity, 4), #probability: confidence of being the authorised student based on similarity
            evidence=f"Authorised person verified (similarity: {similarity:.4f})",
            active_challenge=challenge,
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
                match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.0,
                evidence="Failed to decode frame image from base64",
            )
        if reference is None:
            return self._build_response(
                datetime.now().isoformat(),
                match_similarity=0.0, liveness_score=0.0,
                num_faces=0, quality=0.0,
                flag=True, probability=0.0,
                evidence="Failed to decode reference image from base64",
            )

        return self.compare_faces(frame, reference, session_id=session_id)

    # ==================================================================
    # Internal — Detection
    # ==================================================================
    def _detect_faces(self, image: np.ndarray) -> List:
        """
        Run RetinaFace on *image*.

        Returns a list of insightface ``Face`` objects that carry
        ``.bbox``, ``.kps`` (5-point landmarks), and ``.det_score``.
        """
        faces = self._detector.get(image)
        # Filter by confidence
        return [f for f in faces if f.det_score >= self.detection_threshold]

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
        match_similarity: float,
        liveness_score: float,
        num_faces: int,
        quality: float,
        flag: bool,
        probability: float,
        evidence: str,
        active_challenge: bool = False,
    ) -> Dict[str, Any]:
        # Convert probability-like values from [0, 1] to percentage [0, 100]
        match_similarity_pct = round(match_similarity * 100.0, 2)
        liveness_score_pct = round(liveness_score * 100.0, 2)
        quality_pct = round(quality * 100.0, 2)
        probability_pct = round(probability * 100.0, 2)
        return {
            # --- new FAS contract ---
            "match_similarity": f"{match_similarity_pct}%",
            "liveness_score": f"{liveness_score_pct}%",
            "num_faces": num_faces,
            "quality": f"{quality_pct}%",
            "active_challenge_required": active_challenge,
            # --- legacy proctoring fields ---
            "id": FACE_RECOGNITION_MODULE_ID,
            "timestamp": timestamp,
            "flag": flag,
            "result": "Cheating Detected" if flag else "No Cheating",
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

