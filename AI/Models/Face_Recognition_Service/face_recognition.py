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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_RECOGNITION_MODULE_ID = 3          # matches schema ModuleResult.id
DEFAULT_SIMILARITY_THRESHOLD = 0.5      # strict — suitable for exams
DEFAULT_DETECTION_THRESHOLD = 0.5

ARCFACE_INPUT_SIZE = (112, 112)         # ArcFace expects 112×112 aligned faces

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
    ):
        """
        Args:
            similarity_threshold: Min cosine similarity to accept a match (0–1).
            detection_threshold:  Min confidence for the RetinaFace detector.
            arcface_onnx_path:    Path to the ArcFace ``.onnx`` weights file.
            det_size:             Detection input resolution ``(width, height)``.
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
            name="buffalo_l",
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

    # ==================================================================
    # Public API  (matches the response schema used by all modules)
    # ==================================================================
    def compare_faces(
        self,
        frame: np.ndarray,
        reference_image: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare the face in *frame* against *reference_image*.

        Args:
            frame:           BGR image (current exam webcam frame).
            reference_image: BGR image of the authorised student (ID photo).

        Returns:
            Standardised result dict::

                {
                    "id":          3,
                    "timestamp":   "2026-02-09T…",
                    "flag":        True / False,
                    "probability": 0.0–1.0,
                    "evidence":    "human-readable description"
                }
        """
        timestamp = datetime.now().isoformat()

        # ---- Reference validation ----
        ref_faces = self._detect_faces(reference_image)

        if len(ref_faces) == 0:
            return self._result(
                timestamp, flag=True, probability=0.0,
                evidence="No face detected in reference image",
            )
        if len(ref_faces) > 1:
            return self._result(
                timestamp, flag=True, probability=0.0,
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
            return self._result(
                timestamp, flag=True, probability=0.95,
                evidence=f"Multiple faces detected: {num_faces} faces in frame",
            )

        # Case 2 — No face → CHEATING
        if num_faces == 0:
            return self._result(
                timestamp, flag=True, probability=0.85,
                evidence="No face detected in frame",
            )

        # Case 3 — Single face → verify identity
        frame_embedding = self._extract_embedding(frame, frame_faces[0])
        similarity = self._cosine_similarity(frame_embedding, ref_embedding)

        if similarity < self.similarity_threshold:
            return self._result(
                timestamp, flag=True,
                probability=round(1.0 - similarity, 4),
                evidence=f"Face does not match reference identity (similarity: {similarity:.4f})",
            )

        # All checks passed — NO CHEATING
        return self._result(
            timestamp, flag=False,
            probability=round(similarity, 4),
            evidence=f"Authorised person verified (similarity: {similarity:.4f})",
        )

    def compare_faces_base64(
        self,
        frame_base64: str,
        reference_base64: str,
    ) -> Dict[str, Any]:
        """Same as :meth:`compare_faces` but accepts base64-encoded images."""
        frame = self._decode_base64(frame_base64)
        reference = self._decode_base64(reference_base64)

        if frame is None:
            return self._result(
                datetime.now().isoformat(), flag=True, probability=0.0,
                evidence="Failed to decode frame image from base64",
            )
        if reference is None:
            return self._result(
                datetime.now().isoformat(), flag=True, probability=0.0,
                evidence="Failed to decode reference image from base64",
            )

        return self.compare_faces(frame, reference)

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
    def _result(
        timestamp: str,
        *,
        flag: bool,
        probability: float,
        evidence: str,
    ) -> Dict[str, Any]:
        return {
            "id": FACE_RECOGNITION_MODULE_ID,
            "timestamp": timestamp,
            "flag": flag,
            "probability": probability,
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

