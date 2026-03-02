"""
<<<<<<< HEAD
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
=======
Face Recognition Module
=======================
Face recognition using InsightFace library:
- RetinaFace for face detection
- ArcFace for face recognition/embedding extraction

Usage:
    from face_recognition import FaceRecognition
    
    # Initialize
    fr = FaceRecognition()
    
    # Compare frame with reference image
    result = fr.compare_faces(frame, reference_image)
    # Returns: {
    #     'id': 3,
    #     'timestamp': '2026-02-03T23:38:20.483018',
    #     'flag': True/False,
    #     'probability': 0.8678,
    #     'evidence': '...'
    # }
    
    # Or with base64 encoded images
    result = fr.compare_faces_base64(frame_base64, reference_base64)
"""
"""
HOW TO USE:

from face_recognition import FaceRecognition
import cv2

# Initialize
fr = FaceRecognition()

# Load images
reference = cv2.imread("reference.jpg")  # Authorized person's photo
frame = cv2.imread("frame.jpg")          # Current frame to check

# Compare
result = fr.compare_faces(frame, reference)  # for file inputs
# or result = fr.compare_faces_base64(frame_base64, reference_base64)  # for base64 inputs (ex: from web)
print(result)


Output: 
{
    'id': 3,
    'timestamp': '2026-02-03T23:38:20.483018',
    'flag': False,
    'probability': 0.9123,
    'evidence': 'Authorized person verified (similarity: 0.9123)'
}
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import base64
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
<<<<<<< HEAD
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
=======
        "InsightFace is required. Install with: pip install insightface onnxruntime"
    )


# Constants
FACE_RECOGNITION_MODULE_ID = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_DETECTION_THRESHOLD = 0.5


class FaceRecognition:
    """
    Face Recognition for AI Proctoring.
    
    Uses RetinaFace for detection and ArcFace for recognition.
    
    Cheating is flagged when:
    - Multiple faces are detected (unauthorized person in frame)
    - No face is detected (person left the frame)
    - Face doesn't match registered identity (different person)
    """
    
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
<<<<<<< HEAD
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

=======
        model_name: str = "buffalo_l",
        ctx_id: int = 0,  # 0 for GPU, -1 for CPU
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize the Face Recognition Pipeline.
        
        Args:
            similarity_threshold: Minimum cosine similarity to consider a match (0.0 to 1.0)
            detection_threshold: Minimum confidence for face detection
            model_name: InsightFace model pack name ('buffalo_l', 'buffalo_s', 'buffalo_sc')
            ctx_id: Context ID (0 for GPU, -1 for CPU)
            det_size: Detection input size (width, height)
        """
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold
        
        # Initialize InsightFace FaceAnalysis
        # This automatically loads RetinaFace for detection and ArcFace for recognition
        self.face_analyzer = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def compare_faces(
        self, 
        frame: np.ndarray, 
        reference_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare face in current frame with reference image.
        
        This is the main method - stateless comparison of two images.
        
        Args:
            frame: BGR image from video (current frame to check)
            reference_image: BGR image of the authorized person (reference)
            
        Returns:
            dict: {
                'id': int,              # Module ID (3 for face recognition)
                'timestamp': str,       # ISO format timestamp
                'flag': bool,           # True = cheating detected
                'probability': float,   # Confidence score (0.0 to 1.0)
                'evidence': str         # Description of detection result
            }
        """
        timestamp = datetime.now().isoformat()
        
        # Detect face in reference image first
        ref_faces = self._detect_faces(reference_image)
        
        if len(ref_faces) == 0:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.0,
                'evidence': 'No face detected in reference image'
            }
        
        if len(ref_faces) > 1:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.0,
                'evidence': f'Multiple faces ({len(ref_faces)}) in reference image. Please provide single face.'
            }
        
        # Get reference embedding
        ref_embedding = ref_faces[0].embedding / np.linalg.norm(ref_faces[0].embedding)
        
        # Detect faces in current frame
        frame_faces = self._detect_faces(frame)
        num_faces = len(frame_faces)
        
        # Case 1: Multiple faces detected - CHEATING
        if num_faces > 1:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.95,
                'evidence': f'Multiple faces detected: {num_faces} faces found in frame'
            }
        
        # Case 2: No face detected - CHEATING
        if num_faces == 0:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': 0.85,
                'evidence': 'No face detected in frame'
            }
        
        # Case 3: Single face detected - verify identity
        frame_embedding = frame_faces[0].embedding / np.linalg.norm(frame_faces[0].embedding)
        similarity = self._compute_similarity(frame_embedding, ref_embedding)
        
        # Check if face matches reference
        if similarity < self.similarity_threshold:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': timestamp,
                'flag': True,
                'probability': round(1.0 - similarity, 4), # Higher means more likely cheating, 4: decimal places
                'evidence': f'Face does not match reference identity (similarity: {similarity:.4f})'
            }
        
        # All checks passed - NO CHEATING
        return {
            'id': FACE_RECOGNITION_MODULE_ID,
            'timestamp': timestamp,
            'flag': False,
            'probability': round(similarity, 4), # Higher means less likely cheating, 4: decimal places
            'evidence': f'Authorized person verified (similarity: {similarity:.4f})'
        }
    
    def compare_faces_base64(
        self, 
        frame_base64: str, 
        reference_base64: str
    ) -> Dict[str, Any]:
        """
        Compare faces using base64 encoded images.
        
        Args:
            frame_base64: Base64 encoded current frame
            reference_base64: Base64 encoded reference image
            
        Returns:
            Same as compare_faces()
        """
        frame = self._decode_base64_image(frame_base64)
        reference = self._decode_base64_image(reference_base64)
        
        if frame is None:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': datetime.now().isoformat(),
                'flag': True,
                'probability': 0.0,
                'evidence': 'Failed to decode frame image from base64'
            }
        
        if reference is None:
            return {
                'id': FACE_RECOGNITION_MODULE_ID,
                'timestamp': datetime.now().isoformat(),
                'flag': True,
                'probability': 0.0,
                'evidence': 'Failed to decode reference image from base64'
            }
        
        return self.compare_faces(frame, reference)
    
    def get_face_embeddings(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract face embeddings from a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of 512-D embedding vectors for each detected face
        """
        faces = self._detect_faces(frame)
        return [face.embedding for face in faces]
    
    def _detect_faces(self, frame: np.ndarray) -> List:
        """
        Detect faces in a frame using RetinaFace.
        
        Args:
            frame: BGR image
            
        Returns:
            List of face objects with bbox, landmarks, embedding, etc.
        """
        faces = self.face_analyzer.get(frame)
        # Filter by detection threshold
        faces = [f for f in faces if f.det_score >= self.detection_threshold]
        return faces
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Normalize embeddings
        e1 = emb1.flatten() / (np.linalg.norm(emb1) + 1e-10)
        e2 = emb2.flatten() / (np.linalg.norm(emb2) + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(e1, e2)
        
        # Clip to [0, 1] range (cosine similarity can be negative)
        return float(np.clip(similarity, 0.0, 1.0))
    
    @staticmethod
    def _decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
        """
        Decode a base64 string to an OpenCV image.
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            BGR image as numpy array, or None if decoding fails
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
        
        except Exception:
            return None
        
        except Exception:
            return None
>>>>>>> f283ebec1b7f11684dfeff6e9246326d74ada2d9
