"""
Standalone face detection service (RetinaFace via InsightFace).

This service is intentionally independent from face recognition so it can be
used directly by dedicated endpoints while keeping the same detection logic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    raise ImportError(
        "InsightFace is required for face detection. "
        "Install with: pip install insightface"
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_NO_FACE_PROBABILITY = 0.85
DEFAULT_MULTIPLE_FACES_PROBABILITY = 0.95


def _ort_providers() -> list[str]:
    """Return the best available ONNX-Runtime execution providers."""
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class FaceDetectionService:
    """RetinaFace detector wrapper used by FR and face-detection-only routes."""

    _shared_detector = None
    _shared_det_size: Optional[Tuple[int, int]] = None
    _shared_providers: Optional[List[str]] = None
    _shared_lock: Lock = Lock()

    def __init__(
        self,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[Sequence[str]] = None,
    ):
        self.detection_threshold = detection_threshold
        resolved_providers = list(providers) if providers is not None else _ort_providers()

        with self._shared_lock:
            if self.__class__._shared_detector is None:
                logger.info("Loading RetinaFace detector...")
                detector = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],
                    providers=resolved_providers,
                )
                detector.prepare(ctx_id=0, det_size=det_size)

                self.__class__._shared_detector = detector
                self.__class__._shared_det_size = det_size
                self.__class__._shared_providers = resolved_providers

                logger.info("RetinaFace detector ready (det_size=%s)", det_size)
            elif self.__class__._shared_det_size != det_size:
                logger.warning(
                    "FaceDetectionService already initialized with det_size=%s; "
                    "requested det_size=%s will be ignored.",
                    self.__class__._shared_det_size,
                    det_size,
                )

        self._detector = self.__class__._shared_detector

    def detect_faces(self, image: np.ndarray) -> List:
        """Run RetinaFace and return faces that pass detection threshold."""
        faces = self._detector.get(image)
        return [face for face in faces if face.det_score >= self.detection_threshold]

    def analyze_frame(
        self,
        image: np.ndarray,
        session_id: Optional[str],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return endpoint-ready face-detection evidence for a single frame."""
        ts = timestamp or datetime.now().isoformat()
        faces = self.detect_faces(image)
        num_faces = len(faces)

        response: Dict[str, Any] = {
            "session_id": session_id,
            "timestamp": ts,
            "num_faces": num_faces,
        }

        if num_faces == 0:
            response.update(
                {
                    "evidence": "no_face_detected",
                    "probability": DEFAULT_NO_FACE_PROBABILITY,
                }
            )
            return response

        if num_faces > 1:
            response.update(
                {
                    "evidence": "multiple_faces",
                    "probability": DEFAULT_MULTIPLE_FACES_PROBABILITY,
                }
            )
            return response

        response["evidence"] = "One face detected"
        return response
