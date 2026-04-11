"""
Standalone face detection service (local SCRFD ONNX inference).

This service is intentionally independent from face recognition so it can be
used directly by dedicated endpoints while keeping the same detection logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from .scrfd import SCRFD

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_NO_FACE_PROBABILITY = 0.85
DEFAULT_MULTIPLE_FACES_PROBABILITY = 0.95
DEFAULT_NMS_THRESHOLD = 0.4

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRFD_ONNX_PATH = os.path.join(_SERVICE_DIR, "det_10g.onnx")


@dataclass
class DetectedFace:
    """Lightweight face object compatible with existing FR code."""

    bbox: np.ndarray
    det_score: float
    kps: Optional[np.ndarray] = None


def _ort_providers() -> list[str]:
    """Return the best available ONNX-Runtime execution providers."""
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _resolve_scrfd_model_path(model_path: Optional[str]) -> str:
    """Resolve SCRFD ONNX path from explicit path, env, or local default."""
    env_path = os.getenv("FACE_DETECTION_ONNX_PATH")
    candidates = [model_path, env_path, DEFAULT_SCRFD_ONNX_PATH]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    checked_paths = [path for path in candidates if path]
    raise FileNotFoundError(
        "SCRFD ONNX model not found. Checked: "
        f"{checked_paths}. "
        "Set FACE_DETECTION_ONNX_PATH or place det_10g.onnx in the FaceDetection folder."
    )


class FaceDetectionService:
    """SCRFD detector wrapper used by FR and face-detection-only routes."""

    _shared_detector = None
    _shared_det_size: Optional[Tuple[int, int]] = None
    _shared_providers: Optional[List[str]] = None
    _shared_model_path: Optional[str] = None
    _shared_nms_threshold: Optional[float] = None
    _shared_lock: Lock = Lock()

    def __init__(
        self,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[Sequence[str]] = None,
        model_path: Optional[str] = None,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    ):
        self.detection_threshold = detection_threshold
        resolved_providers = list(providers) if providers is not None else _ort_providers()
        resolved_model_path = _resolve_scrfd_model_path(model_path)

        with self._shared_lock:
            if self.__class__._shared_detector is None:
                logger.info("Loading SCRFD detector from %s ...", resolved_model_path)

                session = ort.InferenceSession(
                    resolved_model_path,
                    providers=resolved_providers,
                )
                detector = SCRFD(model_file=resolved_model_path, session=session)
                detector.prepare(
                    ctx_id=0,
                    input_size=det_size,
                    det_thresh=detection_threshold,
                    nms_thresh=nms_threshold,
                )

                self.__class__._shared_detector = detector
                self.__class__._shared_det_size = det_size
                self.__class__._shared_providers = resolved_providers
                self.__class__._shared_model_path = resolved_model_path
                self.__class__._shared_nms_threshold = nms_threshold

                logger.info(
                    "SCRFD detector ready (det_size=%s, providers=%s)",
                    det_size,
                    resolved_providers,
                )
            elif self.__class__._shared_det_size != det_size:
                logger.warning(
                    "FaceDetectionService already initialized with det_size=%s; "
                    "requested det_size=%s will be ignored.",
                    self.__class__._shared_det_size,
                    det_size,
                )
            elif self.__class__._shared_model_path != resolved_model_path:
                logger.warning(
                    "FaceDetectionService already initialized with model=%s; "
                    "requested model=%s will be ignored.",
                    self.__class__._shared_model_path,
                    resolved_model_path,
                )
            elif self.__class__._shared_nms_threshold != nms_threshold:
                logger.warning(
                    "FaceDetectionService already initialized with nms_threshold=%s; "
                    "requested nms_threshold=%s will be ignored.",
                    self.__class__._shared_nms_threshold,
                    nms_threshold,
                )

        self._detector = self.__class__._shared_detector

    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """Run SCRFD and return faces that pass detection threshold."""
        det, kpss = self._detector.detect(
            image,
            input_size=self.__class__._shared_det_size,
        )

        if det is None or len(det) == 0:
            return []

        faces: List[DetectedFace] = []
        for idx in range(det.shape[0]):
            score = float(det[idx, 4])
            if score < self.detection_threshold:
                continue

            face = DetectedFace(
                bbox=det[idx, :4].astype(np.float32, copy=False),
                det_score=score,
                kps=(
                    kpss[idx].astype(np.float32, copy=False)
                    if kpss is not None
                    else None
                ),
            )
            faces.append(face)

        return faces

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
