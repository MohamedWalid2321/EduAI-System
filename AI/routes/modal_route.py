"""
Proctoring analysis route — unified endpoint.

Accepts a video frame and a reference face image (both base64-encoded),
runs all three AI modules concurrently, and returns a single aggregated
response.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from schemas.analysis import AnalysisResponse, ModuleResult

from Models.objectDetectionYolo.objectDetection import yoloDetect
import Models.EyeGazeDetection.src.Server.localMain as GazeMain
from Models.Face_Recognition_Service import FaceRecognition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Proctoring Analysis"])

# ---------------------------------------------------------------------------
# Shared model instances (initialised once, reused across requests)
# ---------------------------------------------------------------------------
_face_recognition = FaceRecognition()

# Thread-pool for CPU-bound model inference so we don't block the event loop
_executor = ThreadPoolExecutor(max_workers=3)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class AnalyzeFrameRequest(BaseModel):
    """Payload for the /analyze endpoint."""

    frame_base64: str = Field(
        ...,
        description="Base64-encoded current video frame (JPEG / PNG)",
    )
    reference_base64: str = Field(
        ...,
        description="Base64-encoded reference face image for identity verification",
    )
    calibrating: bool = Field(
        default=False,
        description="Whether the eye-gaze module is in calibration mode",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_base64_image(b64: str) -> np.ndarray:
    """Decode a base64 string into an OpenCV BGR image."""
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode base64 image: {exc}",
        ) from exc


def _run_gaze(frame: np.ndarray, calibrating: bool) -> dict:
    """Run the eye-gaze detection model (CPU-bound)."""
    return GazeMain.process_gaze_frame(frame, calibrating)


def _run_yolo(frame: np.ndarray) -> dict:
    """Run the YOLO object-detection model (CPU-bound)."""
    return yoloDetect(frame)


def _run_face(frame: np.ndarray, reference: np.ndarray) -> dict:
    """Run the face-recognition model (CPU-bound)."""
    return _face_recognition.compare_faces(frame, reference)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Run all proctoring models on a single frame",
    status_code=status.HTTP_200_OK,
    )
async def analyze_frame(payload: AnalyzeFrameRequest) -> AnalysisResponse:
    """
    Accept a base64-encoded video frame (and reference image), run **gaze
    detection**, **object detection**, and **face recognition** concurrently,
    and return their results in one response.
    """
    frame = _decode_base64_image(payload.frame_base64)
    reference = _decode_base64_image(payload.reference_base64)

    loop = asyncio.get_running_loop()

    # Run all three models in parallel on the thread-pool
    gaze_future = loop.run_in_executor(
        _executor, partial(_run_gaze, frame, payload.calibrating)
    )
    yolo_future = loop.run_in_executor(_executor, partial(_run_yolo, frame))
    face_future = loop.run_in_executor(
        _executor, partial(_run_face, frame, reference)
    )

    gaze_result, yolo_result, face_result = await asyncio.gather(
        gaze_future, yolo_future, face_future
    )

    return AnalysisResponse(
        gaze_detection=ModuleResult(**gaze_result),
        object_detection=ModuleResult(**yolo_result),
        face_recognition=ModuleResult(**face_result),
    )
