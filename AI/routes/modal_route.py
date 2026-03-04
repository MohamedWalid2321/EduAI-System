"""
Proctoring analysis routes — MODEL-SEPARATED
7 seconds = 210 frames
"""
"""
FACE RECOGNITION Functions:
-Video processing:
extract_frames(video_path: str) -> List[np.ndarray]: Extracts frames from a video file, returning a list of OpenCV images (numpy arrays). It samples frames to ensure a maximum of 210 frames for a 7-second video at 30 FPS.
aggregate(results: list[dict], module_id: int) -> dict: Aggregates results from multiple frames into a single summary dictionary. It calculates the maximum probability, determines if any flags were raised, and compiles evidence into a readable format.

-Obj Detection:
object_detection_frame(image: UploadFile) -> dict: Endpoint for object detection on a single image frame. It reads the uploaded image, decodes it, and runs the YOLO object detection model, returning the results.
object_detection_video(video: UploadFile) -> dict: Endpoint for optimized object detection on a video. It extracts frames, runs YOLO every 5th frame, and aggregates results. It also implements early stopping if a certain evidence is detected frequently.

-Eye Gaze Detection:
gaze_detection_video(video: UploadFile, calibrating: bool) -> dict: Endpoint for eye-gaze detection on a video. It extracts frames and processes each frame through the gaze detection model, aggregating results.

-Face Recognition:

face_recognition_frame(frame: UploadFile, reference: UploadFile) -> dict: Endpoint for
face recognition by comparing a webcam frame against a reference image. It decodes both images and runs the face recognition model, returning the comparison result.
face_recognition_base64(body: _FaceBase64Request) -> dict: Endpoint for face
recognition using base64-encoded images in a JSON body. It decodes the base64 strings and runs the face recognition comparison, returning the result.
_bytes_to_cv2(raw: bytes, label: str) -> np.ndarray: Helper function to
decode raw bytes into an OpenCV image. It raises an HTTPException if decoding fails, ensuring that only valid images are processed.
"""
from __future__ import annotations

import asyncio
import logging
import cv2
import numpy as np
import tempfile
from typing import List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Proctoring Analysis"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SECONDS = 7
FPS = 30
TARGET_FRAMES = TARGET_SECONDS * FPS  # 210

_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        raise ValueError("Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // TARGET_FRAMES, 1)

    idx = 0
    while cap.isOpened() and len(frames) < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()

    if not frames:
        raise ValueError("No frames extracted")

    return frames


def aggregate(results: list[dict], module_id: int) -> dict:
    probabilities = []
    flags = []
    evidence_set = set()

    for r in results:
        prob = r.get("propability") or r.get("probability") or 0.0
        probabilities.append(prob)
        flags.append(bool(r.get("flag", False)))

        ev = r.get("evidence")
        if ev:
            for item in ev.split(";"):
                item = item.strip()
                if item:
                    evidence_set.add(item)

    return {
        "id": module_id,
        "timestamp": datetime.now().isoformat(),
        "flag": any(flags),
        "propability": round(max(probabilities), 4),
        "evidence": ", ".join(sorted(evidence_set))
        if evidence_set
        else "No suspicious activity detected",
    }

# ===========================================================================
# OBJECT DETECTION — SINGLE FRAME (TESTING)
# ===========================================================================

@router.post(
    "/object-frame",
    status_code=status.HTTP_200_OK,
    summary="Object detection on a single frame (testing)",
)
async def object_detection_frame(
    image: UploadFile = File(...)
):
    from Models.objectDetectionYolo.objectDetection import yoloDetect

    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(422, "Invalid image")

    return yoloDetect(frame)

# ===========================================================================
# OBJECT DETECTION — VIDEO
# ===========================================================================

@router.post(
    "/object-video",
    status_code=status.HTTP_200_OK,
    summary="Optimized object detection (real-time)",
)
async def object_detection_video(video: UploadFile = File(...)):
    from Models.objectDetectionYolo.objectDetection import yoloDetect

    FPS = 12
    TARGET_FRAMES = 7 * FPS
    YOLO_STRIDE = 5  # run YOLO every 5th frame
    EARLY_STOP_COUNT = 10

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            video_path = tmp.name

        frames = extract_frames(video_path)

    except Exception as exc:
        raise HTTPException(422, str(exc))

    loop = asyncio.get_running_loop()
    results = []
    evidence_counter = {}

    for idx, frame in enumerate(frames):
        if idx % YOLO_STRIDE != 0:
            continue

        res = await loop.run_in_executor(
            _executor, partial(yoloDetect, frame)
        )
        results.append(res)

        ev = res.get("evidence")
        if ev:
            evidence_counter[ev] = evidence_counter.get(ev, 0) + 1

            if evidence_counter[ev] >= EARLY_STOP_COUNT:
                break

    return {
        "object_detection": aggregate(results, module_id=2),
        "frames_processed": len(results),
        "duration_seconds": 7,
        "early_stopped": True if results else False,
    }

# ===========================================================================
# GAZE DETECTION — VIDEO
# ===========================================================================

@router.post(
    "/gaze-video",
    status_code=status.HTTP_200_OK,
    summary="Eye-gaze detection on 7-second video",
)
async def gaze_detection_video(
    video: UploadFile = File(...),
    calibrating: bool = Form(False),
):
    import Models.EyeGazeDetection.src.Server.localMain as GazeMain

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            video_path = tmp.name

        frames = extract_frames(video_path)

    except Exception as exc:
        raise HTTPException(422, str(exc))

    loop = asyncio.get_running_loop()
    results = []

    for frame in frames:
        res = await loop.run_in_executor(
            _executor,
            partial(GazeMain.process_gaze_frame, frame, calibrating),
        )
        results.append(res)

    return {
        "gaze_detection": aggregate(results, module_id=1),
        "frames_processed": len(frames),
        "duration_seconds": TARGET_SECONDS,
    }


# ===========================================================================
# FACE RECOGNITION — SINGLE FRAME (two image files)
# ===========================================================================

@router.post(
    "/face-frame",
    status_code=status.HTTP_200_OK,
    summary="Face verification: compare frame against reference (image files)",
)
async def face_recognition_frame(
    frame: UploadFile = File(..., description="Current exam webcam frame"),
    reference: UploadFile = File(..., description="Authorised student ID photo"),
):
    """
    Accepts two image files (multipart form-data):
      - **frame**: the current webcam capture
      - **reference**: the student's registered ID photo

    Returns the standard proctoring module result.
    """
    from Models.Face_Recognition_Service import FaceRecognition

    frame_bytes = await frame.read()
    ref_bytes = await reference.read()

    frame_img = _bytes_to_cv2(frame_bytes, label="frame")
    ref_img = _bytes_to_cv2(ref_bytes, label="reference")

    fr = FaceRecognition()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor, partial(fr.compare_faces, frame_img, ref_img)
    )

    return {"face_recognition": result}


# ===========================================================================
# FACE RECOGNITION — BASE64 JSON
# ===========================================================================

class _FaceBase64Request(BaseModel):
    """Request body for base64 face comparison."""
    frame: str
    reference: str


@router.post(
    "/face-base64",
    status_code=status.HTTP_200_OK,
    summary="Face verification: compare frame against reference (base64 JSON)",
)
async def face_recognition_base64(body: _FaceBase64Request):
    """
    Accepts a JSON body with two base64-encoded images::

        { "frame": "<base64>", "reference": "<base64>" }

    Returns the standard proctoring module result.
    """
    from Models.Face_Recognition_Service import FaceRecognition

    fr = FaceRecognition()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        partial(fr.compare_faces_base64, body.frame, body.reference),
    )

    return {"face_recognition": result}


# ===========================================================================
# Shared image-decoding helper
# ===========================================================================

def _bytes_to_cv2(raw: bytes, *, label: str = "image") -> np.ndarray:
    """Decode raw bytes to a BGR OpenCV image; raise 422 on failure."""
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode {label} — unsupported or corrupt file.",
        )
    return img
