"""
Proctoring analysis routes — MODEL-SEPARATED
7 seconds = 210 frames
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
