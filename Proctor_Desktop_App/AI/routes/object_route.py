"""Object detection routes (YOLO) — single frame & video."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from functools import partial

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from routes.helpers import extract_frames, aggregate, executor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Object Detection"])


# ===========================================================================
# SINGLE FRAME (testing)
# ===========================================================================
@router.post(
    "/object-frame",
    status_code=status.HTTP_200_OK,
    summary="Object detection on a single frame (testing)",
)
async def object_detection_frame(image: UploadFile = File(...)):
    from Models.objectDetectionYolo.objectDetection import yoloDetect

    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(422, "Invalid image")

    return yoloDetect(frame)


# ===========================================================================
# VIDEO
# ===========================================================================
@router.post(
    "/object-video",
    status_code=status.HTTP_200_OK,
    summary="Optimised object detection on 7-second video",
)
async def object_detection_video(video: UploadFile = File(...)):
    from Models.objectDetectionYolo.objectDetection import yoloDetect

    YOLO_FPS = 12
    YOLO_TARGET = 7 * YOLO_FPS
    YOLO_STRIDE = 5            # run YOLO every 5th frame
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
    evidence_counter: dict[str, int] = {}

    for idx, frame in enumerate(frames):
        if idx % YOLO_STRIDE != 0:
            continue

        res = await loop.run_in_executor(executor, partial(yoloDetect, frame))
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
