"""
Object detection routes (YOLO) — single frame & video (modified)

Note: /object-video endpoint has been changed to accept a single image (frame)
instead of a video. The original testing endpoint /object-frame is kept.
"""

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

    # run on executor to avoid blocking event loop if yoloDetect is CPU-bound
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(executor, partial(yoloDetect, frame))
    return res


# ===========================================================================
# VIDEO (modified) -> now SINGLE FRAME endpoint
# ===========================================================================
@router.post(
    "/object-detection",
    status_code=status.HTTP_200_OK,
    summary="Object detection on a single uploaded frame (was video)",
)
async def object_detection_(image: UploadFile = File(...)):
    from Models.objectDetectionYolo.objectDetection import yoloDetect

    # Read and decode the uploaded image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid image file")

    # Run YOLO off the executor to avoid blocking
    loop = asyncio.get_running_loop()
    try:
        yolo_res = await loop.run_in_executor(executor, partial(yoloDetect, frame))
    except Exception as exc:
        logger.exception("YOLO inference failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"YOLO failed: {exc}")

    # Wrap the single-frame result into the same aggregated format your frontend expects
    # aggregate expects a list of per-frame dicts, so pass [yolo_res]
    agg = aggregate([yolo_res], module_id=2)

    return {
        "object_detection": agg,
        "frames_processed": 1,
        "duration_seconds": 0,  # single frame, not a duration
    }
