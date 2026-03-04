"""Face recognition routes — frame comparison & base64 JSON."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from routes.helpers import bytes_to_cv2, executor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Face Recognition"])


# ===========================================================================
# SINGLE FRAME (two image files — Postman / multipart form-data)
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

    frame_img = bytes_to_cv2(frame_bytes, label="frame")
    ref_img = bytes_to_cv2(ref_bytes, label="reference")

    fr = FaceRecognition()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.compare_faces, frame_img, ref_img)
    )

    return {"face_recognition": result}


# ===========================================================================
# BASE64 JSON (for .NET backend / programmatic callers)
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
        executor,
        partial(fr.compare_faces_base64, body.frame, body.reference),
    )

    return {"face_recognition": result}
