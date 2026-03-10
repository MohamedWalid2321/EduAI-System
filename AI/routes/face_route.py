"""Face recognition routes — enrollment, verification & legacy comparison."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from routes.helpers import bytes_to_cv2, executor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Face Recognition"])

# ---------------------------------------------------------------------------
# Singleton FaceRecognition instance — shared across requests so that the
# enrollment cache and FR-gate state persist for the container's lifetime.
# ---------------------------------------------------------------------------
_fr_instance = None


def _get_fr():
    """Return (and lazily create) the shared FaceRecognition instance."""
    global _fr_instance
    if _fr_instance is None:
        from Models.Face_Recognition_Service import FaceRecognition
        _fr_instance = FaceRecognition()
    return _fr_instance


# ===========================================================================
# ENROLL — compute & cache reference embedding(s) for a session (Edit 2)
# ===========================================================================
@router.post(
    "/enroll",
    status_code=status.HTTP_200_OK,
    summary="Enroll reference photo(s) for a session",
)
async def enroll(
    session_id: str = Form(..., description="Unique session identifier"),
    references: List[UploadFile] = File(..., description="One or more reference photos"),
):
    """
    Accepts one or more reference images + a ``session_id``.
    Computes an averaged ArcFace embedding and caches it in-memory.
    Subsequent ``/verify`` calls for the same session skip re-computation.
    """
    fr = _get_fr()

    images = []
    for ref in references:
        raw = await ref.read()
        img = bytes_to_cv2(raw, label="reference")
        images.append(img)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.enroll, session_id, images)
    )

    if not result.get("success"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result["error"])

    return result


# ===========================================================================
# UNENROLL — remove enrollment data for a session
# ===========================================================================
@router.post(
    "/unenroll",
    status_code=status.HTTP_200_OK,
    summary="Remove enrollment for a session",
)
async def unenroll(
    session_id: str = Form(..., description="Session ID to unenroll"),
):
    """Remove the cached reference embedding for the given session."""
    fr = _get_fr()
    fr.unenroll(session_id)
    return {"success": True, "session_id": session_id}


# ===========================================================================
# VERIFY — per-frame check against enrolled template (Edit 2 + 3)
# ===========================================================================
@router.post(
    "/verify",
    status_code=status.HTTP_200_OK,
    summary="Verify a frame against the enrolled session template",
)
async def verify(
    session_id: str = Form(..., description="Session ID from /enroll"),
    frame: UploadFile = File(..., description="Current exam webcam frame"),
):
    """
    Detection + FAS run on every call.
    FR embedding comparison runs only when the recognition interval has
    elapsed; between passes the last similarity is reused.
    """
    fr = _get_fr()

    frame_bytes = await frame.read()
    frame_img = bytes_to_cv2(frame_bytes, label="frame")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.verify, session_id, frame_img)
    )

    return {"face_recognition": result}


# ===========================================================================
# SINGLE FRAME — legacy (two image files — Postman / multipart form-data)
# ===========================================================================
@router.post(
    "/face-frame",
    status_code=status.HTTP_200_OK,
    summary="Face verification: compare frame against reference (image files)",
)
async def face_recognition_frame(
    frame: UploadFile = File(..., description="Current exam webcam frame"),
    reference: UploadFile = File(..., description="Authorised student ID photo"),
    session_id: Optional[str] = Form(None, description="Optional session ID for caching"),
):
    """
    Accepts two image files (multipart form-data).
    If *session_id* is supplied, the reference embedding is cached on
    the first call and reused on subsequent calls.
    """
    fr = _get_fr()

    frame_bytes = await frame.read()
    ref_bytes = await reference.read()

    frame_img = bytes_to_cv2(frame_bytes, label="frame")
    ref_img = bytes_to_cv2(ref_bytes, label="reference")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.compare_faces, frame_img, ref_img, session_id)
    )

    return {"face_recognition": result}


# ===========================================================================
# BASE64 JSON (for .NET backend / programmatic callers)
# ===========================================================================
class _FaceBase64Request(BaseModel):
    """Request body for base64 face comparison."""
    frame: str
    reference: str
    session_id: Optional[str] = None


@router.post(
    "/face-base64",
    status_code=status.HTTP_200_OK,
    summary="Face verification: compare frame against reference (base64 JSON)",
)
async def face_recognition_base64(body: _FaceBase64Request):
    """
    Accepts a JSON body with two base64-encoded images.
    Optional ``session_id`` enables embedding caching.
    """
    fr = _get_fr()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        partial(fr.compare_faces_base64, body.frame, body.reference, body.session_id),
    )

    return {"face_recognition": result}
