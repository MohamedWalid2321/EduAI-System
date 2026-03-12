"""Face recognition routes — enrollment, verification & legacy comparison."""

from __future__ import annotations

import asyncio
import base64
import logging
from functools import partial
from typing import List, Optional

import cv2
import numpy as np
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


# ---------------------------------------------------------------------------
# Helper — decode a base64 (or data-URI) string to a BGR numpy array
# ---------------------------------------------------------------------------
def _b64_to_cv2(b64_string: str, label: str = "image") -> np.ndarray:
    """Decode a base64 string to an OpenCV BGR image; raise 422 on failure."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        raw = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 encoding for {label}.",
        )
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode {label} — unsupported or corrupt image.",
        )
    return img


# ===========================================================================
# ENROLL — base64 JSON  (primary — for .NET backend)
# ===========================================================================
class _EnrollBase64Request(BaseModel):
    session_id: str
    references: List[str]   # list of base64-encoded images


@router.post(
    "/enroll",
    status_code=status.HTTP_200_OK,
    summary="Enroll reference photo(s) for a session (base64 JSON)",
)
async def enroll(body: _EnrollBase64Request):
    """
    Accepts a JSON body with a ``session_id`` and a list of base64-encoded
    reference images.  Computes an averaged ArcFace embedding and caches it
    in-memory.  Subsequent ``/verify`` calls for the same session skip
    re-computation.
    """
    fr = _get_fr()

    if not body.references:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="references list is empty.")

    images = [_b64_to_cv2(ref, label=f"reference[{i}]") for i, ref in enumerate(body.references)]

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.enroll, body.session_id, images)
    )

    if not result.get("success"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result["error"])

    return result


# ===========================================================================
# ENROLL — file upload  (testing — Postman / multipart form-data)
# ===========================================================================
@router.post(
    "/enroll-file",
    status_code=status.HTTP_200_OK,
    summary="Enroll reference photo(s) for a session (file upload, testing)",
)
async def enroll_file(
    session_id: str = Form(..., description="Unique session identifier"),
    references: List[UploadFile] = File(..., description="One or more reference photos"),
):
    """File-upload variant of /enroll — kept for Postman testing."""
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
# VERIFY — base64 JSON  (primary — for .NET backend)
# ===========================================================================
class _VerifyBase64Request(BaseModel):
    session_id: str
    frame: str              # base64-encoded image


@router.post(
    "/verify",
    status_code=status.HTTP_200_OK,
    summary="Verify a frame against the enrolled session template (base64 JSON)",
)
async def verify(body: _VerifyBase64Request):
    """
    Accepts a JSON body with ``session_id`` and a base64-encoded ``frame``.
    Detection + FAS run on every call.
    FR embedding comparison runs only when the recognition interval has
    elapsed; between passes the last similarity is reused.
    """
    fr = _get_fr()

    frame_img = _b64_to_cv2(body.frame, label="frame")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, partial(fr.verify, body.session_id, frame_img)
    )

    return {"face_recognition": result}


# ===========================================================================
# VERIFY — file upload  (testing — Postman / multipart form-data)
# ===========================================================================
@router.post(
    "/verify-file",
    status_code=status.HTTP_200_OK,
    summary="Verify a frame against the enrolled session template (file upload, testing)",
)
async def verify_file(
    session_id: str = Form(..., description="Session ID from /enroll"),
    frame: UploadFile = File(..., description="Current exam webcam frame"),
):
    """File-upload variant of /verify — kept for Postman testing."""
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
