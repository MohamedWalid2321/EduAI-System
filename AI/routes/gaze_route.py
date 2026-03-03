"""Eye-gaze detection routes — video analysis."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from functools import partial

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from routes.helpers import extract_frames, aggregate, executor, TARGET_SECONDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Gaze Detection"])


# ===========================================================================
# VIDEO
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
            executor,
            partial(GazeMain.process_gaze_frame, frame, calibrating),
        )
        results.append(res)

    return {
        "gaze_detection": aggregate(results, module_id=1),
        "frames_processed": len(frames),
        "duration_seconds": TARGET_SECONDS,
    }
