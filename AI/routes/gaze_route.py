from __future__ import annotations



"""
gaze_route.py
-------------
FastAPI route running on the desktop app's local server.

The backend sends a batch of base64-encoded frames to this route.
The route passes them to localMain.process_frames_batch(), which:
  - acquires a per-user threading.Lock before touching the session
  - loops through every frame
  - calls Modal per frame to get (h_ratio, v_ratio, face_present)
  - feeds results into the stateful GazeSession (calibration + detection)
  - returns a verdict per frame

Race condition protection lives in localMain — if the same user sends
two batches simultaneously, they are queued and processed one at a time.
The GazeSession is kept alive in localMain._sessions{} across
multiple batch calls so calibration is never lost.
"""
"""
the request body sent to the gaze_route
{
    "session_id": "test_user_001",
    "frames": [
        "/9j/4AAQSkZJRgAB...",
        "/9j/4AAQSkZJRgAB...",
        "/9j/4AAQSkZJRgAB..."
    ]
}
"""

"""
the response of the gaze_route
{
    "session_id":       "test_user_001",
    "frames_processed": 30,
    "summary_flag":     "AWAY_SHORT",
    "verdicts": [
        {
            "id":          1,
            "timestamp":   "2026-03-04T10:00:01",
            "flag":        "ON_SCREEN",
            "probability": 0.0,
            "evidence":    "ON_SCREEN"
        },
        {
            "id":          2,
            "timestamp":   "2026-03-04T10:00:01",
            "flag":        "AWAY_SHORT",
            "probability": 0.5,
            "evidence":    "AWAY_SHORT"
        }
    ]
}
"""


import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

import Models.EyeGazeDetection.src.Server.localMain as localMain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Gaze Detection"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GazeFramesBatchRequest(BaseModel):
    """
    Payload the backend sends to the desktop.

    session_id : stable identifier for the user/exam session.
                 The desktop uses this to look up (or create) the
                 correct GazeSession so calibration persists across
                 multiple batch calls.

    frames     : list of base64-encoded JPEG frames, in chronological order.
                 The backend captures frames and buffers them, then sends
                 the buffer here every few seconds.
    """
    """
    FastAPI uses this to validate the incoming request. If session_id is missing or frames is not a list, 
    FastAPI automatically returns a 422 error before your code even runs.
    """
    session_id: str               = Field(..., description="Unique user/exam session ID")
    frames:     list[str]         = Field(..., description="Base64-encoded JPEG frames array")


#Shape of each verdict in the response. FastAPI uses this to validate the output of the route handler.
class GazeVerdict(BaseModel):
    id:          int
    timestamp:   str
    flag:        str    # INITIALIZING | NO_FACE | ON_SCREEN | AWAY_SHORT | AWAY_LONG
    probability: float
    evidence:    str


#Shape of the full response. FastAPI will use this to validate the output of the route handler.
class GazeFramesBatchResponse(BaseModel):
    session_id:       str
    frames_processed: int
    verdicts:         list[GazeVerdict]
    summary_flag:     str   # worst flag seen in this batch

# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/gaze-frames",
    response_model=GazeFramesBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Eye-gaze detection on a batch of frames",
    description=(
        "Receives an array of base64-encoded frames from the backend. "
        "Each frame is forwarded to Modal (Gaze.py) to get raw gaze ratios. "
        "The stateful GazeSession on the desktop handles calibration and "
        "attention detection, returning one verdict per frame."
    ),
)
async def gaze_detection_frames(body: GazeFramesBatchRequest):

    if not body.frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="frames array is empty.",
        )

    loop = asyncio.get_running_loop()

    try:
        # Run in executor so the blocking Modal HTTP calls
        # don't block the FastAPI event loop
        verdicts: list[dict] = await loop.run_in_executor(
            None,
            partial(
                localMain.process_frames_batch,
                body.session_id,
                body.frames,
            ),
        )
        """
        This is the most important part. `process_frames_batch` is a **blocking** function — 
        it loops through 30 frames, calls Gaze.py for each one, which takes several seconds. 
        If you called it directly it would **freeze FastAPI** and no other requests could be handled during that time.

        run_in_executor moves it to a background thread so FastAPI stays responsive:
        Without run_in_executor:          With run_in_executor:
        ────────────────────────          ────────────────────────
        User A request arrives            User A request arrives
        FastAPI frozen for 10s            → moved to background thread
        No other requests handled         FastAPI free immediately
        User B has to wait                User B request handled instantly
        User A done → User B starts       Both processed simultaneously
        """
    except Exception as exc:
        logger.exception("[gaze_detection_frames] Unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    if not verdicts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No frames could be processed.",
        )

    summary_flag = _summarise(verdicts)

    return GazeFramesBatchResponse(
        session_id       = body.session_id,
        frames_processed = len(verdicts),
        verdicts         = [GazeVerdict(**v) for v in verdicts],
        summary_flag     = summary_flag,
    )


@router.delete(
    "/gaze-frames/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear a gaze session",
    description="Call this when the exam ends to free the GazeSession from memory.",
)
async def clear_gaze_session(session_id: str):
    localMain.clear_session(session_id)
    return {"detail": f"Session '{session_id}' cleared."}


# ---------------------------------------------------------------------------
# Helper — pick the most severe flag seen across the batch
# ---------------------------------------------------------------------------

_FLAG_SEVERITY = {
    "INITIALIZING": 0,
    "ON_SCREEN":    1,
    "NO_FACE":      2,
    "AWAY_SHORT":   3,
    "AWAY_LONG":    4,
}


def _summarise(verdicts: list[dict]) -> str:
    """Return the most severe flag seen in this batch."""
    return max(
        (v["flag"] for v in verdicts),
        key=lambda f: _FLAG_SEVERITY.get(f, 0),
    )