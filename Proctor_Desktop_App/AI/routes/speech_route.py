import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

import Models.SpeechDetection.speech_localMain as speech_localMain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Speech Detection"])


# ---------------------------------------------------------------------------
# Request / Response Pydantic models
# ---------------------------------------------------------------------------

class SpeechChunkRequest(BaseModel):
    """
    Payload the backend sends per audio chunk.

    session_id  : Unique identifier for the exam session (used to track
                  continuous speaking duration across requests).
    audio_chunk : A single Base64-encoded PCM audio chunk.
                  Expected: 16 kHz, mono, float32 raw bytes, 1–2 seconds long.
    sample_rate : Sample rate of the audio (must be 16000 for Silero VAD).
    """

    session_id:  str = Field(..., description="Unique user/exam session ID")
    audio_chunk: str = Field(..., description="Base64-encoded float32 PCM audio chunk")
    sample_rate: int = Field(default=16000, description="Audio sample rate (must be 16000)")


class SpeechVerdict(BaseModel):
    """Shape of the single verdict returned per audio chunk."""

    timestamp:          str    # ISO-8601 UTC timestamp of when this chunk was evaluated
    flag:               str    # SILENCE | SPEAKING | SPEECH_VIOLATION
    speech_probability: float  # Raw Silero VAD confidence score [0.0 – 1.0]
    speaking_duration:  float  # Continuous speaking seconds accumulated so far
    evidence:           str    # Human-readable explanation of the flag


class SpeechChunkResponse(BaseModel):
    """Full response returned to the backend for every POST."""

    session_id: str
    verdict:    SpeechVerdict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/speech-chunk",
    response_model=SpeechChunkResponse,
    status_code=status.HTTP_200_OK,
    summary="Speech detection on a single audio chunk",
    description=(
        "Receives one Base64-encoded PCM audio chunk from the backend. "
        "The chunk is decoded and passed to Silero VAD locally. "
        "A per-session timer tracks continuous speaking duration; "
        "if it reaches 5 seconds the verdict flag is SPEECH_VIOLATION. "
        "Strike counting (5 violations → cheater) is handled by the main backend."
    ),
)
async def speech_detection_chunk(body: SpeechChunkRequest):

    if not body.audio_chunk:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="audio_chunk is empty.",
        )

    if body.sample_rate != 16000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported sample_rate {body.sample_rate}. Silero VAD requires 16000 Hz.",
        )

    loop = asyncio.get_running_loop()

    try:
        verdict: dict = await loop.run_in_executor(
            None,
            partial(
                speech_localMain.process_audio_chunk,
                body.session_id,
                body.audio_chunk,
                body.sample_rate,
            ),
        )
        """
        `process_audio_chunk` is blocking — it runs PyTorch inference (Silero VAD)
        which can take 10–50 ms per chunk.  Moving it to a thread pool via
        run_in_executor keeps FastAPI's event loop free so other concurrent
        requests (gaze, other speech sessions) are not stalled.
        """
    except Exception as exc:
        logger.exception("[speech_detection_chunk] Unexpected error for session '%s'", body.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return SpeechChunkResponse(
        session_id=body.session_id,
        verdict=SpeechVerdict(**verdict),
    )


@router.delete(
    "/speech-chunk/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear a speech session",
    description="Call this when the exam ends to free the SpeechSession state from memory.",
)
async def clear_speech_session(session_id: str):
    speech_localMain.manager.clear(session_id)
    return {"detail": f"Session '{session_id}' cleared."}