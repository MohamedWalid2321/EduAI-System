"""
test_clip_upload_service.py — Unit tests for ClipUploadService.

Tests are fully isolated: all external I/O (ffmpeg, requests, filesystem) is
mocked. The tests verify the four scenarios required by the spec:

  1. Success path — encode + PUT succeed on first attempt.
  2. Retry-once path — first PUT returns 500, second returns 200.
  3. Exhausted-retry path — both PUTs return 500 → UPLOAD_EXHAUSTED.
  4. Cleanup verification — both temp files are deleted in every path.
"""

import os
import types
import pytest
from unittest.mock import MagicMock, patch, mock_open, call


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

def _make_config(crf=23, preset="fast", retry_delay_ms=100):
    cfg = types.SimpleNamespace()
    cfg.base_url = "https://test.example.com"
    cfg.clip_recording = {
        "ffmpeg_crf": crf,
        "ffmpeg_preset": preset,
        "upload_retry_delay_ms": retry_delay_ms,
    }
    return cfg


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

FAKE_ENV = {
    "BUNNY_STORAGE_URL": "https://storage.bunnycdn.com/test-zone",
    "BUNNY_API_KEY": "test-api-key-do-not-log",
    "BUNNY_CDN_BASE_URL": "https://test-pullzone.b-cdn.net",
}

FAKE_METADATA = {
    "studentId": "student-123",
    "examAttemptId": "42",
    "sessionId": "42",
    "captureWindowStart": "2026-05-05T14:32:01.000Z",
    "captureWindowEnd": "2026-05-05T14:32:21.000Z",
    "primaryViolationType": "UNAUTHORIZED_OBJECT",
    "primaryConfidence": 0.92,
    "description": "Phone detected in frame",
    "allViolations": [
        {
            "violationType": "UNAUTHORIZED_OBJECT",
            "confidence": 0.92,
            "timestamp": "2026-05-05T14:32:01.000Z",
            "description": "Phone detected in frame",
        }
    ],
    "token": "test-bearer-token",
}

FAKE_WEBM = "/tmp/clip-123.webm"
FAKE_MP4 = "/tmp/clip-456.mp4"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_200_response():
    resp = MagicMock()
    resp.status_code = 200
    return resp


def _make_500_response():
    resp = MagicMock()
    resp.status_code = 500
    return resp


# ---------------------------------------------------------------------------
# Test 1: Success path
# ---------------------------------------------------------------------------

@patch.dict(os.environ, FAKE_ENV)
@patch("services.clip_upload_service.requests.post")
@patch("services.clip_upload_service.requests.put")
@patch("services.clip_upload_service.ffmpeg")
@patch("services.clip_upload_service.tempfile.mkstemp")
@patch("services.clip_upload_service.os.path.exists")
@patch("services.clip_upload_service.os.remove")
@patch("services.clip_upload_service.os.close")
@patch("builtins.open", mock_open(read_data=b"fake-mp4-data"))
def test_success_path(
    mock_os_close,
    mock_os_remove,
    mock_exists,
    mock_mkstemp,
    mock_ffmpeg,
    mock_put,
    mock_post,
):
    """Encode + PUT succeed on first attempt → uploadStatus == 'success'."""
    from services.clip_upload_service import ClipUploadService

    mock_mkstemp.return_value = (5, FAKE_MP4)
    mock_put.return_value = _make_200_response()
    mock_exists.return_value = True

    # ffmpeg stubs
    mock_stream = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value = mock_stream
    mock_ffmpeg.run.return_value = None

    service = ClipUploadService(_make_config())
    result = service.upload_clip(FAKE_WEBM, FAKE_METADATA)

    assert result["uploadStatus"] == "success"
    cdn_base = FAKE_ENV["BUNNY_CDN_BASE_URL"]
    assert result["evidenceUrl"] is not None
    assert result["evidenceUrl"].startswith(cdn_base)
    assert result["reasonCode"] is None

    # Only one PUT was needed
    mock_put.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: Retry-once path (first PUT 500, second PUT 200)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, FAKE_ENV)
@patch("services.clip_upload_service.requests.post")
@patch("services.clip_upload_service.requests.put")
@patch("services.clip_upload_service.time.sleep")
@patch("services.clip_upload_service.ffmpeg")
@patch("services.clip_upload_service.tempfile.mkstemp")
@patch("services.clip_upload_service.os.path.exists")
@patch("services.clip_upload_service.os.remove")
@patch("services.clip_upload_service.os.close")
@patch("builtins.open", mock_open(read_data=b"fake-mp4-data"))
def test_retry_once_path(
    mock_os_close,
    mock_os_remove,
    mock_exists,
    mock_mkstemp,
    mock_ffmpeg,
    mock_sleep,
    mock_put,
    mock_post,
):
    """First PUT returns 500, retry returns 200 → uploadStatus == 'success', two PUT calls."""
    from services.clip_upload_service import ClipUploadService

    mock_mkstemp.return_value = (5, FAKE_MP4)
    mock_put.side_effect = [_make_500_response(), _make_200_response()]
    mock_exists.return_value = True

    mock_stream = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value = mock_stream
    mock_ffmpeg.run.return_value = None

    service = ClipUploadService(_make_config(retry_delay_ms=100))
    result = service.upload_clip(FAKE_WEBM, FAKE_METADATA)

    assert result["uploadStatus"] == "success"
    assert mock_put.call_count == 2
    mock_sleep.assert_called_once()  # delay between attempts


# ---------------------------------------------------------------------------
# Test 3: Exhausted-retry path (both PUTs return 500)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, FAKE_ENV)
@patch("services.clip_upload_service.requests.post")
@patch("services.clip_upload_service.requests.put")
@patch("services.clip_upload_service.time.sleep")
@patch("services.clip_upload_service.ffmpeg")
@patch("services.clip_upload_service.tempfile.mkstemp")
@patch("services.clip_upload_service.os.path.exists")
@patch("services.clip_upload_service.os.remove")
@patch("services.clip_upload_service.os.close")
@patch("builtins.open", mock_open(read_data=b"fake-mp4-data"))
def test_exhausted_retry_path(
    mock_os_close,
    mock_os_remove,
    mock_exists,
    mock_mkstemp,
    mock_ffmpeg,
    mock_sleep,
    mock_put,
    mock_post,
):
    """Both PUTs return 500 → uploadStatus == 'upload_failed', reasonCode == 'UPLOAD_EXHAUSTED'."""
    from services.clip_upload_service import ClipUploadService

    mock_mkstemp.return_value = (5, FAKE_MP4)
    mock_put.return_value = _make_500_response()
    mock_exists.return_value = True

    mock_stream = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value = mock_stream
    mock_ffmpeg.run.return_value = None

    service = ClipUploadService(_make_config(retry_delay_ms=100))
    result = service.upload_clip(FAKE_WEBM, FAKE_METADATA)

    assert result["uploadStatus"] == "upload_failed"
    assert result["reasonCode"] == "UPLOAD_EXHAUSTED"
    assert result["evidenceUrl"] is None
    assert mock_put.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: Cleanup verification (temp files removed in all paths)
# ---------------------------------------------------------------------------

@patch.dict(os.environ, FAKE_ENV)
@patch("services.clip_upload_service.requests.post")
@patch("services.clip_upload_service.requests.put")
@patch("services.clip_upload_service.ffmpeg")
@patch("services.clip_upload_service.tempfile.mkstemp")
@patch("services.clip_upload_service.os.path.exists")
@patch("services.clip_upload_service.os.remove")
@patch("services.clip_upload_service.os.close")
@patch("builtins.open", mock_open(read_data=b"fake-mp4-data"))
def test_cleanup_both_temp_files(
    mock_os_close,
    mock_os_remove,
    mock_exists,
    mock_mkstemp,
    mock_ffmpeg,
    mock_put,
    mock_post,
):
    """Both .webm and .mp4 temp files are removed after the call (success path)."""
    from services.clip_upload_service import ClipUploadService

    mock_mkstemp.return_value = (5, FAKE_MP4)
    mock_put.return_value = _make_200_response()
    mock_exists.return_value = True

    mock_stream = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value = mock_stream
    mock_ffmpeg.run.return_value = None

    service = ClipUploadService(_make_config())
    service.upload_clip(FAKE_WEBM, FAKE_METADATA)

    removed_paths = {c.args[0] for c in mock_os_remove.call_args_list}
    assert FAKE_WEBM in removed_paths, ".webm temp file was not removed"
    assert FAKE_MP4 in removed_paths, ".mp4 temp file was not removed"


@patch.dict(os.environ, FAKE_ENV)
@patch("services.clip_upload_service.requests.post")
@patch("services.clip_upload_service.requests.put")
@patch("services.clip_upload_service.time.sleep")
@patch("services.clip_upload_service.ffmpeg")
@patch("services.clip_upload_service.tempfile.mkstemp")
@patch("services.clip_upload_service.os.path.exists")
@patch("services.clip_upload_service.os.remove")
@patch("services.clip_upload_service.os.close")
@patch("builtins.open", mock_open(read_data=b"fake-mp4-data"))
def test_cleanup_on_failure(
    mock_os_close,
    mock_os_remove,
    mock_exists,
    mock_mkstemp,
    mock_ffmpeg,
    mock_sleep,
    mock_put,
    mock_post,
):
    """Temp files are cleaned up even when both retries fail."""
    from services.clip_upload_service import ClipUploadService

    mock_mkstemp.return_value = (5, FAKE_MP4)
    mock_put.return_value = _make_500_response()
    mock_exists.return_value = True

    mock_stream = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value = mock_stream
    mock_ffmpeg.run.return_value = None

    service = ClipUploadService(_make_config(retry_delay_ms=100))
    service.upload_clip(FAKE_WEBM, FAKE_METADATA)

    removed_paths = {c.args[0] for c in mock_os_remove.call_args_list}
    assert FAKE_WEBM in removed_paths, ".webm temp file was not removed on failure"
    assert FAKE_MP4 in removed_paths, ".mp4 temp file was not removed on failure"
