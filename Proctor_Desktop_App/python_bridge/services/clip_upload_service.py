"""
clip_upload_service.py — Violation clip upload service for Lumina AI.

Responsibilities:
  1. Re-encode the incoming .webm (VP9) clip to H.264 MP4 via ffmpeg-python.
  2. Upload the MP4 to Bunny CDN via HTTP PUT.
  3. Retry once after `upload_retry_delay_ms` on a non-2xx response.
  4. Emit an AiProctoringViolationEvent to the LMS backend API.
  5. Clean up both temp files in a try/finally block.

Security rules:
  - BUNNY_STORAGE_URL, BUNNY_API_KEY, BUNNY_CDN_BASE_URL are read from
    config.json (clip_recording block) with environment variable fallback.
    They are never logged, returned in results, or transmitted over IPC to
    the renderer.
  - The access token (for the LMS backend POST) is received as a parameter and
    used only as a Bearer header. It is never logged.
"""

import json
import os
import time
import tempfile
import datetime
import requests
import urllib3

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore — guarded at call site

try:
    import imageio_ffmpeg
    _FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_BIN = "ffmpeg"  # fall back to system PATH

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ClipUploadService:
    """
    Encode, upload, and notify the backend about a violation clip.

    Credentials are read from the `clip_recording` block in config.json
    (keys: bunny_storage_url, bunny_api_key, bunny_cdn_base_url). If a
    config value is absent or empty, the corresponding environment variable
    (BUNNY_STORAGE_URL, BUNNY_API_KEY, BUNNY_CDN_BASE_URL) is used as a
    fallback.
    """

    def __init__(self, config):
        """
        Initialise the service.

        Args:
            config: AppConfig instance (provides base_url and clip_recording block).

        Raises:
            RuntimeError: If any required environment variable is missing.
        """
        clip_cfg = getattr(config, "clip_recording", {}) or {}

        def _resolve(cfg_key: str, env_key: str) -> str:
            """Return config value if non-empty, else fall back to env var."""
            return str(clip_cfg.get(cfg_key, "") or os.environ.get(env_key, ""))

        self._storage_url = _resolve("bunny_storage_url", "BUNNY_STORAGE_URL").rstrip("/")
        self._api_key = _resolve("bunny_api_key", "BUNNY_API_KEY")
        self._cdn_base_url = _resolve("bunny_cdn_base_url", "BUNNY_CDN_BASE_URL").rstrip("/")

        missing = []
        if not self._storage_url:
            missing.append("bunny_storage_url / BUNNY_STORAGE_URL")
        if not self._api_key:
            missing.append("bunny_api_key / BUNNY_API_KEY")
        if not self._cdn_base_url:
            missing.append("bunny_cdn_base_url / BUNNY_CDN_BASE_URL")

        if missing:
            raise RuntimeError(
                f"ClipUploadService: missing required credential(s): "
                f"{', '.join(missing)}"
            )

        self._base_url = getattr(config, "base_url", "").rstrip("/")

        self._ffmpeg_crf = int(clip_cfg.get("ffmpeg_crf", 23))
        self._ffmpeg_preset = str(clip_cfg.get("ffmpeg_preset", "fast"))
        self._retry_delay_s = int(clip_cfg.get("upload_retry_delay_ms", 3000)) / 1000.0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def upload_clip(self, temp_webm_path: str, metadata: dict) -> dict:
        """
        Encode the .webm clip to H.264 MP4 (video) or upload the .webm audio
        directly (audio), upload to Bunny CDN, and notify the backend.

        For video clips (default): re-encodes with ffmpeg → .mp4, then uploads.
        For audio clips (mediaType == 'audio'): re-encodes the browser-recorded
        .webm blob to MP3 via ffmpeg, then uploads as .mp3.

        Args:
            temp_webm_path: Absolute path to the temporary .webm file written
                            by the Electron main process.
            metadata:       ClipMetadata dict (see ipc-save-and-upload-clip.json
                            contract). Must contain at minimum: studentId,
                            examAttemptId, sessionId, captureWindowStart,
                            captureWindowEnd, primaryViolationType,
                            primaryConfidence, description, allViolations, token.
                            Optional: mediaType ('audio' | 'video', default 'video').

        Returns:
            dict with keys: uploadStatus, evidenceUrl (or None), reasonCode (or None).
        """
        media_type = str(metadata.get("mediaType", "video") or "video").lower()
        is_audio = (media_type == "audio")

        temp_mp4_path: str | None = None
        temp_mp3_path: str | None = None
        upload_status = "upload_failed"
        evidence_url: str | None = None
        reason_code: str | None = None

        try:
            student_id = str(metadata.get("studentId", "unknown"))
            primary_type = str(metadata.get("primaryViolationType", "VIOLATION"))
            capture_start = str(metadata.get("captureWindowStart", ""))
            url_safe_ts = capture_start.replace(":", "-")

            if is_audio:
                # ------------------------------------------------------------------
                # Audio path: ffmpeg re-encode .webm → MP3, then upload
                # ------------------------------------------------------------------
                if ffmpeg is None:
                    raise RuntimeError("ffmpeg-python package is not installed.")

                _fd, temp_mp3_path = tempfile.mkstemp(suffix=".mp3")
                os.close(_fd)

                try:
                    _, ff_err_audio = ffmpeg.run(
                        ffmpeg.input(temp_webm_path).audio
                            .filter("asetpts", "PTS-STARTPTS")
                            .output(
                                temp_mp3_path,
                                acodec="libmp3lame",
                                **{"b:a": "192k"},  # correct ffmpeg-python kwarg for audio bitrate
                                vn=None,  # discard any video stream
                            ),
                        capture_stdout=True,
                        capture_stderr=True,
                        overwrite_output=True,
                        cmd=_FFMPEG_BIN,
                    )
                except ffmpeg.Error as ff_exc:
                    raise RuntimeError(
                        f"ffmpeg audio encode failed: "
                        f"{ff_exc.stderr.decode(errors='replace') if ff_exc.stderr else str(ff_exc)}"
                    ) from ff_exc

                if not os.path.exists(temp_mp3_path) or os.path.getsize(temp_mp3_path) == 0:
                    raise RuntimeError(
                        "ffmpeg produced an empty MP3 — input WebM may be unreadable. stderr: "
                        + (ff_err_audio.decode(errors='replace') if ff_err_audio else '')
                    )

                filename = f"{student_id}/{url_safe_ts}_{primary_type}.mp3"
                upload_url = f"{self._storage_url}/{filename}"

                def _do_put_audio():
                    with open(temp_mp3_path, "rb") as fh:
                        resp = requests.put(
                            upload_url,
                            data=fh,
                            headers={"AccessKey": self._api_key},
                            timeout=60,
                        )
                    return resp

                resp = _do_put_audio()
                if 200 <= resp.status_code < 300:
                    upload_status = "success"
                    evidence_url = f"{self._cdn_base_url}/{filename}"
                else:
                    time.sleep(self._retry_delay_s)
                    resp = _do_put_audio()
                    if 200 <= resp.status_code < 300:
                        upload_status = "success"
                        evidence_url = f"{self._cdn_base_url}/{filename}"
                    else:
                        upload_status = "upload_failed"
                        reason_code = "UPLOAD_EXHAUSTED"

            else:
                # ------------------------------------------------------------------
                # Video path: ffmpeg re-encode .webm → H.264 MP4, then upload
                # ------------------------------------------------------------------
                if ffmpeg is None:
                    raise RuntimeError("ffmpeg-python package is not installed.")

                _fd, temp_mp4_path = tempfile.mkstemp(suffix=".mp4")
                os.close(_fd)

                # MediaRecorder chunks carry timestamps relative to recording
                # start (not clip start), so the ring-buffer pre-snap begins at
                # e.g. T+55s rather than T+0s.  Without normalisation ffmpeg
                # preserves those offsets, producing an MP4 whose reported
                # duration equals the session elapsed time (~75s) even though
                # only 20s of frames exist.  setpts resets the video stream to
                # start at 0 so players show the correct ~20s duration.
                # NOTE: the video WebM is captured with audio:false (no audio
                # track).  acodec/audio filters must NOT be used here — they
                # would cause ffmpeg to fail with "no audio stream" on every
                # video clip.  an=None emits -an (suppress audio output).
                _inp = ffmpeg.input(temp_webm_path)
                stream = ffmpeg.output(
                    _inp.video.filter("setpts", "PTS-STARTPTS"),
                    temp_mp4_path,
                    vcodec="libx264",
                    an=None,  # -an: video-only source, no audio output
                    crf=self._ffmpeg_crf,
                    preset=self._ffmpeg_preset,
                )
                try:
                    _, ff_err = ffmpeg.run(
                        stream,
                        capture_stdout=True,
                        capture_stderr=True,
                        overwrite_output=True,
                        cmd=_FFMPEG_BIN,
                    )
                except ffmpeg.Error as ff_exc:
                    raise RuntimeError(
                        f"ffmpeg video encode failed: "
                        f"{ff_exc.stderr.decode(errors='replace') if ff_exc.stderr else str(ff_exc)}"
                    ) from ff_exc

                # Guard: ffmpeg can exit 0 but produce an empty file when the
                # input WebM lacks a valid initialization segment.
                if not os.path.exists(temp_mp4_path) or os.path.getsize(temp_mp4_path) == 0:
                    raise RuntimeError(
                        "ffmpeg produced an empty MP4 — input WebM may be missing "
                        "codec headers. stderr: "
                        + (ff_err.decode(errors='replace') if ff_err else '')
                    )

                filename = f"Cheating_Reports/{student_id}/{url_safe_ts}_{primary_type}.mp4"
                upload_url = f"{self._storage_url}/{filename}"

                def _do_put():
                    with open(temp_mp4_path, "rb") as fh:
                        resp = requests.put(
                            upload_url,
                            data=fh,
                            headers={"AccessKey": self._api_key},
                            timeout=60,
                        )
                    return resp

                resp = _do_put()
                if 200 <= resp.status_code < 300:
                    upload_status = "success"
                    evidence_url = f"{self._cdn_base_url}/{filename}"
                else:
                    time.sleep(self._retry_delay_s)
                    resp = _do_put()
                    if 200 <= resp.status_code < 300:
                        upload_status = "success"
                        evidence_url = f"{self._cdn_base_url}/{filename}"
                    else:
                        upload_status = "upload_failed"
                        reason_code = "UPLOAD_EXHAUSTED"

        except Exception as _enc_err:
            import logging
            logging.getLogger(__name__).error("clip_upload error: %s", _enc_err, exc_info=True)
            upload_status = "upload_failed"
            reason_code = "ENCODE_FAILED"

        finally:
            # ------------------------------------------------------------------
            # Clean up temp files unconditionally
            # ------------------------------------------------------------------
            if temp_webm_path and os.path.exists(temp_webm_path):
                try:
                    os.remove(temp_webm_path)
                except OSError:
                    pass
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                try:
                    os.remove(temp_mp4_path)
                except OSError:
                    pass
            if temp_mp3_path and os.path.exists(temp_mp3_path):
                try:
                    os.remove(temp_mp3_path)
                except OSError:
                    pass

        # ----------------------------------------------------------------------
        # Step 4: POST AiProctoringViolationEvent to the LMS backend
        # ----------------------------------------------------------------------
        token = str(metadata.get("token", "") or "")
        if self._base_url and token:
            self._post_violation_event(metadata, upload_status, evidence_url, reason_code, token)

        # Step 5: Append result to sessions/{sessionId}.jsonl for temporary local record
        self._append_to_session_log(metadata, upload_status, evidence_url, reason_code)

        return {
            "uploadStatus": upload_status,
            "evidenceUrl": evidence_url,
            "reasonCode": reason_code,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _append_to_session_log(self, metadata: dict, upload_status: str,
                                evidence_url: str | None, reason_code: str | None) -> None:
        """
        Append one JSON line to sessions/{sessionId}.jsonl.
        Failures are silently swallowed — this is best-effort.
        """
        try:
            session_id = str(metadata.get("sessionId", "unknown"))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sessions_dir = os.path.join(project_root, "sessions")
            os.makedirs(sessions_dir, exist_ok=True)
            log_path = os.path.join(sessions_dir, f"{session_id}.jsonl")

            record = {
                "type": "clip_upload",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "sessionId": session_id,
                "studentId": str(metadata.get("studentId", "")),
                "examAttemptId": str(metadata.get("examAttemptId", "")),
                "primaryViolationType": metadata.get("primaryViolationType", ""),
                "captureWindowStart": metadata.get("captureWindowStart", ""),
                "captureWindowEnd": metadata.get("captureWindowEnd", ""),
                "uploadStatus": upload_status,
                "evidenceUrl": evidence_url,
                "reasonCode": reason_code,
            }

            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass  # best-effort — never raise

    def _post_violation_event(
        self,
        metadata: dict,
        upload_status: str,
        evidence_url: str | None,
        reason_code: str | None,
        token: str,
    ) -> None:
        """
        POST one violation record to POST /api/CheatingReport/{reportId}/violations.
        Called only when a reportId is present in metadata and the CDN upload
        succeeded (evidenceUrl is required by the endpoint).
        The description covers every violation captured in the clip, not just
        the primary one.
        Failures are silently swallowed — never raise from here.
        """
        report_id = metadata.get("reportId")
        if report_id is None or not evidence_url:
            return

        all_violations: list = metadata.get("allViolations") or []

        if len(all_violations) == 1:
            full_description = (
                all_violations[0].get("description")
                or all_violations[0].get("violationType")
                or metadata.get("description", "Violation detected")
            )
        elif len(all_violations) > 1:
            parts = [
                v.get("description") or v.get("violationType") or "Violation"
                for v in all_violations
            ]
            full_description = "; ".join(parts)
        else:
            full_description = metadata.get("description", "Violation detected")

        violation_payload = {
            "evidenceUrl": evidence_url,
            "timestamp": metadata.get("captureWindowStart", ""),
            "description": full_description,
        }
        try:
            requests.post(
                f"{self._base_url}/api/CheatingReport/{report_id}/violations",
                json=violation_payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
                verify=False,
            )
        except Exception:
            pass  # fire-and-forget — never propagate
