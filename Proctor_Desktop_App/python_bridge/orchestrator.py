import json
import os
import time
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Callable

class ProctoringOrchestrator:
    """
    Fuses AI signals into high-level alerts and a running risk score.
    Persists all events to a Session Log (JSONL).

    Strict separation of responsibilities:
    - face-detection handles "missing face" scoring (weight 15) and alerts.
    - face-recognition handles "impersonation / spoof" scoring (weight 50) and alerts.
    This guarantees that the exact same offense is always weighted the same way,
    regardless of which pipeline happens to process it.
    """

    def __init__(self, config: Dict[str, Any], emit_callback: Callable[[Dict[str, Any]], None]):
        self.config = config.get("orchestration", {})
        self.rules_config = self.config.get("rules", {})
        self.risk_score_decay = self.config.get("risk_score_decay", 0.9997)
        self.emit_callback = emit_callback

        # Always write session logs into the project-level `sessions/` directory
        # (not dependent on the router process working directory).
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.sessions_dir = os.path.join(project_root, "sessions")

        self.current_session_id: Optional[str] = None
        self.log_file = None

        # State tracking
        self.risk_score = 0.0
        self.active_violations: Dict[str, bool] = {
            "eye-gaze":         False,
            "face-detection":   False,
            "face-recognition": False,
            "speech-detection": False,
            "object-detection": False,
        }
        # Guard: SPEECH_CHEATING_FLAGGED must fire exactly once per session.
        # Once emitted it stays True for the session lifetime so subsequent
        # predict() polls with is_cheater=True do not re-emit the alert.
        self._speech_cheater_flagged: bool = False

        # Timers for time-based rules
        self.missing_face_start: Optional[float]     = None
        self.off_screen_gaze_start: Optional[float]  = None

        # Lock for thread-safe logging
        self._log_lock = asyncio.Lock()

    def set_session(self, session_id: str):
        """Initialize logging for a new session."""
        if self.current_session_id == session_id:
            return

        self.current_session_id = session_id
        self.risk_score = 0.0
        self._speech_cheater_flagged = False

        # Ensure sessions directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        log_path = os.path.join(self.sessions_dir, f"{session_id}.jsonl")
        self.log_file_path = log_path

    async def on_detection_event(self, event: Dict[str, Any]):
        """Entry point for incoming AI detections."""
        if not self.current_session_id:
            self.set_session(event.get("sessionId", "default-session"))

        # Process rules and annotate risk score BEFORE logging so that any
        # annotation (e.g. risk_suppressed) is captured in the log entry.
        await self._process_rules(event)
        await self._update_risk_score(event)
        await self._log_event(event)

    async def _process_rules(self, event: Dict[str, Any]):
        service = event.get("service")
        payload = event.get("payload", {})
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # --- Rule: Object Detection (Immediate) ---
        if service == "object-detection":
            if payload.get("suspicious"):
                await self._emit_alert(
                    "UNAUTHORIZED_OBJECT", "high",
                    f"Suspicious object detected: {', '.join(payload.get('objects', []))}",
                    event
                )

        # --- Rule: Speech Detection (Strike-based) ---
        if service == "speech-detection":
            new_violations = payload.get("new_violations", [])
            is_cheater = payload.get("is_cheater", False)
            total_strikes = payload.get("total_strikes", 0)

            for violation in new_violations:
                await self._emit_alert(
                    "SPEECH_DETECTED",
                    "high" if is_cheater else "medium",
                    f"Student spoke for {violation['duration_seconds']}s. "
                    f"Strike {violation['strike_number']}/{total_strikes}.",
                    event
                )

            if is_cheater and not self._speech_cheater_flagged:
                self._speech_cheater_flagged = True
                await self._emit_alert(
                    "SPEECH_CHEATING_FLAGGED",
                    "critical",
                    f"Student flagged as cheater after {total_strikes} speech violations.",
                    event
                )

        # --- Rule: Eye Gaze (Time-based logic managed by localMain.py) ---
        if service == "eye-gaze":
            status = payload.get("status")

            if status == "away":
                if not self.active_violations["eye-gaze"]:
                    self.active_violations["eye-gaze"] = True
                    await self._emit_alert(
                        "GAZE_OFF_SCREEN", "low",
                        "Gaze away from screen (detected by local model)", event
                    )
            else:
                # "on-screen", "no-face", "initializing" — all clear the away violation.
                # "no-face" is already scored separately by face-detection; do not
                # double-count it here. Clearing the flag allows the next genuine
                # "away" event to re-trigger an alert.
                self.active_violations["eye-gaze"] = False

        # --- Rule: Face Detection (Missing face timer) ---
        if service == "face-detection":
            face_detected = payload.get("face_detected", True)
            threshold = self.rules_config.get("face-detection", {}).get("missing_threshold_seconds", 5)

            if not face_detected:
                self.active_violations["face-detection"] = True
                if self.missing_face_start is None:
                    self.missing_face_start = now
                elif now - self.missing_face_start > threshold:
                    await self._emit_alert(
                        "NO_FACE_DETECTED", "high",
                        f"Student face missing for > {threshold}s", event
                    )
                    self.missing_face_start = now
            else:
                self.active_violations["face-detection"] = False
                self.missing_face_start = None

        # --- Rule: Face Recognition (Spoof + Impersonation) ---
        if service == "face-recognition":
            # Spoof: immediate critical alert — does not wait for any timer.
            if payload.get("is_spoof"):
                self.active_violations["face-recognition"] = True
                await self._emit_alert(
                    "SPOOF_DETECTED", "critical",
                    "Anti-spoofing check failed — a non-live face was presented.",
                    event
                )

            # Impersonation: recognition ran and the person does not match the reference.
            # Using 'elif' ensures we don't fire an 'Unauthorized Person' alert if we 
            # already fired a 'Spoof Detected' alert for the same frame.
            elif not payload.get("is_matched", True) and payload.get("recognition_ran", False):
                self.active_violations["face-recognition"] = True
                await self._emit_alert(
                    "UNAUTHORIZED_PERSON", "critical",
                    "Unrecognized person detected at the workstation.",
                    event
                )
            elif payload.get("is_matched", True) and payload.get("recognition_ran", False):
                self.active_violations["face-recognition"] = False

    async def _update_risk_score(self, event: Dict[str, Any]):
        """
        Additive weighted-sum risk calculation.

        Strict Mutually Exclusive Separation:
        - face-detection ONLY scores for "missing face" scenarios (weight 15).
        - face-recognition ONLY scores for "impersonation" or "spoof" scenarios
          (face is present, but identity does not match or a spoof was detected) (weight 50).
        This eliminates double-counting without complex time-window deduplication.
        """
        service = event.get("service")
        payload = event.get("payload", {})

        weight = self.rules_config.get(service, {}).get("weight", 10)

        is_suspicious = False

        if service == "eye-gaze" and payload.get("status") == "away":
            is_suspicious = True

        elif service == "face-detection":
            if not payload.get("face_detected", True):
                is_suspicious = True

        elif service == "face-recognition" and payload.get("recognition_ran", False):
            # Score for EITHER a spoof OR an unmatched identity — but not both at once.
            # is_spoof takes priority (higher severity) over a plain mismatch.
            if payload.get("is_spoof"):
                is_suspicious = True
            elif not payload.get("is_matched", True):
                is_suspicious = True

        elif service == "speech-detection" and len(payload.get("new_violations", [])) > 0:
            is_suspicious = True

        elif service == "object-detection" and payload.get("suspicious"):
            is_suspicious = True

        if is_suspicious:
            self.risk_score = min(100.0, self.risk_score + (weight * 0.1))
        else:
            self.risk_score *= self.risk_score_decay

        score_update = {
            "type":      "riskScore",
            "score":     int(round(self.risk_score)),
            "trend":     "rising" if is_suspicious else "falling",
            "service":   service,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.emit_callback(score_update)
        await self._log_event(score_update)

    async def _emit_alert(self, code: str, severity: str, message: str, evidence: Dict[str, Any]):
        alert = {
            "type":      "alert",
            "code":      code,
            "severity":  severity,
            "message":   message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sessionId": self.current_session_id,
            "questionId": evidence.get("questionId"),
            "evidence": {
                "service":    evidence.get("service"),
                "confidence": evidence.get("confidence"),
            },
        }
        self.emit_callback(alert)
        await self._log_event(alert)

    async def _log_event(self, event: Dict[str, Any]):
        """Write event to JSONL file."""
        if not self.current_session_id:
            return

        async with self._log_lock:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                import sys
                print(f"Logging error: {str(e)}", file=sys.stderr)