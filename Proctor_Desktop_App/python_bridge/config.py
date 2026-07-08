"""
config.py — Load and validate config.json for the Lumina AI Python bridge.

Usage:
    from config import load_config, AppConfig, ConfigError

Raises ConfigError with a typed code and human-readable message for every
failure case. All errors are also emitted to stderr as a single-line JSON
string so Electron can parse them from the subprocess stderr stream.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when config.json is missing, malformed, or contains invalid values."""

    CODES = {
        "FILE_NOT_FOUND",
        "INVALID_JSON",
        "MISSING_BASE_URL",
        "INSECURE_PROTOCOL",
        "INVALID_PORT",
        "MISSING_MODAL_CONFIG",
        "MISSING_MODAL_ENDPOINT",
    }

    def __init__(self, code: str, message: str) -> None:
        if code not in self.CODES:
            raise ValueError(f"Unknown ConfigError code: {code!r}")
        self.code = code
        self.message = message
        super().__init__(message)

    def emit_stderr(self) -> None:
        """Write a single-line JSON error to stderr for Electron to parse."""
        payload = json.dumps({"error": self.code, "message": self.message})
        print(payload, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    base_url: str
    python_port: int = field(default=5050)
    modal: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    orchestration: Dict[str, Any] = field(default_factory=dict)
    clip_recording: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> AppConfig:
    """
    Read and validate config.json at *config_path*.

    Returns an AppConfig on success.
    Raises ConfigError (and emits to stderr) on any failure.

    Validation rules (from contracts/config-schema.md):
      - File must exist and be readable.
      - File must contain valid JSON.
      - 'baseUrl' key must be present and non-empty.
      - 'baseUrl' must start with 'https://' (INSECURE_PROTOCOL otherwise).
      - 'pythonPort', when present, must be an integer in range [1024, 65535].
    """
    # Rule 1: File must exist
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        err = ConfigError(
            "FILE_NOT_FOUND",
            f"config.json not found at path: {config_path}",
        )
        err.emit_stderr()
        raise err
    except OSError as exc:
        err = ConfigError(
            "FILE_NOT_FOUND",
            f"Could not read config.json: {exc}",
        )
        err.emit_stderr()
        raise err

    # Rule 2: Must be valid JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        err = ConfigError(
            "INVALID_JSON",
            f"config.json contains invalid JSON: {exc.msg} (line {exc.lineno})",
        )
        err.emit_stderr()
        raise err

    # Rule 3: 'baseUrl' must be present and non-empty
    base_url: Optional[str] = data.get("baseUrl")
    if not base_url or not isinstance(base_url, str) or not base_url.strip():
        err = ConfigError(
            "MISSING_BASE_URL",
            "config.json must contain a non-empty 'baseUrl' string.",
        )
        err.emit_stderr()
        raise err

    base_url = base_url.strip().rstrip("/")

    # Rule 4: baseUrl must use HTTPS
    if not base_url.startswith("https://"):
        err = ConfigError(
            "INSECURE_PROTOCOL",
            "baseUrl must use HTTPS (https://). HTTP URLs are not permitted.",
        )
        err.emit_stderr()
        raise err

    # Rule 5: pythonPort must be in valid range when provided
    python_port = data.get("pythonPort", 5050)
    if not isinstance(python_port, int) or isinstance(python_port, bool):
        err = ConfigError(
            "INVALID_PORT",
            f"pythonPort must be an integer, got {type(python_port).__name__!r}.",
        )
        err.emit_stderr()
        raise err

    if not (1024 <= python_port <= 65535):
        err = ConfigError(
            "INVALID_PORT",
            f"pythonPort must be between 1024 and 65535, got {python_port}.",
        )
        err.emit_stderr()
        raise err

    # Capture optional fields
    modal = data.get("modal", {})
    services = data.get("services", {})
    orchestration = data.get("orchestration", {})
    clip_recording = data.get("clip_recording", {})

    def require_non_empty_str(value: Any, field_name: str, code: str) -> str:
        if not isinstance(value, str) or not value.strip():
            err = ConfigError(code, f"config.json missing required field: {field_name}.")
            err.emit_stderr()
            raise err
        return value.strip()

    if not isinstance(modal, dict):
        err = ConfigError("MISSING_MODAL_CONFIG", "config.json modal config must be an object.")
        err.emit_stderr()
        raise err

    require_non_empty_str(modal.get("token_id"), "modal.token_id", "MISSING_MODAL_CONFIG")
    require_non_empty_str(modal.get("token_secret"), "modal.token_secret", "MISSING_MODAL_CONFIG")
    require_non_empty_str(modal.get("user_id"), "modal.user_id", "MISSING_MODAL_CONFIG")

    if not isinstance(services, dict):
        err = ConfigError("INVALID_JSON", "config.json services config must be an object.")
        err.emit_stderr()
        raise err

    for service_name in ["face-recognition", "object-detection"]:
        service_cfg = services.get(service_name)
        if not isinstance(service_cfg, dict):
            err = ConfigError(
                "MISSING_MODAL_ENDPOINT",
                f"config.json missing services.{service_name} config.",
            )
            err.emit_stderr()
            raise err
        require_non_empty_str(
            service_cfg.get("endpoint_url"),
            f"services.{service_name}.endpoint_url",
            "MISSING_MODAL_ENDPOINT",
        )

    # Optional: Basic validation for local service settings if they exist
    for service_name in ["eye-gaze", "speech-detection"]:
        if service_name in services:
            s_cfg = services[service_name]
            if not isinstance(s_cfg, dict):
                err = ConfigError("INVALID_JSON", f"Service config for {service_name} must be an object.")
                err.emit_stderr()
                raise err

    return AppConfig(
        base_url=base_url,
        python_port=python_port,
        modal=modal,
        services=services,
        orchestration=orchestration,
        clip_recording=clip_recording,
    )
