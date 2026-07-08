"""
auth.py — Authentication blueprint for Lumina AI Python bridge.

Exposes one endpoint:
    POST /login
        Body:    {"email": str, "password": str}
        Success: LoginResponse fields from the LMS (200 OK).
        Failure: BridgeLoginError {"code": str, "message": str} (4xx / 5xx).

Security rules (constitution.md — Security by Default):
    - The student password is NEVER logged, printed, or included in any
      error response body or exception message.
    - Raw LMS errorMessage strings are NEVER forwarded to the caller.
      All error text is sanitised through map_lms_error().
    - The BASE_URL used for LMS calls is read from the Flask app config,
      which is populated from the validated config.json (https:// enforced).
"""

import requests
from flask import Blueprint, jsonify, request, current_app
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
auth_bp = Blueprint("auth", __name__)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def map_lms_error(status_code: int, error_message: str) -> dict:
    """
    Map a raw LMS error to a typed, sanitised BridgeLoginError dict.

    The *error_message* string is inspected with case-insensitive substring
    matching, but its raw value is NEVER forwarded in the returned dict.
    Only the pre-defined safe display strings are returned.

    Args:
        status_code:   HTTP status code from the LMS (unused in matching logic,
                       kept for future use / logging at a higher level).
        error_message: Raw errorMessage string from the LMS response body.

    Returns:
        dict with keys "code" (str) and "message" (str).
    """
    msg = error_message.lower()

    if "invalid email/password" in msg:
        return {
            "code": "INVALID_CREDENTIALS",
            "message": "Invalid email or password",
        }
    if "is not confirmed" in msg:
        return {
            "code": "EMAIL_NOT_CONFIRMED",
            "message": "Please confirm your email address before logging in.",
        }
    if "is locked out" in msg:
        return {
            "code": "LOCKED_OUT",
            "message": "Your account is temporarily locked. Please contact your administrator.",
        }
    if "is disabled" in msg:
        return {
            "code": "ACCOUNT_DISABLED",
            "message": "Your account has been disabled. Please contact your administrator.",
        }

    return {
        "code": "BRIDGE_ERROR",
        "message": "Unable to reach the server. Please check your connection and try again.",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /login — Proxy student authentication to the LMS.

    Request body (JSON):
        {"email": str, "password": str}

    Responses:
        200  — Full LMS LoginResponse relayed as-is.
        400  — BRIDGE_ERROR when email or password is blank.
        4xx  — Mapped BridgeLoginError when LMS rejects credentials.
        503  — BRIDGE_ERROR when the LMS is unreachable (network failure).

    Security: the password value is NEVER included in any log statement,
    print() call, exception message, or error response body.
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    # Password intentionally not stripped — whitespace may be significant
    password = data.get("password", "")

    if not email or not password:
        return jsonify({
            "code": "BRIDGE_ERROR",
            "message": "Unable to reach the server. Please check your connection and try again.",
        }), 400

    base_url = current_app.config.get("BASE_URL", "").rstrip("/")
    lms_url = f"{base_url}/api/Authuntication/login"

    try:
        response = requests.post(
            lms_url,
            json={"email": email, "password": password},
            timeout=15,
            verify=False,  # local dev: self-signed cert
        )
    except requests.exceptions.RequestException:
        # Network-level failure (DNS, timeout, refused connection, etc.)
        # Do NOT include exception details — they may contain URL / credential fragments.
        return jsonify(map_lms_error(0, "")), 503

    if response.status_code == 200:
        return jsonify(response.json()), 200

    # Non-200 from LMS — extract and map the error message
    error_body = {}
    try:
        error_body = response.json()
    except Exception:  # noqa: BLE001
        pass

    # Raw LMS errorMessage is consumed here but NEVER forwarded in the response
    raw_error = error_body.get("errorMessage", "")
    return jsonify(map_lms_error(response.status_code, raw_error)), response.status_code
