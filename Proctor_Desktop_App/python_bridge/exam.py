"""
exam.py — Exam access blueprint for Lumina AI Python bridge.

Exposes one endpoint:
    POST /exam-access
        Body:    {"quizCode": str, "token": str}
        Success: ExamSession fields from the LMS (200 OK).
        Failure: BridgeExamError {"code": str, "message": str} (4xx / 5xx).

Security rules (constitution.md — Security by Default):
    - The access token is NEVER logged, printed, or included in any
      error response body or exception message.
    - Raw LMS error messages are NEVER forwarded to the caller.
      All error text is sanitised through map_lms_exam_error().
    - The BASE_URL used for LMS calls is read from the Flask app config,
      which is populated from the validated config.json (https:// enforced).
"""

import requests
from flask import Blueprint, jsonify, request, current_app
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

exam_bp = Blueprint("exam", __name__)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def map_lms_exam_error(status_code: int):
    """
    Map an LMS HTTP status code to a typed, sanitised BridgeExamError tuple.

    The pre-defined safe display strings are returned — raw LMS error text
    is never forwarded.

    Args:
        status_code: HTTP status code from the LMS response.

    Returns:
        Tuple of (response_dict, http_status_code) for use with jsonify().
    """
    if status_code == 404:
        return (
            {
                "code": "EXAM_NOT_FOUND",
                "message": "Exam code not found. Please check the code and try again.",
            },
            404,
        )
    if status_code == 409:
        return (
            {
                "code": "ALREADY_ATTEMPTED",
                "message": "You have already attempted this exam.",
            },
            409,
        )
    if status_code == 401:
        return (
            {
                "code": "UNAUTHORIZED",
                "message": "Your session has expired. Please log in again.",
            },
            401,
        )
    return (
        {
            "code": "BRIDGE_ERROR",
            "message": "Unable to reach the server. Please check your connection and try again.",
        },
        503,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@exam_bp.route("/exam-access", methods=["POST"])
def exam_access():
    """
    Start an exam attempt by validating the exam code against the LMS.

    Reads quizCode and token from the request body. The token is used
    exclusively as the Authorization header for the outbound LMS GET call —
    it is never logged, stored by this blueprint, or returned in any response.
    """
    data = request.get_json(silent=True) or {}
    quiz_code = (data.get("quizCode") or "").strip()
    token = data.get("token") or ""

    if not quiz_code or not token:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            400,
        )

    base_url = current_app.config["BASE_URL"]
    lms_url = f"{base_url}/api/QuizAttempts/attempt/{quiz_code}"

    try:
        response = requests.get(
            lms_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
            verify=False,
        )
    except requests.exceptions.RequestException:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            503,
        )

    if response.status_code == 200:
        session_data = response.json()

        # ── Create CheatingReport container for this attempt ────────────────
        # POST /api/CheatingReport/attempt/{attemptId} is idempotent — safe on
        # reconnect. We merge reportId into the session payload so the Electron
        # main process can pass it through to every subsequent clip upload
        # without making its own outbound call with the student token.
        attempt_id = session_data.get("attemptId")
        if attempt_id is not None:
            try:
                report_resp = requests.post(
                    f"{base_url}/api/CheatingReport/attempt/{attempt_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15,
                    verify=False,
                )
                if report_resp.status_code in (200, 201):
                    report_body = report_resp.json()
                    session_data["reportId"] = report_body.get("id")
            except Exception:
                pass  # non-blocking — exam proceeds even if report creation fails

        return jsonify(session_data), 200

    error_body, error_status = map_lms_exam_error(response.status_code)
    return jsonify(error_body), error_status


# ---------------------------------------------------------------------------
# Submit-exam helpers
# ---------------------------------------------------------------------------

def map_lms_submit_error(status_code: int):
    """
    Map an LMS HTTP status code to a typed, sanitised BridgeSubmitError tuple.

    Raw LMS error text is never forwarded.

    Args:
        status_code: HTTP status code from the LMS response.

    Returns:
        Tuple of (response_dict, http_status_code) for use with jsonify().
    """
    if status_code == 401:
        return (
            {
                "code": "UNAUTHORIZED",
                "message": "Your session has expired. Please log in again.",
            },
            401,
        )
    if status_code == 409:
        return (
            {
                "code": "ALREADY_SUBMITTED",
                "message": "This exam has already been submitted.",
            },
            409,
        )
    return (
        {
            "code": "BRIDGE_ERROR",
            "message": "Unable to reach the server. Please check your connection and try again.",
        },
        503,
    )


@exam_bp.route("/submit-exam", methods=["POST"])
def submit_exam():
    """
    Submit answers for a completed exam attempt.

    Reads attemptId, answers, and token from the request body.
    The token is used exclusively as the Authorization header for the outbound
    LMS POST call — it is never logged, stored, or returned in any response.

    Body: { "attemptId": int, "answers": list[{"questionId": int, "choiceId": int}], "token": str }
    Success: { "score": number, "total": number, "passed": bool, "questions": [...] } (200 OK)
    Failure: BridgeSubmitError {"code": str, "message": str} (4xx / 5xx)
    """
    data = request.get_json(silent=True) or {}
    attempt_id = data.get("attemptId")
    answers = data.get("answers")
    token = data.get("token") or ""

    if attempt_id is None or not isinstance(answers, list) or not token:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            400,
        )

    base_url = current_app.config["BASE_URL"]
    lms_url= f"{base_url}/api/QuizAttempts/submit/{attempt_id}"

    try:
        response = requests.post(
            lms_url,
            json={"answers": answers},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
            verify=False,
        )
    except requests.exceptions.RequestException:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            503,
        )

    if response.status_code == 200:
        return jsonify(response.json()), 200

    error_body, error_status = map_lms_submit_error(response.status_code)
    return jsonify(error_body), error_status


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def map_lms_result_error(status_code: int):
    """
    Map an LMS HTTP status code to a typed, sanitised BridgeResultError tuple.

    Raw LMS error text is never forwarded.

    Args:
        status_code: HTTP status code from the LMS response.

    Returns:
        Tuple of (response_dict, http_status_code) for use with jsonify().
    """
    if status_code == 401:
        return (
            {
                "code": "UNAUTHORIZED",
                "message": "Your session has expired. Please log in again.",
            },
            401,
        )
    if status_code == 404:
        return (
            {
                "code": "RESULT_NOT_FOUND",
                "message": "Result not found. Please contact your instructor.",
            },
            404,
        )
    return (
        {
            "code": "BRIDGE_ERROR",
            "message": "Unable to reach the server. Please check your connection and try again.",
        },
        503,
    )


@exam_bp.route("/result", methods=["POST"])
def get_result():
    """
    Retrieve the result for a completed exam attempt.

    Reads attemptId and token from the request body. The token is used
    exclusively as the Authorization header for the outbound LMS GET call —
    it is never logged, stored by this blueprint, or returned in any response.

    Body:    { "attemptId": int, "token": str }
    Success: SubmitResult fields from the LMS (200 OK).
    Failure: BridgeResultError {"code": str, "message": str} (4xx / 5xx).
    """
    data = request.get_json(silent=True) or {}
    attempt_id = data.get("attemptId")
    token = data.get("token") or ""

    if attempt_id is None or not token:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            400,
        )

    base_url = current_app.config["BASE_URL"]
    lms_url = f"{base_url}/api/QuizAttempts/result/{attempt_id}"

    try:
        response = requests.get(
            lms_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
            verify=False,
        )
    except requests.exceptions.RequestException:
        return (
            jsonify(
                {
                    "code": "BRIDGE_ERROR",
                    "message": "Unable to reach the server. Please check your connection and try again.",
                }
            ),
            503,
        )

    if response.status_code == 200:
        return jsonify(response.json()), 200

    error_body, error_status = map_lms_result_error(response.status_code)
    return jsonify(error_body), error_status
