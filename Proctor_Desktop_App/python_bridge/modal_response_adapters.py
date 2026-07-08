"""
Map Modal FastAPI JSON bodies into bridge DetectionEvent dicts.

The desktop bridge validates every predict() result against ai-service-contract.json.
Hosted Modal routes return module-specific shapes — normalize here.

Object-detection adapters
~~~~~~~~~~~~~~~~~~~~~~~~~
  adapt_object_yolo_json()  — /analysis/object-frame  (YOLO, multipart field: 'image')
                               Response: {id, timestamp, flag, propability, evidence}
  adapt_object_modal_json() — /analysis/detect_objects (OWL-ViT, multipart field: 'file')
                               Response: {id, timestamp, probability, evidence}
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_BRIDGE_SERVICES = frozenset(
    {"eye-gaze", "object-detection", "face-recognition", "speech-detection"}
)


def is_bridge_detection_event(data: Any) -> bool:
    """True if *data* is already a normalized DetectionEvent (including Modal error events)."""
    if not isinstance(data, dict):
        return False
    if data.get("service") not in _BRIDGE_SERVICES:
        return False
    if not isinstance(data.get("payload"), dict):
        return False
    for key in ("timestamp", "confidence", "sessionId"):
        if key not in data:
            return False
    return True


def _parse_percent_or_fraction(val: Any) -> float:
    """Parse API fields that may be 0.85, 85, or \"85%\" into [0, 1]."""
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        x = float(val)
        if x > 1.0:
            return max(0.0, min(1.0, x / 100.0))
        return max(0.0, min(1.0, x))
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        try:
            x = float(s)
        except ValueError:
            return 0.0
        if x > 1.0:
            return max(0.0, min(1.0, x / 100.0))
        return max(0.0, min(1.0, x))
    return 0.0


# Sentinel used to distinguish "threshold not provided" from threshold=0.0.
_THRESHOLD_DEFAULT = 0.5


def _infer_face_is_matched(inner: Dict[str, Any], threshold: float = _THRESHOLD_DEFAULT) -> bool:
    """
    Map Modal `face_recognition` object to bridge `is_matched` (True = authorised / OK).

    Threshold-based decision layer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For the two similarity-based outcomes — 'Authorised person verified' and
    'Face does not match reference' — the final decision is made by comparing the
    returned `probability` value against *threshold* (read from
    ``config.json → services.face-recognition.probability_threshold``).

      probability >= threshold  →  is_matched = True   (face accepted)
      probability <  threshold  →  is_matched = False  (face rejected)

    All other outcomes (no face, multiple faces, spoof, no enrollment, no `flag`
    field present) are **not affected** by the threshold — their result is returned
    directly from the API evidence / flag fields as before.

    The logged payload is unchanged: only the value of `is_matched` may differ
    when the threshold overrides the API's own verdict.

    The FaceRecognition pipeline in AI/Models/Face_Recognition_Service/face_recognition.py
    sets `flag`: True means suspicious (spoof, mismatch, no face, etc.), False means match.
    Some deployments strip `flag`; never default missing `flag` to True (that marks everyone
    as not matched). Use `result` or `evidence` when `flag` is absent.
    """
    if "flag" in inner and inner["flag"] is not None:
        return not bool(inner["flag"])

    result = inner.get("result")
    if isinstance(result, str):
        r = result.strip()
        if r == "No Cheating":
            return True
        if r == "Cheating Detected":
            return False

    ev = (inner.get("evidence") or "").lower()

    # ── Threshold-gated cases ────────────────────────────────────────────────
    # Only 'verified' and 'mismatch' carry a meaningful cosine-similarity
    # probability that can be compared against the project threshold.
    # For all other cases the API result is authoritative (see below).
    is_verified_case  = "authorised person verified" in ev or "authorized person verified" in ev
    is_mismatch_case  = "face does not match" in ev or "does not match reference" in ev

    if is_verified_case or is_mismatch_case:
        prob = _parse_percent_or_fraction(inner.get("probability"))
        return prob >= threshold

    # ── Non-threshold cases — API verdict is authoritative ───────────────────
    if "no enrollment" in ev:
        return False
    if "no face detected" in ev:
        return False
    if "multiple faces" in ev:
        return False
    if "spoof detected" in ev:
        return False

    try:
        faces = int(inner.get("num_faces", 0) or 0)
    except (TypeError, ValueError):
        faces = 0
    if faces < 1:
        return False

    prob = _parse_percent_or_fraction(inner.get("probability"))
    return prob >= threshold


def adapt_face_modal_json(
    body: Dict[str, Any],
    create_detection_event: Callable[[float, Dict[str, Any]], Dict[str, Any]],
    threshold: float = _THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    """
    Adapt a Modal /analysis/verify-file JSON response into a bridge DetectionEvent.

    Args:
        body: Raw JSON dict from Modal (must contain a ``face_recognition`` key).
        create_detection_event: Factory from the calling AIService.
        threshold: Project-side probability threshold for the match/mismatch
            decision.  Read from
            ``config.json → services.face-recognition.probability_threshold``
            and forwarded here by FaceRecognitionService.  Default: 0.5.
    """
    inner = body.get("face_recognition")
    if not isinstance(inner, dict):
        raise ValueError("Modal face response missing face_recognition object")

    # Pass the configurable threshold into the decision layer.
    # Logging payload (conf, faces_count, is_matched) is unchanged — only
    # the *value* of is_matched may differ from the API's own verdict.
    is_matched = _infer_face_is_matched(inner, threshold=threshold)
    conf = _parse_percent_or_fraction(inner.get("probability"))
    if conf <= 0.0 and is_matched:
        conf = _parse_percent_or_fraction(inner.get("match_similarity"))
    if conf <= 0.0:
        conf = 0.01 if is_matched else 0.0
    conf = max(0.0, min(1.0, conf))

    try:
        faces = int(inner.get("num_faces", 0) or 0)
    except (TypeError, ValueError):
        faces = 0

    # liveness_score: anti-spoofing signal (MiniFASNetV2). Returned as "89.21%".
    # quality: frame sharpness/brightness signal. Returned as "75.00%".
    # Both are parsed to [0, 1] floats for uniform downstream handling.
    liveness_score: float = _parse_percent_or_fraction(inner.get("liveness_score"))
    quality: float        = _parse_percent_or_fraction(inner.get("quality"))

    # Extract the full outcome message from the API for logging.
    # Covers all five outcome cases:
    #   "Authorised person verified (similarity: X)"  → is_matched True
    #   "No enrollment found for session '...'"        → is_matched False
    #   "No face detected in frame"                   → is_matched False
    #   "Multiple faces detected: N faces in frame"   → is_matched False
    #   "Face does not match reference identity (similarity: X)" → is_matched False
    recognition_message: str = (inner.get("evidence") or "").strip()

    # Derive is_spoof from the evidence string so the orchestrator can emit a
    # dedicated SPOOF_DETECTED alert rather than the generic NO_FACE_DETECTED one.
    is_spoof: bool = "spoof detected" in recognition_message.lower()

    payload: Dict[str, Any] = {
        "is_matched": is_matched,
        "faces_count": faces,
        "recognition_message": recognition_message,
        "is_spoof": is_spoof,
        "liveness_score": liveness_score,
        "quality": quality,
    }
    return create_detection_event(conf, payload)


def adapt_object_modal_json(
    body: Dict[str, Any],
    create_detection_event: Callable[[float, Dict[str, Any]], Dict[str, Any]],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Adapt a Modal /analysis/detect_objects (OWL-ViT) JSON response into a bridge DetectionEvent.

    Threshold-based decision layer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    OWL-ViT returns a ``probability`` field representing the max confidence across
    all detected objects in the frame (already filtered server-side at >0.1).
    The ``suspicious`` flag is only raised when:

      - At least one restricted object appears in ``evidence``, **AND**
      - ``probability >= threshold``

    If ``probability < threshold`` the objects list is preserved in the payload
    (for logging) but ``suspicious`` is set to ``False`` so no alert is raised.

    Args:
        body: Raw JSON dict from Modal OWL-ViT endpoint.
        create_detection_event: Factory from the calling AIService.
        threshold: Project-side confidence threshold.  Read from
            ``config.json -> services.object-detection.probability_threshold``.
            Default: 0.3  (above server minimum of 0.1).
    """
    if not isinstance(body.get("evidence"), str):
        raise ValueError("Modal object response missing evidence string")

    prob = _parse_percent_or_fraction(body.get("probability"))
    evidence = body["evidence"].strip()
    objects: list[str] = []
    if "Detected:" in evidence:
        rest = evidence.split("Detected:", 1)[1].strip()
        if rest and "no restricted" not in evidence.lower():
            objects = [x.strip() for x in rest.split(",") if x.strip()]

    # Objects present in the evidence but probability below threshold -> log
    # the objects for forensic purposes but do NOT raise the suspicious flag.
    suspicious = len(objects) > 0 and prob >= threshold
    payload: Dict[str, Any] = {
        "objects": objects,
        "count": len(objects),
        "suspicious": suspicious,
    }
    return create_detection_event(prob, payload)


def adapt_object_yolo_json(
    body: Dict[str, Any],
    create_detection_event: Callable[[float, Dict[str, Any]], Dict[str, Any]],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Adapt a Modal /analysis/object-frame (YOLO) JSON response into a bridge DetectionEvent.

    YOLO response shape
    ~~~~~~~~~~~~~~~~~~~
    The YOLO endpoint returns::

        {
          "id": 2,
          "timestamp": "<ISO-8601>",
          "flag": <bool>,        # True = cheating object detected
          "propability": <float>, # note: typo in server code ('propability')
          "evidence": "<str>"    # comma-separated class names, or "None"
        }

    Threshold-based decision layer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``flag=True`` means YOLO's built-in confidence (>=0.4 server-side) already
    accepted the detection.  The project-side ``threshold`` adds a second gate
    on top of YOLO's own ``propability`` (max conf across flagged boxes):

      flag=True  AND  propability >= threshold  →  suspicious = True
      flag=True  AND  propability <  threshold  →  objects logged, suspicious = False
      flag=False                                →  suspicious = False

    Args:
        body: Raw JSON dict from the YOLO /analysis/object-frame endpoint.
        create_detection_event: Factory from the calling AIService.
        threshold: Project-side confidence threshold read from
            ``config.json -> services.object-detection.probability_threshold``.
            Default: 0.3.
    """
    # 'propability' is the server-side typo; fall back to 'probability' if ever fixed.
    raw_prob = body.get("propability", body.get("probability", 0.0))
    prob = _parse_percent_or_fraction(raw_prob)

    flag: bool = bool(body.get("flag", False))
    evidence_str: str = (body.get("evidence") or "None").strip()

    # Parse comma-separated object names; treat bare "None" as empty.
    objects: list[str] = []
    if flag and evidence_str.lower() not in ("", "none"):
        objects = [x.strip() for x in evidence_str.split(",") if x.strip()]

    # Apply project-side threshold on top of YOLO's internal 0.4 gate.
    suspicious = flag and len(objects) > 0 and prob >= threshold

    payload: Dict[str, Any] = {
        "objects": objects,
        "count": len(objects),
        "suspicious": suspicious,
    }
    return create_detection_event(prob, payload)
