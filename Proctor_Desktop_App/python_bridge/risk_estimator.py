"""
Overall Risk Estimation — Post-Session Scoring Backend

Implements the risk formula from the research paper:
  "A Visual Analytics Approach to Facilitate the Proctoring of Online Exams"

  risk^sq = Σ  w_t × n^sq_t        (t ∈ {f, h, c, b})

Where:
  n^sq_t  = raw occurrence count of cheating-type t for student s on question q
             (IMPORTANT: these raw counts are stored as-is in this report;
              min-max normalisation to (0,1) is performed LATER, across all
              students in the exam, not within a single session log)
  w_t     = proctor-customisable weight per category (default 1.0)
  t maps to the 4 behaviour categories:
    f → face_absence_mismatch   (face-detection + face-recognition)
    h → suspicious_movement     (eye-gaze)
    c → conversation_noise      (speech-detection)
    b → forbidden_objects       (object-detection)

This script computes either:
  (a) A whole-session report  — raw counts + pre-normalisation weighted sum
  (b) A per-question report   — per AI-service raw counts keyed by question

Mode (a) — whole-session pipeline:
  1. Parse every DetectionEvent in the session log.
  2. Map the 5 AI services → 4 behaviour categories (f, h, c, b).
  3. Count raw anomaly occurrences n_t per category.
  4. Compute pre-normalisation risk = Σ (w_t × n_t).
  5. Write a structured JSON report containing raw_counts and risk_score.
     NOTE: Normalise raw_counts across the student population before
           comparing students or ranking risk levels.

Mode (b) — per-question pipeline:
  1. Parse every DetectionEvent in the session log.
  2. Pre-fill all N questions (1 … N) with zero counts for every AI service.
  3. For each suspicious event, read its `questionId` field and increment
     the matching service counter for that question.
     Events without a `questionId` go into an `unassigned` bucket.
  4. Write a structured JSON report keyed by `question_1` … `question_N`.

Usage:
  # Whole-session risk score
  python risk_estimator.py <session_log.jsonl> [--output <path.json>]
      [--student-id <id>] [--exam-id <id>]
      [--wf <float>] [--wh <float>] [--wc <float>] [--wb <float>]

  # Per-question breakdown
  python risk_estimator.py <session_log.jsonl> --by-question --total-questions <N>
      [--output <path.json>] [--student-id <id>] [--exam-id <id>]

If --output is omitted the report is written next to the input file as:
  <session_id>_risk_report.json          (whole-session)
  <session_id>_question_report.json      (per-question)
"""

import json
import os
import sys
import argparse
import datetime
from typing import Dict, Any, List, Optional


# ─── Behaviour categories — correspond to {f, h, c, b} in the paper ────
CATEGORY_FACE   = "face_absence_mismatch"    # n_f  (face-detection + face-recognition)
CATEGORY_MOVE   = "suspicious_movement"      # n_h  (eye-gaze)
CATEGORY_CONV   = "conversation_noise"       # n_c  (speech-detection)
CATEGORY_OBJ    = "forbidden_objects"        # n_b  (object-detection)

# Order must match the paper's t ∈ {f, h, c, b}
ALL_CATEGORIES = [CATEGORY_FACE, CATEGORY_MOVE, CATEGORY_CONV, CATEGORY_OBJ]

# ─── Per-service keys used by the question-level report ─────────────────
# Ordered to match the target output structure.
SERVICE_KEYS = [
    "face_detection",
    "face_recognition",
    "eye_gaze",
    "speech_detection",
    "object_detection",
]

# Mapping from raw service names in the JSONL to the SERVICE_KEYS above.
_SERVICE_NAME_TO_KEY: Dict[str, str] = {
    "face-detection":   "face_detection",
    "face-recognition": "face_recognition",
    "eye-gaze":         "eye_gaze",
    "speech-detection": "speech_detection",
    "object-detection": "object_detection",
}


# ─── Event classification ──────────────────────────────────────────────

def _is_suspicious_event(event: Dict[str, Any]) -> Optional[str]:
    """Return the behaviour category if the event is suspicious, else None.

    The rules intentionally mirror ProctoringOrchestrator._update_risk_score
    so that the post-session report agrees with the real-time score.
    """
    service = event.get("service")
    payload = event.get("payload", {})

    # --- Face Detection: missing face ---
    if service == "face-detection":
        if not payload.get("face_detected", True):
            return CATEGORY_FACE

    # --- Face Recognition: impersonation / spoof ---
    elif service == "face-recognition":
        if payload.get("is_spoof"):
            return CATEGORY_FACE
        if not payload.get("is_matched", True) and payload.get("recognition_ran", False):
            return CATEGORY_FACE

    # --- Object Detection: prohibited object ---
    elif service == "object-detection":
        if payload.get("suspicious"):
            return CATEGORY_OBJ

    # Eye-gaze and Speech-detection are counted from alerts, not raw detections.
    return None

def _is_alert_suspicious(event: Dict[str, Any]) -> Optional[str]:
    """Map orchestrator-generated AlertEvents to a category.
    Only handles alerts for services that aren't already counted via raw detections.
    """
    code = event.get("code", "")
    code_map = {
        "GAZE_OFF_SCREEN":          CATEGORY_MOVE,
        "SPEECH_DETECTED":          CATEGORY_CONV,
        "SPEECH_CHEATING_FLAGGED":  CATEGORY_CONV,
    }
    return code_map.get(code)


def _suspicious_service_key(event: Dict[str, Any]) -> Optional[str]:
    """Return the SERVICE_KEY string if this DetectionEvent is suspicious.

    Used by the per-question analyser, which tracks counts at the individual
    AI-service level (not the 4-category level used by the session scorer).
    """
    service = event.get("service")
    key = _SERVICE_NAME_TO_KEY.get(service or "")
    if key is None:
        return None

    payload = event.get("payload", {})

    if service == "face-detection":
        return key if not payload.get("face_detected", True) else None

    if service == "face-recognition":
        if payload.get("is_spoof") or (
            not payload.get("is_matched", True) and payload.get("recognition_ran", False)
        ):
            return key
        return None

    if service == "object-detection":
        return key if payload.get("suspicious") else None

    # eye-gaze and speech-detection are handled via alerts now
    return None


# ─── Core logic ─────────────────────────────────────────────────────────

def count_anomalies(log_path: str) -> Dict[str, int]:
    """Read a JSONL session log and return raw anomaly counts per category."""
    counts: Dict[str, int] = {cat: 0 for cat in ALL_CATEGORIES}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            # DetectionEvents (from AI services) have a "service" key but no "type"
            if "service" in event and event_type is None:
                cat = _is_suspicious_event(event)
                if cat:
                    counts[cat] += 1

            # AlertEvents generated by the orchestrator (used for speech/gaze)
            elif event_type == "alert":
                cat = _is_alert_suspicious(event)
                if cat:
                    counts[cat] += 1

    return counts


# ─── Per-question anomaly counting ──────────────────────────────────────

def _empty_service_counts() -> Dict[str, int]:
    """Return a fresh zero-initialised dict for all 5 AI services."""
    return {key: 0 for key in SERVICE_KEYS}


def count_anomalies_by_question(
    events: List[Dict[str, Any]],
    total_number_of_questions: int,
    question_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Group suspicious anomaly counts from raw event objects by question.

    Args:
        events: A list of DetectionEvent dicts (already parsed from JSONL).
            Each event may optionally carry a ``questionId`` field (int or str)
            that identifies which question was active when the event occurred.
            Events without ``questionId`` are tallied under ``"unassigned"``.
        total_number_of_questions: The total number of questions in the exam.
        question_ids: Optional list of real question IDs from the LMS.

    Returns:
        A dict with the structure::

            {
                "total_questions": N,
                "questions": {
                    "question_1":  {"face_detection": 0, ...},
                    ...
                    "question_N":  {"face_detection": 0, ...},
                    # Only present when at least one event lacked a questionId:
                    "unassigned":  {"face_detection": 0, ...},
                }
            }
    """
    if total_number_of_questions < 1:
        raise ValueError(
            f"total_number_of_questions must be ≥ 1, got {total_number_of_questions}."
        )

    # Step 1 — Pre-fill every question with zeroes.
    if question_ids is not None and len(question_ids) > 0:
        questions: Dict[str, Dict[str, int]] = {
            f"question_{q}": _empty_service_counts()
            for q in question_ids
        }
    else:
        questions: Dict[str, Dict[str, int]] = {
            f"question_{q}": _empty_service_counts()
            for q in range(1, total_number_of_questions + 1)
        }

    # Separate bucket for events that carry no questionId.
    unassigned: Dict[str, int] = _empty_service_counts()
    has_unassigned = False

    # Step 2 — Iterate events and increment the appropriate counter.
    for event in events:
        event_type = event.get("type")
        service_key = None
        raw_qid = None

        if event_type is None and "service" in event:
            # DetectionEvent
            service_key = _suspicious_service_key(event)
            raw_qid = event.get("questionId")
        elif event_type == "alert":
            # AlertEvent
            code = event.get("code")
            if code == "GAZE_OFF_SCREEN":
                service_key = "eye_gaze"
            elif code in ["SPEECH_DETECTED", "SPEECH_CHEATING_FLAGGED"]:
                service_key = "speech_detection"
            
            raw_qid = event.get("questionId")
            if raw_qid is None:
                raw_qid = event.get("evidence", {}).get("questionId")

        if service_key is None:
            # Event is not suspicious or not a tracked alert — nothing to count.
            continue


        if raw_qid is None:
            # No question context attached — tally in the unassigned bucket.
            unassigned[service_key] += 1
            has_unassigned = True
        else:
            # Normalise: accept both int and string questionId values.
            try:
                q_num = int(raw_qid)
            except (TypeError, ValueError):
                unassigned[service_key] += 1
                has_unassigned = True
                continue

            q_key = f"question_{q_num}"
            if q_key in questions:
                questions[q_key][service_key] += 1
            else:
                # questionId is out of the declared range — still track it.
                questions.setdefault(q_key, _empty_service_counts())
                questions[q_key][service_key] += 1

    # Step 3 — Attach the unassigned bucket only if it received any counts.
    if has_unassigned:
        questions["unassigned"] = unassigned

    # Step 4 — Annotate each question with a convenience total.
    for q_key, service_counts in questions.items():
        service_counts["violation_total"] = sum(
            v for k, v in service_counts.items() if k != "violation_total"
        )

    return {
        "total_questions": total_number_of_questions,
        "questions": questions,
    }


def build_question_report(
    log_path: str,
    total_number_of_questions: int,
    student_id: str,
    exam_id: str,
    weights: Optional[Dict[str, float]] = None,
    question_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a JSONL session log and return a complete per-question JSON report.

    This is the primary implementation of the paper's risk formula:
        risk^sq = Σ_{t∈{f,h,c,b}} w_t × n^sq_t

    Each question receives:
      - Raw n^sq_t counts per AI service (not yet normalised)
      - violation_total: sum of all raw counts for that question
      - pre_normalisation_risk_score: weighted sum using provided weights

    The session_summary reports:
      - questions_violated / total_questions (violation rate)

    NOTE: Normalise raw counts across the full student cohort before
          comparing risk scores between students.

    Args:
        log_path:                  Absolute path to the session JSONL file.
        total_number_of_questions: Total questions in the exam.
        student_id:                Student identifier for report metadata.
        exam_id:                   Exam identifier for report metadata.
        weights:                   Per-category weights (default 1.0 each).

    Returns:
        A fully populated report dict ready for ``json.dump``.
    """
    if weights is None:
        weights = {
            CATEGORY_FACE: 1.0,
            CATEGORY_MOVE: 1.0,
            CATEGORY_CONV: 1.0,
            CATEGORY_OBJ:  1.0,
        }

    events: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    breakdown = count_anomalies_by_question(events, total_number_of_questions, question_ids)
    questions  = breakdown["questions"]

    # ─ Compute pre-normalisation risk score per question ────────────────────
    # Map service keys back to the 4 behaviour categories for the formula.
    _SERVICE_TO_CATEGORY = {
        "face_detection":   CATEGORY_FACE,
        "face_recognition": CATEGORY_FACE,
        "eye_gaze":         CATEGORY_MOVE,
        "speech_detection": CATEGORY_CONV,
        "object_detection": CATEGORY_OBJ,
    }

    # ─ Session-level summary ────────────────────────────────────────
    # Only count the declared questions (1..N), exclude 'unassigned'.
    if question_ids is not None and len(question_ids) > 0:
        declared_q_keys = [f"question_{q}" for q in question_ids]
    else:
        declared_q_keys = [f"question_{q}" for q in range(1, total_number_of_questions + 1)]

    questions_violated = sum(
        1 for q_key in declared_q_keys
        if questions.get(q_key, {}).get("violation_total", 0) > 0
    )
    questions_clean = total_number_of_questions - questions_violated
    violation_rate  = round(questions_violated / total_number_of_questions, 6) \
        if total_number_of_questions > 0 else 0.0

    questions_list = []
    for q_key, service_counts in questions.items():
        # Resolve the LMS question ID from the dict key
        if q_key.startswith("question_"):
            try:
                qid = int(q_key.split("_")[1])
            except ValueError:
                qid = q_key.split("_", 1)[1]
        else:
            qid = q_key
        # Build an ordered dict so question_id appears first (matches API schema)
        ordered = {"question_id": qid}
        ordered.update(service_counts)
        questions_list.append(ordered)

    breakdown["questions"] = questions_list

    return {
        "report_metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "student_id":   student_id,
            "Attempt_Id":   exam_id,
            "mode":         "per_question",
            "normalisation": "pending — apply min-max across exam cohort",
        },
        "session_summary": {
            "total_questions":    total_number_of_questions,
            "questions_violated": questions_violated,
            "questions_clean":    questions_clean,
            # fraction of questions where at least one violation occurred
            "violation_rate":     violation_rate,
            "weights_used":       weights,
        },
        # Extract only the questions list — NOT the whole breakdown dict.
        # Spreading **breakdown would also leak 'total_questions' as a
        # top-level key, which the LMS schema does not accept.
        "questions": breakdown["questions"],
    }


def compute_risk_score(
    counts: Dict[str, int],
    weights: Dict[str, float],
) -> float:
    """Paper formula: risk^sq = Σ_{t∈{f,h,c,b}} w_t × n^sq_t

    Args:
        counts:  Raw anomaly occurrence counts per category (n_f, n_h, n_c, n_b).
                 These are NOT yet normalised — normalisation to (0,1) must be
                 applied across the full student population BEFORE comparing
                 risk scores between students.
        weights: Proctor-customisable weight per category (default 1.0 each).

    Returns:
        Pre-normalisation weighted risk score for this session.
    """
    return sum(weights[cat] * counts[cat] for cat in ALL_CATEGORIES)


def build_report(
    log_path: str,
    student_id: str,
    exam_id: str,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Build the full risk-estimation report dict for a single student session.

    Implements step 3–4 of the paper's Overall Risk Estimation pipeline.
    The ``raw_counts`` in the output are the n^sq_t values that MUST be
    min-max normalised across the full exam cohort before risk scores can be
    meaningfully compared between students.
    """
    # n^sq_t  — raw occurrence counts for the 4 cheating categories
    raw_counts = count_anomalies(log_path)

    # risk^sq = Σ (w_t × n^sq_t)  — pre-normalisation weighted sum
    total_risk = compute_risk_score(raw_counts, weights)

    return {
        "report_metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "student_id": student_id,
            "Attempt_Id": exam_id,
            # Reminder: normalise raw_counts across all students before ranking.
            "normalisation": "pending — apply min-max across exam cohort",
        },
        # Raw n^sq_t counts: {f, h, c, b} — not yet normalised
        "raw_counts": raw_counts,
        "weights": weights,
    }


# ─── CLI ────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overall Risk Estimation — compute a raw-count weighted risk score from a session JSONL log.",
    )
    parser.add_argument(
        "log_file",
        help="Path to the session JSONL log file (e.g. sessions/1009.jsonl).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON report path. Defaults to <session_id>_risk_report.json next to the log.",
    )
    parser.add_argument("--student-id", default=None, help="Student ID to embed in the report.")
    parser.add_argument("--exam-id", default=None, help="Exam / Question ID to embed in the report.")

    # ── Whole-session weights (default 1.0 each) ─────────────────────────
    parser.add_argument("--wf", type=float, default=1.0, help="Weight for face absence/mismatch (default 1.0).")
    parser.add_argument("--wh", type=float, default=1.0, help="Weight for suspicious movement (default 1.0).")
    parser.add_argument("--wc", type=float, default=1.0, help="Weight for conversation/noise (default 1.0).")
    parser.add_argument("--wb", type=float, default=1.0, help="Weight for forbidden objects (default 1.0).")

    # ── Per-question mode ─────────────────────────────────────────────────
    # NOTE: per-question is the only mode — --by-question was removed because
    # main() always calls build_question_report(). The flag no longer exists.
    parser.add_argument(
        "--total-questions",
        type=int,
        default=None,
        metavar="N",
        help="Total number of questions in the exam. Required when --by-question is set.",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help="Comma-separated list of real question IDs from the LMS.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    log_path = os.path.abspath(args.log_file)
    if not os.path.isfile(log_path):
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Derive student / exam IDs from the filename when not specified
    session_id = os.path.splitext(os.path.basename(log_path))[0]
    student_id = args.student_id or session_id
    exam_id    = args.exam_id    or session_id

    # ── Per-question mode is the primary (default) mode ───────────────────
    if args.total_questions is None:
        print(
            "Error: --total-questions N is required.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.total_questions < 1:
        print(
            "Error: --total-questions must be a positive integer.",
            file=sys.stderr,
        )
        sys.exit(1)

    weights = {
        CATEGORY_FACE: args.wf,
        CATEGORY_MOVE: args.wh,
        CATEGORY_CONV: args.wc,
        CATEGORY_OBJ:  args.wb,
    }

    question_ids = args.question_ids.split(",") if args.question_ids else None

    report = build_question_report(
        log_path, args.total_questions, student_id, exam_id, weights, question_ids
    )

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        out_dir  = os.path.dirname(log_path)
        out_path = os.path.join(out_dir, f"{session_id}_question_report.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Question report written to: {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
