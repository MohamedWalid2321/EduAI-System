"""
Tests for risk_estimator.py

Covers:
  - Anomaly counting from raw DetectionEvents
  - Anomaly counting from AlertEvents
  - Min-max normalisation (including the all-zero / all-equal edge case)
  - Weighted-sum computation
  - Full report generation against a real session log
"""

import json
import os
import tempfile
import pytest

# Ensure the python_bridge package is importable
import sys
sys.path.insert(0, os.path.dirname(__file__))

from risk_estimator import (
    count_anomalies,
    normalise_min_max,
    compute_risk_score,
    build_report,
    count_anomalies_by_question,
    build_question_report,
    CATEGORY_FACE,
    CATEGORY_MOVE,
    CATEGORY_CONV,
    CATEGORY_OBJ,
    ALL_CATEGORIES,
    SERVICE_KEYS,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_jsonl(events, tmpdir):
    """Write a list of dicts as a JSONL file and return the path."""
    path = os.path.join(tmpdir, "test_session.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── count_anomalies ─────────────────────────────────────────────────────

class TestCountAnomalies:
    def test_empty_log(self, tmpdir):
        path = _write_jsonl([], tmpdir)
        counts = count_anomalies(path)
        assert counts == {cat: 0 for cat in ALL_CATEGORIES}

    def test_eye_gaze_away_counted(self, tmpdir):
        events = [
            {"service": "eye-gaze", "timestamp": "t1", "confidence": 0.9,
             "sessionId": "s", "payload": {"status": "away"}},
            {"service": "eye-gaze", "timestamp": "t2", "confidence": 0.9,
             "sessionId": "s", "payload": {"status": "on-screen"}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_MOVE] == 1
        assert counts[CATEGORY_FACE] == 0

    def test_speech_violation_counted(self, tmpdir):
        events = [
            {"service": "speech-detection", "timestamp": "t1", "confidence": 1.0,
             "sessionId": "s", "payload": {
                 "new_violations": [{"type": "speech_violation", "duration_seconds": 5.0,
                                     "strike_number": 1, "cheater_flagged": False}],
                 "violation_count": 1, "total_strikes": 1, "is_cheater": False}},
            # No new violations → should NOT be counted
            {"service": "speech-detection", "timestamp": "t2", "confidence": 1.0,
             "sessionId": "s", "payload": {
                 "new_violations": [], "violation_count": 0,
                 "total_strikes": 1, "is_cheater": False}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_CONV] == 1

    def test_face_detection_missing(self, tmpdir):
        events = [
            {"service": "face-detection", "timestamp": "t1", "confidence": 0.85,
             "sessionId": "s", "payload": {"face_detected": False}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_FACE] == 1

    def test_face_recognition_mismatch(self, tmpdir):
        events = [
            {"service": "face-recognition", "timestamp": "t1", "confidence": 0.3,
             "sessionId": "s", "payload": {"is_matched": False, "recognition_ran": True}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_FACE] == 1

    def test_face_recognition_spoof(self, tmpdir):
        events = [
            {"service": "face-recognition", "timestamp": "t1", "confidence": 0.9,
             "sessionId": "s", "payload": {"is_spoof": True}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_FACE] == 1

    def test_object_detection_suspicious(self, tmpdir):
        events = [
            {"service": "object-detection", "timestamp": "t1", "confidence": 0.8,
             "sessionId": "s", "payload": {"suspicious": True, "objects": ["cell phone"]}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_OBJ] == 1

    def test_alert_events_counted(self, tmpdir):
        events = [
            {"type": "alert", "code": "NO_FACE_DETECTED", "severity": "high",
             "message": "Face missing", "timestamp": "t1", "sessionId": "s",
             "evidence": {"service": "face-detection", "confidence": 0.0}},
            {"type": "alert", "code": "SPEECH_DETECTED", "severity": "medium",
             "message": "Speech", "timestamp": "t2", "sessionId": "s",
             "evidence": {"service": "speech-detection", "confidence": 1.0}},
            {"type": "alert", "code": "UNAUTHORIZED_OBJECT", "severity": "high",
             "message": "Object", "timestamp": "t3", "sessionId": "s",
             "evidence": {"service": "object-detection", "confidence": 0.85}},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts[CATEGORY_FACE] == 1
        assert counts[CATEGORY_CONV] == 1
        assert counts[CATEGORY_OBJ] == 1

    def test_risk_score_events_ignored(self, tmpdir):
        events = [
            {"type": "riskScore", "score": 3, "trend": "falling", "timestamp": "t1"},
        ]
        path = _write_jsonl(events, tmpdir)
        counts = count_anomalies(path)
        assert counts == {cat: 0 for cat in ALL_CATEGORIES}


# ── normalise_min_max ───────────────────────────────────────────────────

class TestNormalise:
    def test_all_zero(self):
        counts = {cat: 0 for cat in ALL_CATEGORIES}
        norm = normalise_min_max(counts)
        assert all(v == 0.0 for v in norm.values())

    def test_all_equal_nonzero(self):
        counts = {cat: 5 for cat in ALL_CATEGORIES}
        norm = normalise_min_max(counts)
        assert all(v == 0.0 for v in norm.values())

    def test_simple_spread(self):
        counts = {
            CATEGORY_FACE: 0,
            CATEGORY_MOVE: 10,
            CATEGORY_CONV: 5,
            CATEGORY_OBJ: 10,
        }
        norm = normalise_min_max(counts)
        assert norm[CATEGORY_FACE] == 0.0
        assert norm[CATEGORY_MOVE] == 1.0
        assert norm[CATEGORY_CONV] == 0.5
        assert norm[CATEGORY_OBJ] == 1.0


# ── compute_risk_score ──────────────────────────────────────────────────

class TestRiskScore:
    def test_zero_normalised(self):
        normalised = {cat: 0.0 for cat in ALL_CATEGORIES}
        weights = {cat: 1.0 for cat in ALL_CATEGORIES}
        assert compute_risk_score(normalised, weights) == 0.0

    def test_uniform_weights(self):
        normalised = {
            CATEGORY_FACE: 1.0,
            CATEGORY_MOVE: 0.5,
            CATEGORY_CONV: 0.5,
            CATEGORY_OBJ: 0.0,
        }
        weights = {cat: 1.0 for cat in ALL_CATEGORIES}
        # 1.0 + 0.5 + 0.5 + 0.0 = 2.0
        assert compute_risk_score(normalised, weights) == 2.0

    def test_custom_weights(self):
        normalised = {
            CATEGORY_FACE: 1.0,
            CATEGORY_MOVE: 0.5,
            CATEGORY_CONV: 0.0,
            CATEGORY_OBJ: 0.5,
        }
        weights = {
            CATEGORY_FACE: 2.0,
            CATEGORY_MOVE: 1.0,
            CATEGORY_CONV: 1.0,
            CATEGORY_OBJ: 3.0,
        }
        # (2.0*1.0) + (1.0*0.5) + (1.0*0.0) + (3.0*0.5) = 2 + 0.5 + 0 + 1.5 = 4.0
        assert compute_risk_score(normalised, weights) == 4.0


# ── build_report (integration) ──────────────────────────────────────────

class TestBuildReport:
    def test_full_report_structure(self, tmpdir):
        events = [
            {"service": "eye-gaze", "timestamp": "t", "confidence": 0.9,
             "sessionId": "s", "payload": {"status": "away"}},
            {"service": "eye-gaze", "timestamp": "t", "confidence": 0.9,
             "sessionId": "s", "payload": {"status": "away"}},
            {"service": "speech-detection", "timestamp": "t", "confidence": 1.0,
             "sessionId": "s", "payload": {
                 "new_violations": [{"duration_seconds": 3.0, "strike_number": 1}],
                 "violation_count": 1, "total_strikes": 1, "is_cheater": False}},
        ]
        path = _write_jsonl(events, tmpdir)
        weights = {cat: 1.0 for cat in ALL_CATEGORIES}
        report = build_report(path, "student-42", "exam-101", weights)

        # Structure checks
        assert "report_metadata" in report
        assert report["report_metadata"]["student_id"] == "student-42"
        assert report["report_metadata"]["exam_id"] == "exam-101"
        assert "raw_counts" in report
        assert "normalised_counts" in report
        assert "weights" in report
        assert "total_risk_score" in report

        # Values check (raw: face=0, move=2, conv=1, obj=0 → norm: 0, 1, 0.5, 0)
        assert report["raw_counts"][CATEGORY_MOVE] == 2
        assert report["raw_counts"][CATEGORY_CONV] == 1
        assert report["normalised_counts"][CATEGORY_MOVE] == 1.0
        assert report["normalised_counts"][CATEGORY_CONV] == 0.5
        assert report["total_risk_score"] == 1.5


# ── Test against a real session log (smoke test) ────────────────────────

class TestRealSessionLog:
    """Runs against an actual session file if available (non-destructive)."""

    REAL_LOG = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sessions", "1009.jsonl"
    )

    @pytest.mark.skipif(
        not os.path.isfile(REAL_LOG),
        reason="Real session log not found — skipping smoke test.",
    )
    def test_real_log_produces_valid_report(self, tmpdir):
        weights = {cat: 1.0 for cat in ALL_CATEGORIES}
        report = build_report(self.REAL_LOG, "student-1009", "exam-1009", weights)

        assert report["report_metadata"]["student_id"] == "student-1009"
        assert isinstance(report["total_risk_score"], float)
        assert report["total_risk_score"] >= 0.0

        # Write to tmp to verify JSON serialisation round-trips
        out = os.path.join(tmpdir, "real_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded == report


# ── count_anomalies_by_question ─────────────────────────────────────────

class TestCountAnomaliesByQuestion:
    """Unit tests for the per-question grouping function."""

    # Helper: build a minimal suspicious DetectionEvent.
    @staticmethod
    def _gaze_away(question_id=None):
        ev = {
            "service": "eye-gaze", "timestamp": "t",
            "confidence": 0.9, "sessionId": "s",
            "payload": {"status": "away"},
        }
        if question_id is not None:
            ev["questionId"] = question_id
        return ev

    @staticmethod
    def _speech_violation(question_id=None):
        ev = {
            "service": "speech-detection", "timestamp": "t",
            "confidence": 1.0, "sessionId": "s",
            "payload": {
                "new_violations": [{"duration_seconds": 3.0, "strike_number": 1}],
                "violation_count": 1, "total_strikes": 1, "is_cheater": False,
            },
        }
        if question_id is not None:
            ev["questionId"] = question_id
        return ev

    @staticmethod
    def _object_detected(question_id=None):
        ev = {
            "service": "object-detection", "timestamp": "t",
            "confidence": 0.8, "sessionId": "s",
            "payload": {"suspicious": True, "objects": ["phone"]},
        }
        if question_id is not None:
            ev["questionId"] = question_id
        return ev

    # ── pre-fill guarantee ──────────────────────────────────────────────

    def test_all_questions_present_even_with_no_events(self):
        result = count_anomalies_by_question([], total_number_of_questions=5)
        assert result["total_questions"] == 5
        questions = result["questions"]
        for q in range(1, 6):
            key = f"question_{q}"
            assert key in questions, f"{key} missing"
            assert questions[key] == {k: 0 for k in SERVICE_KEYS}
        # No unassigned bucket when there are no events.
        assert "unassigned" not in questions

    def test_zero_anomaly_question_still_shows_all_services(self):
        # Only question 1 gets an anomaly; question 2 must still appear with zeroes.
        events = [self._gaze_away(question_id=1)]
        result = count_anomalies_by_question(events, total_number_of_questions=2)
        q2 = result["questions"]["question_2"]
        assert q2 == {k: 0 for k in SERVICE_KEYS}

    def test_total_questions_metadata(self):
        result = count_anomalies_by_question([], total_number_of_questions=10)
        assert result["total_questions"] == 10

    def test_invalid_total_raises(self):
        import pytest
        with pytest.raises(ValueError):
            count_anomalies_by_question([], total_number_of_questions=0)

    # ── per-service routing ─────────────────────────────────────────────

    def test_eye_gaze_increments_correct_key(self):
        events = [self._gaze_away(question_id=1)]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"]["eye_gaze"] == 1
        assert result["questions"]["question_1"]["face_detection"] == 0

    def test_speech_increments_correct_key(self):
        events = [self._speech_violation(question_id=2)]
        result = count_anomalies_by_question(events, total_number_of_questions=3)
        assert result["questions"]["question_2"]["speech_detection"] == 1
        assert result["questions"]["question_1"]["speech_detection"] == 0

    def test_object_increments_correct_key(self):
        events = [self._object_detected(question_id=3)]
        result = count_anomalies_by_question(events, total_number_of_questions=3)
        assert result["questions"]["question_3"]["object_detection"] == 1

    def test_face_detection_missing_increments_correct_key(self):
        events = [{
            "service": "face-detection", "timestamp": "t", "confidence": 0.9,
            "sessionId": "s", "payload": {"face_detected": False}, "questionId": 1,
        }]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"]["face_detection"] == 1

    def test_face_recognition_mismatch_increments_correct_key(self):
        events = [{
            "service": "face-recognition", "timestamp": "t", "confidence": 0.3,
            "sessionId": "s",
            "payload": {"is_matched": False, "recognition_ran": True},
            "questionId": 2,
        }]
        result = count_anomalies_by_question(events, total_number_of_questions=2)
        assert result["questions"]["question_2"]["face_recognition"] == 1

    def test_multiple_anomalies_same_question_accumulate(self):
        events = [
            self._gaze_away(question_id=1),
            self._gaze_away(question_id=1),
            self._gaze_away(question_id=1),
        ]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"]["eye_gaze"] == 3

    def test_multiple_services_same_question(self):
        events = [
            self._gaze_away(question_id=2),
            self._speech_violation(question_id=2),
            self._object_detected(question_id=2),
        ]
        result = count_anomalies_by_question(events, total_number_of_questions=3)
        q2 = result["questions"]["question_2"]
        assert q2["eye_gaze"] == 1
        assert q2["speech_detection"] == 1
        assert q2["object_detection"] == 1

    # ── unassigned bucket ───────────────────────────────────────────────

    def test_events_without_question_id_go_to_unassigned(self):
        events = [self._gaze_away()]  # no questionId
        result = count_anomalies_by_question(events, total_number_of_questions=2)
        assert "unassigned" in result["questions"]
        assert result["questions"]["unassigned"]["eye_gaze"] == 1

    def test_no_unassigned_key_when_all_events_have_question_id(self):
        events = [self._gaze_away(question_id=1)]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert "unassigned" not in result["questions"]

    def test_invalid_string_question_id_goes_to_unassigned(self):
        ev = self._gaze_away()
        ev["questionId"] = "notanumber"
        result = count_anomalies_by_question([ev], total_number_of_questions=2)
        assert result["questions"]["unassigned"]["eye_gaze"] == 1

    # ── out-of-range questionId ─────────────────────────────────────────

    def test_out_of_range_question_id_added_to_output(self):
        # questionId=99 while total_questions=3 — should still appear.
        events = [self._gaze_away(question_id=99)]
        result = count_anomalies_by_question(events, total_number_of_questions=3)
        assert "question_99" in result["questions"]
        assert result["questions"]["question_99"]["eye_gaze"] == 1

    # ── non-suspicious events skipped ──────────────────────────────────

    def test_on_screen_gaze_not_counted(self):
        events = [{
            "service": "eye-gaze", "timestamp": "t", "confidence": 0.9,
            "sessionId": "s", "payload": {"status": "on-screen"}, "questionId": 1,
        }]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"]["eye_gaze"] == 0

    def test_alert_events_ignored(self):
        events = [{
            "type": "alert", "code": "GAZE_OFF_SCREEN", "severity": "low",
            "message": "gaze", "timestamp": "t", "sessionId": "s",
            "evidence": {"service": "eye-gaze", "confidence": 0.9},
        }]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"] == {k: 0 for k in SERVICE_KEYS}

    def test_risk_score_events_ignored(self):
        events = [{"type": "riskScore", "score": 10, "trend": "rising", "timestamp": "t"}]
        result = count_anomalies_by_question(events, total_number_of_questions=1)
        assert result["questions"]["question_1"] == {k: 0 for k in SERVICE_KEYS}

    # ── integer vs string questionId coercion ───────────────────────────

    def test_string_digit_question_id_coerced(self):
        ev = self._gaze_away()
        ev["questionId"] = "3"   # string, but a valid number
        result = count_anomalies_by_question([ev], total_number_of_questions=3)
        assert result["questions"]["question_3"]["eye_gaze"] == 1


# ── build_question_report (integration, real log) ───────────────────────

class TestBuildQuestionReport:
    REAL_LOG = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sessions", "1009.jsonl"
    )

    @pytest.mark.skipif(
        not os.path.isfile(REAL_LOG),
        reason="Real session log 1009.jsonl not found.",
    )
    def test_real_log_structure(self, tmpdir):
        report = build_question_report(self.REAL_LOG, 5, "student-1009", "exam-1009")

        assert report["total_questions"] == 5
        assert report["report_metadata"]["mode"] == "per_question"

        questions = report["questions"]
        for q in range(1, 6):
            qk = f"question_{q}"
            assert qk in questions
            for svc in SERVICE_KEYS:
                assert svc in questions[qk]

        # Round-trip JSON check
        out = os.path.join(tmpdir, "q_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded == report
