import pytest
import asyncio
import json
import os
import time
import datetime
from orchestrator import ProctoringOrchestrator

@pytest.fixture
def orchestrator_config():
    return {
        "orchestration": {
            "rules": {
                "eye-gaze": {"away_threshold_seconds": 0.1, "weight": 20},
                "face-recognition": {"missing_threshold_seconds": 0.1, "weight": 50},
                "speech-detection": {"weight": 30},
                "object-detection": {"weight": 60}
            },
            "risk_score_decay": 0.9
        }
    }

@pytest.fixture
def mock_emit():
    emitted = []
    def emit(event):
        emitted.append(event)
    return emit, emitted

@pytest.mark.asyncio
async def test_immediate_alert_object(orchestrator_config, mock_emit):
    emit_func, emitted_list = mock_emit
    orch = ProctoringOrchestrator(orchestrator_config, emit_func)
    orch.set_session("test-session-obj")
    
    event = {
        "service": "object-detection",
        "sessionId": "test-session-obj",
        "confidence": 0.9,
        "payload": {"suspicious": True, "objects": ["cell phone"]}
    }
    
    await orch.on_detection_event(event)
    
    # Should see: 1 score update, 1 alert
    alerts = [e for e in emitted_list if e.get("type") == "alert"]
    assert len(alerts) == 1
    assert alerts[0]["code"] == "UNAUTHORIZED_OBJECT"
    assert "cell phone" in alerts[0]["message"]

@pytest.mark.asyncio
async def test_time_based_alert_face(orchestrator_config, mock_emit):
    emit_func, emitted_list = mock_emit
    orch = ProctoringOrchestrator(orchestrator_config, emit_func)
    orch.set_session("test-session-face")
    
    # 1st event: face missing
    event_missing = {
        "service": "face-recognition",
        "sessionId": "test-session-face",
        "confidence": 1.0,
        "payload": {"is_matched": False}
    }
    await orch.on_detection_event(event_missing)
    
    # No alert yet (threshold is 0.1s, but we just started)
    assert not any(e.get("type") == "alert" for e in emitted_list)
    
    # Wait a bit
    await asyncio.sleep(0.2)
    
    # 2nd event: face still missing
    await orch.on_detection_event(event_missing)
    
    # Should trigger alert
    alerts = [e for e in emitted_list if e.get("type") == "alert"]
    assert len(alerts) == 1
    assert alerts[0]["code"] == "NO_FACE_DETECTED"

@pytest.mark.asyncio
async def test_risk_score_calculation(orchestrator_config, mock_emit):
    emit_func, emitted_list = mock_emit
    orch = ProctoringOrchestrator(orchestrator_config, emit_func)
    orch.set_session("test-session-score")
    
    # Speech detected (weight 30)
    event = {
        "service": "speech-detection",
        "sessionId": "test-session-score",
        "confidence": 0.8,
        "payload": {"is_speech_detected": True}
    }
    await orch.on_detection_event(event)
    
    # Initial score should be (30 * 0.1) = 3
    # Look for the last riskScore update
    scores = [e["score"] for e in emitted_list if e.get("type") == "riskScore"]
    assert scores[-1] == 3
    
    # Another speech event
    await orch.on_detection_event(event)
    assert emitted_list[-1]["score"] == 6

@pytest.mark.asyncio
async def test_risk_score_decay(orchestrator_config, mock_emit):
    emit_func, emitted_list = mock_emit
    orch = ProctoringOrchestrator(orchestrator_config, emit_func)
    orch.set_session("test-session-decay")
    
    # Raise score first
    orch.risk_score = 50.0
    
    # Send "safe" event
    event_safe = {
        "service": "eye-gaze",
        "payload": {"status": "on-screen"}
    }
    await orch.on_detection_event(event_safe)
    
    # Score should decay: 50 * 0.9 = 45
    assert emitted_list[-1]["score"] == 45

@pytest.mark.asyncio
async def test_session_logging(orchestrator_config, mock_emit):
    emit_func, _ = mock_emit
    session_id = "test-session-log"
    orch = ProctoringOrchestrator(orchestrator_config, emit_func)
    orch.set_session(session_id)
    
    log_path = os.path.join("sessions", f"{session_id}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
        
    event = {
        "service": "eye-gaze",
        "sessionId": session_id,
        "payload": {"status": "away"}
    }
    await orch.on_detection_event(event)
    
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 2 # 1 for detection, 1 for score update
        data = json.loads(lines[0])
        assert data["service"] == "eye-gaze"
