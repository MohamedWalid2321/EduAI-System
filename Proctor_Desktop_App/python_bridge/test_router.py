import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
import asyncio
from router import AIRouter

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.base_url = "https://api.lumina.com"
    config.python_port = 5050
    config.services = {
        "eye-gaze": {"threshold": 0.5},
        "face-recognition": {"endpoint_url": "https://test.modal.run/face"}
    }
    config.modal = {"token_id": "test"}
    return config

@pytest.fixture
def router(tmp_path):
    schema_content = {
        "type": "object",
        "required": ["service", "timestamp", "confidence", "sessionId", "payload"],
        "properties": {
            "service": {"type": "string"},
            "timestamp": {"type": "string"},
            "confidence": {"type": "number"},
            "sessionId": {"type": "string"},
            "payload": {"type": "object"}
        }
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema_content))
    
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    
    return AIRouter(str(config_path), str(schema_path))

@pytest.mark.asyncio
@patch('router.load_config')
async def test_handle_start_service(mock_load, router, mock_config):
    mock_load.return_value = mock_config
    router.load_configuration()
    router.write_stdout = MagicMock()
    
    # Mock the start method since it might do network calls (pre-warm)
    with patch('face_recognition.FaceRecognitionService.start', new_callable=AsyncMock):
        params = {"service": "face-recognition", "sessionId": "test-session"}
        await router.handle_start_service(1, params)
    
    assert "face-recognition" in router.services
    router.write_stdout.assert_called()
    response = router.write_stdout.call_args[0][0]
    assert response["result"]["status"] == "started"

@pytest.mark.asyncio
@patch('router.load_config')
async def test_handle_predict_modal(mock_load, router, mock_config):
    mock_load.return_value = mock_config
    router.load_configuration()
    router.load_schema()
    router.write_stdout = MagicMock()
    
    # Start service with mocked start (to avoid real pre-warm)
    with patch('face_recognition.FaceRecognitionService.start', new_callable=AsyncMock):
        await router.handle_start_service(1, {"service": "face-recognition"})
    
    service = router.services["face-recognition"]
    # Mock the predict method of the service to return a valid event
    service.predict = AsyncMock(return_value={
        "service": "face-recognition",
        "timestamp": "2026-04-19T00:00:00Z",
        "confidence": 0.99,
        "sessionId": "default-session",
        "payload": {"is_matched": True}
    })
    
    params = {"service": "face-recognition", "frame": "base64data"}
    await router.handle_predict(2, params)
    
    # Check stdout for the detection notification
    calls = [call[0][0] for call in router.write_stdout.call_args_list]
    detection_notif = next(c for c in calls if c.get("method") == "detection")
    assert detection_notif["params"]["service"] == "face-recognition"
    assert detection_notif["params"]["payload"]["is_matched"] is True

@pytest.mark.asyncio
@patch('router.load_config')
async def test_handle_mock_detection(mock_load, router, mock_config):
    mock_load.return_value = mock_config
    router.load_configuration()
    router.load_schema()
    router.write_stdout = MagicMock()
    
    await router.handle_start_service(1, {"service": "eye-gaze"})
    await router.handle_mock_detection(2, {"service": "eye-gaze"})
    
    calls = [call[0][0] for call in router.write_stdout.call_args_list]
    detection_notif = next(c for c in calls if c.get("method") == "detection")
    assert detection_notif["params"]["service"] == "eye-gaze"
