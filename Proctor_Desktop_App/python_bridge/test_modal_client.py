import pytest
import httpx
import json
from modal_client import ModalClient

@pytest.mark.asyncio
async def test_modal_client_success(respx_mock):
    endpoint = "https://test.modal.run/v1"
    token = "test-token"
    client = ModalClient(endpoint, token)
    
    # Mock success response
    respx_mock.post(endpoint).mock(return_value=httpx.Response(200, json={
        "service": "test-service",
        "timestamp": "2026-04-19T00:00:00Z",
        "confidence": 0.95,
        "sessionId": "test-session",
        "payload": {"status": "ok"}
    }))
    
    result = await client.predict("test-service", "test-session", "base64data")
    
    assert result["service"] == "test-service"
    assert result["confidence"] == 0.95
    assert result["payload"]["status"] == "ok"

@pytest.mark.asyncio
async def test_modal_client_cold_start(respx_mock):
    endpoint = "https://test.modal.run/v1"
    client = ModalClient(endpoint, "token")
    
    # Mock 503 response
    respx_mock.post(endpoint).mock(return_value=httpx.Response(503))
    
    result = await client.predict("test-service", "session-1", "data")
    
    assert result["payload"]["status"] == "error"
    assert result["payload"]["code"] == "SERVICE_UNAVAILABLE"

@pytest.mark.asyncio
async def test_modal_client_timeout(respx_mock):
    endpoint = "https://test.modal.run/v1"
    client = ModalClient(endpoint, "token", timeout=0.1)
    
    # Mock timeout
    respx_mock.post(endpoint).mock(side_effect=httpx.TimeoutException("Too slow"))
    
    result = await client.predict("test-service", "session-1", "data")
    
    assert result["payload"]["status"] == "error"
    assert result["payload"]["code"] == "TIMEOUT"
