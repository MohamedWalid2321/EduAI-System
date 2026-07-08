import pytest
import unittest.mock as mock
import time
import asyncio
from services.eye_gaze_local import LocalEyeGazeService
from services.speech_local import LocalSpeechDetectionService

@pytest.fixture
def mock_config():
    return {
        "services": {
            "eye-gaze": {"fps": 10, "camera_index": 0, "input_mode": "camera"},
            "speech-detection": {"chunk_size": 1024, "sample_rate": 16000, "threshold": 0.05}
        }
    }

@pytest.mark.asyncio
async def test_eye_gaze_lifecycle(mock_config):
    emitter = mock.Mock()
    service = LocalEyeGazeService("session-1", mock_config, emitter)
    
    with mock.patch('cv2.VideoCapture') as mock_vc:
        mock_cap = mock_vc.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock.Mock())
        
        await service.start()
        assert service.is_running is True
        
        # Let it run for a bit
        time.sleep(0.2)
        
        await service.stop()
        assert service.is_running is False
        mock_cap.release.assert_called_once()

@pytest.mark.asyncio
async def test_speech_detection_lifecycle(mock_config):
    emitter = mock.Mock()
    service = LocalSpeechDetectionService("session-1", mock_config, emitter)
    
    with mock.patch('pyaudio.PyAudio') as mock_pa:
        mock_inst = mock_pa.return_value
        mock_stream = mock_inst.open.return_value
        mock_stream.read.return_value = b'\x00' * 4096 # Silence
        
        await service.start()
        assert service.is_running is True
        
        time.sleep(0.2)
        
        await service.stop()
        assert service.is_running is False
        mock_inst.terminate.assert_called_once()

@pytest.mark.asyncio
async def test_eye_gaze_hardware_failure(mock_config):
    emitter = mock.Mock()
    service = LocalEyeGazeService("session-1", mock_config, emitter)
    
    with mock.patch('cv2.VideoCapture') as mock_vc:
        mock_cap = mock_vc.return_value
        mock_cap.isOpened.return_value = False
        
        await service.start()
        assert service.is_running is False
        
        # Check that error was emitted
        emitter.assert_called_once()
        args = emitter.call_args[0][0]
        assert args["method"] == "serviceError"
        assert args["params"]["code"] == "HARDWARE_FAILURE"


@pytest.mark.asyncio
async def test_eye_gaze_shared_frame_mode_does_not_open_camera():
    emitter = mock.Mock()
    shared_mode_config = {
        "services": {
            "eye-gaze": {"fps": 10, "camera_index": 0, "input_mode": "shared_frame"}
        }
    }
    service = LocalEyeGazeService("session-1", shared_mode_config, emitter)

    with mock.patch('cv2.VideoCapture') as mock_vc:
        await service.start()
        assert service.is_running is True
        mock_vc.assert_not_called()

        await service.stop()
        assert service.is_running is False
