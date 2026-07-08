# python_bridge/speech_detection.py
# This file is intentionally a thin wrapper.
# All real logic lives in services/speech_local.py

from services.speech_local import SpeechDetectionService

__all__ = ["SpeechDetectionService"]