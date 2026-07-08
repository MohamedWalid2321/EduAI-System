import abc
import datetime
from typing import Dict, Any

class AIService(abc.ABC):
    """Base class for all AI services."""

    def __init__(self, service_name: str, session_id: str, config: Dict[str, Any]):
        self.service_name = service_name
        self.session_id = session_id
        self.config = config
        self.is_running = False

    @abc.abstractmethod
    async def start(self):
        """Start the AI service."""
        pass

    @abc.abstractmethod
    async def stop(self):
        """Stop the AI service."""
        pass

    @abc.abstractmethod
    async def predict(self, frame: str) -> Dict[str, Any]:
        """Process a frame and return a DetectionEvent."""
        pass

    def create_detection_event(self, confidence: float, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a normalized DetectionEvent dictionary."""
        return {
            "service": self.service_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "confidence": confidence,
            "sessionId": self.session_id,
            "payload": payload
        }
