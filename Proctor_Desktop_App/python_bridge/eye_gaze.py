from ai_base import AIService

class EyeGazeService(AIService):
    """Eye Gaze Detection service stub (runs locally)."""

    def __init__(self, session_id: str, config: dict):
        super().__init__("eye-gaze", session_id, config)

    async def start(self):
        self.is_running = True

    async def stop(self):
        self.is_running = False

    async def predict(self, frame: str) -> dict:
        """Process a frame for eye gaze (stub)."""
        return self.get_mock_event()

    def get_mock_event(self) -> dict:
        """Returns a mock gaze event."""
        return self.create_detection_event(0.95, {
            "gaze_x": 0.5,
            "gaze_y": 0.5,
            "status": "on-screen"
        })
