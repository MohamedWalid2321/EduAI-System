import asyncio
import json
import os

class MockOrchestrator:
    def set_session(self, session_id):
        pass
    async def _log_event(self, payload):
        print("LOG EVENT:", payload)

async def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.environ.get("LUMINA_CONFIG_PATH", os.path.join(base_dir, "config.json"))
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    from face_recognition import FaceRecognitionService
    service = FaceRecognitionService("test_session_001", config_dict)
    await service.start()
    
    print("Testing enroll...")
    result = await service.enroll(
        frame="WARMUP",
        profile_picture_url="https://moustafaalaa30--eduai-proctoring-proctoring-serve.modal.run/docs"
    )
    print("ENROLL RESULT:", result)

if __name__ == "__main__":
    asyncio.run(main())
