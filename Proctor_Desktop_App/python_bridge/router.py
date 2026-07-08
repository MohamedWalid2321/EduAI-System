import json
import sys
import os
import asyncio
from typing import Dict, Any, Optional

try:
    from jsonschema import validate, ValidationError
except ImportError:
    class ValidationError(Exception):
        """Fallback error type when jsonschema is unavailable."""
        pass

    def validate(instance, schema):
        """
        No-op validator fallback.
        Keeps router operational in environments missing jsonschema.
        """
        return None

from config import load_config, ConfigError

# Phase 8: Proctoring Orchestration
from orchestrator import ProctoringOrchestrator

# ---------------------------------------------------------------------------
# Lazy service imports — guarded so the router stays operational even when
# optional AI packages (cv2, torch, mediapipe, sounddevice …) are absent.
# ---------------------------------------------------------------------------
_service_import_errors: Dict[str, str] = {}

try:
    from services.eye_gaze_local import LocalEyeGazeService as _LocalEyeGazeService
except Exception as _e:
    _LocalEyeGazeService = None  # type: ignore
    _service_import_errors["eye-gaze"] = str(_e)

try:
    from services.speech_local import SpeechDetectionService as _SpeechDetectionService
except Exception as _e:
    _SpeechDetectionService = None  # type: ignore
    _service_import_errors["speech-detection"] = str(_e)

try:
    from face_recognition import FaceRecognitionService as _FaceRecognitionService
except Exception as _e:
    _FaceRecognitionService = None  # type: ignore
    _service_import_errors["face-recognition"] = str(_e)

try:
    from face_detection import FaceDetectionService as _FaceDetectionService
except Exception as _e:
    _FaceDetectionService = None  # type: ignore
    _service_import_errors["face-detection"] = str(_e)

try:
    from object_detection import ObjectDetectionService as _ObjectDetectionService
except Exception as _e:
    _ObjectDetectionService = None  # type: ignore
    _service_import_errors["object-detection"] = str(_e)

try:
    from services.clip_upload_service import ClipUploadService as _ClipUploadService
except Exception as _e:
    _ClipUploadService = None  # type: ignore
    _service_import_errors["clip-upload"] = str(_e)

class AIRouter:
    def __init__(self, config_path: str, schema_path: str):
        self.config_path = config_path
        self.schema_path = schema_path
        self.config = None
        self.schema = None
        self.orchestrator = None
        self.services = {}
        # Only register services whose packages were successfully imported.
        self.service_classes = {}
        if _LocalEyeGazeService is not None:
            self.service_classes["eye-gaze"] = _LocalEyeGazeService
        if _SpeechDetectionService is not None:
            self.service_classes["speech-detection"] = _SpeechDetectionService
        if _FaceRecognitionService is not None:
            self.service_classes["face-recognition"] = _FaceRecognitionService
        if _FaceDetectionService is not None:
            self.service_classes["face-detection"] = _FaceDetectionService
        if _ObjectDetectionService is not None:
            self.service_classes["object-detection"] = _ObjectDetectionService
        self.loop = None

    def load_configuration(self):
        try:
            self.config = load_config(self.config_path)
        except ConfigError as e:
            self.send_error(None, -32000, f"Config error: {e.message}")
            sys.exit(1)

    def load_schema(self):
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        except Exception as e:
            self.send_error(None, -32000, f"Schema load error: {str(e)}")
            sys.exit(1)

    async def run(self):
        self.load_configuration()
        self.load_schema()
        self.loop = asyncio.get_running_loop()
        
        # Initialize Orchestrator
        self.orchestrator = ProctoringOrchestrator(
            vars(self.config), 
            self.emit_notification_from_orchestrator
        )

        while True:
            line = await self.loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            try:
                request = json.loads(line)
                asyncio.create_task(self.handle_request(request))
            except json.JSONDecodeError:
                self.send_error(None, -32700, "Parse error")
            except Exception as e:
                self.send_error(None, -32603, f"Internal error: {str(e)}")

    def emit_notification_from_orchestrator(self, notification: Dict[str, Any]):
        """Callback for orchestrator to send alerts and risk scores to stdout."""
        # Risk scores and alerts are already formatted as parameters
        msg_type = notification.get("type", "notification")
        self.send_notification(msg_type, notification)

    def thread_safe_emit(self, event: Dict[str, Any]):
        """
        Callback passed to local services. 
        Background threads call this to emit events to the async loop.
        """
        if "method" in event and event["method"] == "serviceError":
            # Direct notification (error)
            self.loop.call_soon_threadsafe(
                self.send_notification, event["method"], event["params"]
            )
        else:
            # Detection event - pipe through orchestrator
            def process_and_send():
                try:
                    validate(instance=event, schema=self.schema)
                    # Router sends raw detection immediately
                    self.send_notification("detection", event)
                    # AND sends to orchestrator for fusion/logging
                    asyncio.create_task(self.orchestrator.on_detection_event(event))
                except ValidationError as e:
                    self.send_notification("serviceError", {
                        "service": event.get("service", "unknown"),
                        "code": "CONTRACT_VIOLATION",
                        "message": f"Background event failed contract validation: {e.message}"
                    })
            
            self.loop.call_soon_threadsafe(process_and_send)

    async def handle_request(self, request: Dict[str, Any]):
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "startService":
            await self.handle_start_service(request_id, params)
        elif method == "stopService":
            await self.handle_stop_service(request_id, params)
        elif method == "queryStatus":
            self.handle_query_status(request_id, params)
        elif method == "predict":
            await self.handle_predict(request_id, params)
        elif method == "recalibrateGaze":
            self.handle_recalibrate_gaze(request_id)
        elif method == "mockDetection": # Helper for testing/wiring
            await self.handle_mock_detection(request_id, params)
        elif method == "enrollReference":
            await self.handle_enroll_reference(request_id, params)
        elif method == "unenrollReference":
            await self.handle_unenroll_reference(request_id, params)
        elif method == "upload_clip":
            await self.handle_upload_clip(request_id, params)
        else:
            self.send_error(request_id, -32601, "Method not found")

    def handle_recalibrate_gaze(self, request_id):
        """Reset eye-gaze calibration so the next frames re-run it from scratch."""
        service = self.services.get("eye-gaze")
        if service is None:
            self.send_error(request_id, -32602, "Eye-gaze service not running")
            return
        if hasattr(service, "recalibrate"):
            service.recalibrate()
        self.send_result(request_id, {"ok": True})

    async def handle_predict(self, request_id: Any, params: Dict[str, Any]):
        service_name = params.get("service")
        frame = params.get("frame")
        
        if service_name not in self.services:
            self.send_error(request_id, -32602, f"Service not running: {service_name}")
            return
            
        if not frame:
            self.send_error(request_id, -32602, "Missing frame data")
            return

        try:
            service = self.services[service_name]

            # Forward isAllowableToLookDown to the eye-gaze service so the
            # GazeSession switches between writing/normal mode per question.
            # DB column: IsAllowableToLookDown — handle both JSON casing variants.
            if service_name == "eye-gaze" and hasattr(service, "set_question_mode"):
                is_allowed = (
                    params.get("isAllowableToLookDown")
                    or params.get("IsAllowableToLookDown")
                    or False
                )
                service.set_question_mode(bool(is_allowed))

            # All services should now implement an async predict method
            event = await service.predict(frame)

            # Stamp the event with the current questionId so the post-exam
            # risk estimator can attribute violations to the right question.
            question_id = params.get("questionId")
            if question_id is not None and "questionId" not in event:
                event["questionId"] = question_id
            
            # Validate event against schema
            try:
                validate(instance=event, schema=self.schema)
                self.send_notification("detection", event)
                self.send_result(request_id, {"status": "success"})
                # Pipe to orchestrator
                await self.orchestrator.on_detection_event(event)
            except ValidationError as e:
                self.send_notification("serviceError", {
                    "service": service_name,
                    "code": "CONTRACT_VIOLATION",
                    "message": f"Detection event failed contract validation: {e.message}"
                })
                self.send_error(request_id, -32001, "Contract violation")
        except Exception as e:
            self.send_error(request_id, -32603, f"Internal error during prediction: {str(e)}")


    async def handle_enroll_reference(self, request_id: Any, params: Dict[str, Any]):
        """Enroll a reference frame for face recognition.

        Expected params:
          frame             : str  — base64 data-URL of the live webcam capture
          sessionId         : str  — exam/attempt session identifier
          profilePictureUrl : str | None — official CDN URL from the login response;
                              when provided, identity is confirmed via /analysis/face-frame
                              before the embedding is stored.
        """
        frame = params.get("frame")
        session_id = params.get("sessionId", "default-session")
        profile_picture_url = params.get("profilePictureUrl") or None

        if not frame:
            self.send_error(request_id, -32602, "Missing frame data")
            return

        service = self.services.get("face-recognition")
        if service is None:
            self.send_result(request_id, {
                "ok": False,
                "error": {"code": "SERVICE_NOT_RUNNING",
                           "message": "Face recognition service is not running."}
            })
            return

        try:
            result = await service.enroll(frame, profile_picture_url=profile_picture_url)
            self.send_result(request_id, result)
            # Log the lifecycle event so post-exam forensics can confirm enrollment.
            if self.orchestrator:
                self.orchestrator.set_session(session_id)
                
                log_payload = {
                    "type": "lifecycle",
                    "event": "enrollment",
                    "ok": result.get("ok", False),
                    "sessionId": session_id,
                    "timestamp": __import__('datetime').datetime.now(
                        __import__('datetime').timezone.utc).isoformat(),
                }
                
                if "probability" in result:
                    log_payload["probability"] = result["probability"]
                if "evidence" in result:
                    log_payload["evidence"] = result["evidence"]
                if "error" in result:
                    log_payload["error"] = result["error"]
                    
                await self.orchestrator._log_event(log_payload)
        except Exception as e:
            self.send_result(request_id, {
                "ok": False,
                "error": {"code": "BRIDGE_ERROR", "message": str(e)}
            })

    async def handle_unenroll_reference(self, request_id: Any, params: Dict[str, Any]):
        """Unenroll (remove) the stored face embedding for a session (fire-and-forget)."""
        service = self.services.get("face-recognition")
        if service is not None:
            try:
                await service.unenroll()
            except Exception:
                pass  # fire-and-forget
        # Always respond with ok — unenroll is best-effort
        self.send_result(request_id, {"ok": True})
        # Log the lifecycle event so the session log records when cleanup happened.
        if self.orchestrator and self.orchestrator.current_session_id:
            await self.orchestrator._log_event({
                "type": "lifecycle",
                "event": "unenrollment",
                "sessionId": self.orchestrator.current_session_id,
                "timestamp": __import__('datetime').datetime.now(
                    __import__('datetime').timezone.utc).isoformat(),
            })

    async def handle_start_service(self, request_id: Any, params: Dict[str, Any]):
        service_name = params.get("service")
        session_id = params.get("sessionId", "default-session")

        # Report import failures as a clear error (not just "Invalid service")
        if service_name not in self.service_classes:
            import_err = _service_import_errors.get(service_name)
            if import_err:
                self.send_error(
                    request_id, -32603,
                    f"Service '{service_name}' unavailable — missing dependency: {import_err}"
                )
            else:
                self.send_error(request_id, -32602, f"Invalid service: {service_name}")
            return

        if service_name in self.services:
            self.send_result(request_id, {"status": "already_running", "service": service_name})
            return

        service_cls = self.service_classes[service_name]
        
        try:
            if service_name == "eye-gaze":
                service = service_cls(session_id, vars(self.config), self.thread_safe_emit)
            else:
                service = service_cls(session_id, vars(self.config))
            await service.start()
        except Exception as e:
            self.send_error(request_id, -32603, f"Failed to start service {service_name}: {str(e)}")
            return

        self.services[service_name] = service
        self.send_result(request_id, {"status": "started", "service": service_name})

    async def handle_stop_service(self, request_id: Any, params: Dict[str, Any]):
        service_name = params.get("service")
        if service_name in self.services:
            await self.services[service_name].stop()
            del self.services[service_name]
            self.send_result(request_id, {"status": "stopped", "service": service_name})
        else:
            self.send_error(request_id, -32602, f"Service not running: {service_name}")

    def handle_query_status(self, request_id: Any, params: Dict[str, Any]):
        all_service_names = set(self.service_classes) | set(_service_import_errors)
        service_name = params.get("service")
        if service_name:
            if service_name in _service_import_errors:
                status = "unavailable"
            elif service_name in self.services:
                status = "running"
            else:
                status = "stopped"
            self.send_result(request_id, {"service": service_name, "status": status})
        else:
            statuses = {}
            for name in all_service_names:
                if name in _service_import_errors:
                    statuses[name] = "unavailable"
                elif name in self.services:
                    statuses[name] = "running"
                else:
                    statuses[name] = "stopped"
            self.send_result(request_id, statuses)

    async def handle_mock_detection(self, request_id: Any, params: Dict[str, Any]):
        service_name = params.get("service")
        if service_name in self.services:
            # We assume get_mock_event might be async in the future
            if asyncio.iscoroutinefunction(self.services[service_name].get_mock_event):
                event = await self.services[service_name].get_mock_event()
            else:
                event = self.services[service_name].get_mock_event()
                
            # Validate event against schema
            try:
                validate(instance=event, schema=self.schema)
                self.send_notification("detection", event)
                self.send_result(request_id, {"status": "mock_event_sent"})
            except ValidationError as e:
                self.send_notification("serviceError", {
                    "service": service_name,
                    "code": "CONTRACT_VIOLATION",
                    "message": f"Detection event failed contract validation: {e.message}"
                })
                self.send_error(request_id, -32001, "Contract violation")
        else:
            self.send_error(request_id, -32602, f"Service not running: {service_name}")

    async def handle_upload_clip(self, request_id: Any, params: Dict[str, Any]):
        """
        Handle the upload_clip JSON-RPC method.

        Expected params:
            tempFilePath : str  — absolute path to the .webm temp file
            metadata     : dict — ClipMetadata dict (includes token for backend POST)

        Runs the blocking ClipUploadService in a thread-pool executor so the
        async event loop is never blocked during ffmpeg encode or HTTP upload.
        """
        if _ClipUploadService is None:
            import_err = _service_import_errors.get("clip-upload", "unknown import error")
            self.send_error(
                request_id, -32603,
                f"ClipUploadService unavailable: {import_err}"
            )
            return

        temp_file_path = params.get("tempFilePath")
        metadata = params.get("metadata") or {}

        if not temp_file_path:
            self.send_error(request_id, -32602, "Missing tempFilePath parameter")
            return

        try:
            service = _ClipUploadService(self.config)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, service.upload_clip, temp_file_path, metadata
            )
            self.send_result(request_id, result)
        except RuntimeError as e:
            # Missing env vars — configuration error
            self.send_error(request_id, -32603, f"ClipUploadService config error: {str(e)}")
        except Exception as e:
            self.send_error(request_id, -32603, f"upload_clip internal error: {str(e)}")

    def send_result(self, request_id: Any, result: Any):
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        self.write_stdout(response)

    def send_error(self, request_id: Any, code: int, message: str):
        response = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id
        }
        self.write_stdout(response)

    def send_notification(self, method: str, params: Any):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self.write_stdout(notification)

    def write_stdout(self, data: Dict[str, Any]):
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.environ.get("LUMINA_CONFIG_PATH", os.path.join(base_dir, "config.json"))
    schema_path = os.path.join(base_dir, "specs", "ai-service-contract.json")
    
    router = AIRouter(config_path, schema_path)
    asyncio.run(router.run())


