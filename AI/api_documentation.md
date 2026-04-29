# EduAI Proctoring System — Complete API Documentation

**Service**: EduAI Proctoring AI (Modal Deployment)
**Framework**: FastAPI (Python)
**Transport**: REST / HTTPS — JSON or multipart/form-data
**Base URL (all endpoints)**: `https://eduai-proctoring.modal.run` *(replace with actual Modal URL after deploy)*
**Auth**: None enforced in code — Modal Secrets recommended for production

---

## Table of Contents

| # | Route File | Method | Path | Summary |
|---|---|---|---|---|
| 1 | `face_route.py` | POST | `/analysis/enroll` | Enroll student (base64) |
| 2 | `face_route.py` | POST | `/analysis/enroll-file` | Enroll student (file upload, testing) |
| 3 | `face_route.py` | POST | `/analysis/unenroll` | Remove enrollment |
| 4 | `face_route.py` | POST | `/analysis/verify` | Verify frame (base64) |
| 5 | `face_route.py` | POST | `/analysis/verify-file` | Verify frame (file upload, testing) |
| 6 | `face_route.py` | POST | `/analysis/face-detection` | Face count only (base64) |
| 7 | `face_route.py` | POST | `/analysis/face-detection-file` | Face count only (file upload, testing) |
| 8 | `face_route.py` | POST | `/analysis/face-frame` | Legacy compare two images (file upload) |
| 9 | `face_route.py` | POST | `/analysis/face-base64` | Legacy compare two images (base64) |
| 10 | `object_route.py` | POST | `/analysis/object-video` | YOLO object detection on 7-s video |
| 11 | `object_route.py` | POST | `/analysis/object-frame` | YOLO object detection on single frame |
| 12 | `new_object_route.py` | POST | `/analysis/detect_objects` | OWL-ViT zero-shot detection |
| 13 | `gaze_route.py` | POST | `/analysis/gaze-frames` | Eye gaze batch analysis |
| 14 | `gaze_route.py` | DELETE | `/analysis/gaze-frames/{session_id}` | Clear gaze session |
| 15 | `speech_route.py` | POST | `/analysis/speech-chunk` | Speech VAD analysis |
| 16 | `speech_route.py` | DELETE | `/analysis/speech-chunk/{session_id}` | Clear speech session |
| 17 | `main.py` | GET | `/health` | Health check |

> **Note**: `modal_route.py` contains duplicate implementations of endpoints 10, 11, 13, 8, 9 (`/object-frame`, `/object-video`, `/gaze-video`, `/face-frame`, `/face-base64`). These are not registered in `main.py` and are **not active** in the deployed app. Only the routes above (from `face_route.py`, `object_route.py`, `gaze_route.py`, `speech_route.py`, `new_object_route.py`) are mounted.

---

## Section 1 — Face Recognition Service

**Tag**: `Face Recognition`
**Router prefix**: `/analysis`
**Primary caller**: .NET backend via JSON (base64)
**Models used**: SCRFD (face detection), ArcFace ONNX `w600k_r50.onnx` (recognition), MiniFASNetV2 (liveness)
**Embedding store**: Upstash Redis (TTL default 10 800 s = 3 h) + in-memory cache

### Standard Face Recognition Response Object

All verification/comparison endpoints wrap their result as:
```json
{
  "face_recognition": {
    "session_id": "string | null",
    "liveness_score": "string (e.g. '89.21%')",
    "num_faces": "integer",
    "quality": "string (e.g. '75.00%')",
    "id": 3,
    "timestamp": "ISO-8601 string",
    "probability": "string (e.g. '98.21%')",
    "evidence": "string"
  }
}
```

> `probability`, `liveness_score`, and `quality` are all returned as percentage strings (multiplied ×100 by `_build_response`).

---

### 1.1 POST `/analysis/enroll`
**Purpose**: Compute and persist an averaged ArcFace embedding for a student session.
**Content-Type**: `application/json`
**Used by**: .NET backend at exam start

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Unique exam session key. Stored in Redis as key `face:enrollment:<session_id>`. |
| `references` | `string[]` | ✅ | One or more base64-encoded images (JPEG/PNG). Raw base64 or Data-URI both accepted. |

#### Success Response (200)
```json
{
  "success": true,
  "session_id": "exam_001_user_42",
  "num_images": 2
}
```

#### Error Responses
| Code | Condition | Example |
|:---|:---|:---|
| 422 | `references` list is empty | `{"detail": "references list is empty."}` |
| 422 | No face in image *i* | `{"detail": "No face detected in reference image 0"}` |
| 422 | Multiple faces in image *i* | `{"detail": "Multiple faces in reference image 0"}` |
| 422 | Redis persistence failed | `{"detail": "Failed to persist enrollment embedding in Redis."}` |
| 422 | Invalid base64 encoding | `{"detail": "Invalid base64 encoding for reference[0]."}` |
| 422 | Corrupt/unreadable image | `{"detail": "Could not decode reference[0] — unsupported or corrupt image."}` |

#### Business Logic
- Detects one face per image using SCRFD. Rejects if zero or multiple faces.
- Computes 512-D ArcFace embedding per image.
- **Averages** all embeddings and L2-normalises → single reference vector.
- Writes to Upstash Redis and local in-memory cache.
- TTL: controlled by env var `FACE_ENROLLMENT_TTL_SECONDS` (default `10800`).

---

### 1.2 POST `/analysis/enroll-file`
**Purpose**: Identical to `/enroll` but accepts uploaded image files. For Postman/testing.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` (Form) | ✅ | Unique session identifier. |
| `references` | `file[]` (File) | ✅ | One or more image files. |

#### Response
Same as `1.1 /enroll`.

---

### 1.3 POST `/analysis/unenroll`
**Purpose**: Delete cached ArcFace embedding for a session (Redis + in-memory).
**Content-Type**: `application/x-www-form-urlencoded`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` (Form) | ✅ | Session ID to remove. |

#### Success Response (200)
```json
{
  "success": true,
  "session_id": "exam_001_user_42"
}
```

#### Notes
- Always returns 200 — no error if session does not exist.
- Call this at exam end to free Redis storage.

---

### 1.4 POST `/analysis/verify`
**Purpose**: Per-frame identity verification against enrolled template.
**Content-Type**: `application/json`
**Used by**: .NET backend on every proctoring frame

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Session ID previously enrolled via `/enroll`. |
| `frame` | `string` | ✅ | Base64-encoded webcam frame. |

#### Success Response (200)
Returns the **Standard Face Recognition Response Object** (see Section 1 header).

**Evidence values by scenario**:
| Scenario | `evidence` | `probability` | `liveness_score` |
|:---|:---|:---|:---|
| Verified match | `"Authorised person verified (similarity: 0.9821)"` | similarity ×100% | real score ×100% |
| No face | `"No face detected in frame"` | `"85.00%"` | `"0.0%"` |
| Multiple faces | `"Multiple faces detected: N faces in frame"` | `"95.00%"` | `"0.0%"` |
| Spoof detected | `"Spoof detected — liveness score X.XXXX (threshold 0.7)"` | `"-100.0%"` | liveness ×100% |
| Mismatch | `"Face does not match reference identity (similarity: X.XXXX)"` | similarity ×100% | real score ×100% |
| No enrollment | `"No enrollment found for session 'ID'"` | `"0.0%"` | `"0.0%"` |

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | Invalid base64 or corrupt image |

#### Decision Thresholds
| Threshold | Value | Configured In |
|:---|:---|:---|
| Cosine similarity (match) | `>= 0.5` | `face_recognition.py` L56, configurable via `FaceRecognition(similarity_threshold=…)` |
| Liveness healthy | `>= 0.7` | `face_recognition.py` L61 |
| Liveness risk trigger (active challenge) | `< 0.5` for 3 consecutive frames | `face_recognition.py` L63 |
| SCRFD detection confidence | `>= 0.5` | `face_recognition.py` L56 |

---

### 1.5 POST `/analysis/verify-file`
**Purpose**: Identical to `/verify` but accepts uploaded file. For Postman/testing.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` (Form) | ✅ | Session ID from `/enroll`. |
| `frame` | `file` (File) | ✅ | Webcam frame image file. |

#### Response
Same as `1.4 /verify`.

---

### 1.6 POST `/analysis/face-detection`
**Purpose**: Count faces in a single frame without identity verification.
**Content-Type**: `application/json`
**Used by**: .NET backend for lightweight face presence checks

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Session identifier (echoed in response). |
| `frame` | `string` | ✅ | Base64-encoded frame. |

#### Success Response (200)
```json
{
  "session_id": "exam_001_user_42",
  "timestamp": "2026-04-26T01:39:20.123456",
  "num_faces": 0,
  "evidence": "no_face_detected",
  "probability": 0.85
}
```

**Evidence values**:
| `num_faces` | `evidence` | `probability` |
|:---|:---|:---|
| 0 | `"no_face_detected"` | `0.85` |
| 1 | `"One face detected"` | *(not set — field absent)* |
| >1 | `"multiple_faces"` | `0.95` |

> Note: `probability` is a raw float here, **not** a percentage string. Only the full face recognition pipeline returns percentage strings.

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | Invalid base64 or corrupt image |
| 500 | Internal inference error |

---

### 1.7 POST `/analysis/face-detection-file`
**Purpose**: Identical to `/face-detection` but accepts uploaded file. For Postman/testing.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` (Form) | ✅ | Session identifier. |
| `frame` | `file` (File) | ✅ | Image file. |

#### Response
Same as `1.6 /face-detection`.

---

### 1.8 POST `/analysis/face-frame` *(Legacy)*
**Purpose**: Stateless face comparison — accepts two image files directly.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `frame` | `file` (File) | ✅ | Current exam webcam capture. |
| `reference` | `file` (File) | ✅ | Authorised student ID photo. |
| `session_id` | `string` (Form) | ❌ | Optional. If provided, reference embedding is cached on first call. |

#### Response
Same as **Standard Face Recognition Response Object**.

#### Business Logic
- If `session_id` is provided and enrollment is already cached, delegates to `verify()` (skips re-computing reference embedding).
- If `session_id` is provided and no cache exists, auto-enrolls the reference image then verifies.
- If `session_id` is `null`, runs full stateless detection+FAS+FR on both images every call.

---

### 1.9 POST `/analysis/face-base64` *(Legacy)*
**Purpose**: Stateless face comparison — accepts two base64-encoded images in JSON body.
**Content-Type**: `application/json`

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `frame` | `string` | ✅ | Base64-encoded webcam frame. |
| `reference` | `string` | ✅ | Base64-encoded reference ID photo. |
| `session_id` | `string` | ❌ | Optional. Enables embedding caching (same logic as 1.8). |

#### Response
Same as **Standard Face Recognition Response Object**.

---

## Section 2 — Object Detection Service

**Tag**: `Object Detection`
**Router prefix**: `/analysis`

### Standard Object Detection Response Object
```json
{
  "id": 2,
  "timestamp": "ISO-8601 string",
  "flag": "boolean",
  "propability": "float (max confidence across flagged frames)",
  "evidence": "string (comma-separated restricted items, or 'No suspicious activity detected')"
}
```
> Note: Field name is spelled `propability` (sic) in the codebase.

---

### 2.1 POST `/analysis/object-video`
**Purpose**: Process a 7-second video for prohibited items using YOLOv8.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `video` | `file` (File) | ✅ | Video file (MP4 recommended, ~7 seconds). |

#### Success Response (200)
```json
{
  "object_detection": {
    "id": 2,
    "timestamp": "2026-04-26T01:39:20.123456",
    "flag": true,
    "propability": 0.8721,
    "evidence": "Earphone, Mobile_phone"
  },
  "frames_processed": 16,
  "duration_seconds": 7,
  "early_stopped": true
}
```

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | Cannot open video / no frames extracted |

#### Business Logic & Thresholds
| Parameter | Value | Description |
|:---|:---|:---|
| Effective FPS | 12 | Frames captured from video |
| Target frames | 84 (7 × 12) | Max frames before stopping |
| YOLO stride | 5 | Run YOLO only on every 5th frame |
| YOLO confidence | `>= 0.4` | Minimum confidence to register a detection |
| Early stop count | 10 | If one evidence type appears in ≥10 sampled frames, stop early |
| Restricted classes | `Mobile_phone`, `Earphone`, `headset`, `smart_watch`, `sunglasses`, `cap` | |

**Aggregation**: Takes **max probability** across all processed frames; `flag = true` if any frame was flagged.

---

### 2.2 POST `/analysis/object-frame`
**Purpose**: YOLO detection on a single image. Testing only.
**Content-Type**: `multipart/form-data`

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `image` | `file` (File) | ✅ | JPEG or PNG image file. |

#### Success Response (200)
```json
{
  "id": 2,
  "timestamp": "2026-04-26T01:39:20.123456",
  "flag": false,
  "propability": 0.0,
  "evidence": "None"
}
```

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | Cannot decode image (`"Invalid image"`) |

---

### 2.3 POST `/analysis/detect_objects`
**Purpose**: Zero-shot object detection using OWL-ViT (Google `owlvit-base-patch32`).
**Content-Type**: `multipart/form-data`
**Note**: Alternative to YOLO. Runs on CPU only.

#### Request Fields
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `file` | `file` (File) | ✅ | Image file (must be `image/*` MIME type). |

#### Success Response (200)
```json
{
  "id": 2,
  "timestamp": "2026-04-26T01:39:20.123456",
  "probability": 0.4231,
  "evidence": "Detected: Mobile phone, Earphone"
}
```
When nothing found:
```json
{
  "id": 2,
  "timestamp": "2026-04-26T01:39:20.123456",
  "probability": 0.0,
  "evidence": "No restricted items detected."
}
```

#### Error Responses
| Code | Condition |
|:---|:---|
| 400 | File is not an image (`"File must be an image format (jpeg, png, etc.)."`) |
| 500 | Inference error (`"Error processing image: <detail>"`) |

#### Thresholds
| Parameter | Value | Description |
|:---|:---|:---|
| OWL-ViT score threshold | `> 0.1` | Minimum score to register a candidate label |
| Candidate labels | `Mobile phone`, `Earphone`, `headset`, `smart watch`, `sunglasses`, `cap` | Slightly different capitalisation from YOLO |

---

## Section 3 — Eye Gaze Detection Service

**Tag**: `Gaze Detection`
**Router prefix**: `/analysis`
**Model**: MediaPipe-based `GazeDetector` (via `Gaze.py`)
**State**: Per-session `GazeSession` object managed by `SessionManager`

---

### 3.1 POST `/analysis/gaze-frames`
**Purpose**: Process a batch of webcam frames and classify gaze direction.
**Content-Type**: `application/json`

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Unique user/exam session ID. |
| `frames` | `string[]` | ✅ | Array of base64-encoded JPEG frames. Must be non-empty. |
| `fps` | `integer` | ❌ | Capture FPS (default: 30). Used to compute per-frame timestamps. |

#### Success Response (200)
```json
{
  "session_id": "exam_001_user_42",
  "frames_processed": 30,
  "summary_flag": "AWAY_SHORT",
  "verdicts": [
    {
      "id": 1,
      "timestamp": "2026-04-26T01:39:20.123456",
      "flag": "ON_SCREEN",
      "probability": 0.0,
      "evidence": "ON_SCREEN"
    },
    {
      "id": 1,
      "timestamp": "2026-04-26T01:39:20.156789",
      "flag": "AWAY_SHORT",
      "probability": 0.5,
      "evidence": "AWAY_SHORT"
    }
  ]
}
```

**Flag values per verdict**:
| Flag | `probability` | Meaning |
|:---|:---|:---|
| `INITIALIZING` | `0.0` | Session calibrating (first 90 frames) |
| `ON_SCREEN` | `0.0` | Gaze within tolerance of baseline |
| `NO_FACE` | `1.0` | No face detected in frame |
| `AWAY_SHORT` | `0.5` | Looking away 3–5 seconds |
| `AWAY_LONG` | `1.0` | Looking away ≥5 seconds |

**`summary_flag`**: The **most severe** flag seen in the batch (severity order: `INITIALIZING < ON_SCREEN < NO_FACE < AWAY_SHORT < AWAY_LONG`).

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | `frames` array is empty |
| 422 | No frames could be processed |
| 500 | Unexpected internal error |

#### Business Logic & Thresholds
| Parameter | Value | Set In |
|:---|:---|:---|
| Calibration window | 90 frames | `localMain.py` L17 |
| Smoothing buffer | 3 frames | `localMain.py` L16 |
| Tolerance | `max(0.07, baseline_std × 3.5)` | Dynamic, computed from baseline |
| AWAY_SHORT trigger | ≥ 3.0 s continuous | `localMain.py` L18 |
| AWAY_LONG trigger | ≥ 5.0 s continuous | `localMain.py` L19 |

---

### 3.2 DELETE `/analysis/gaze-frames/{session_id}`
**Purpose**: Free `GazeSession` state from memory at exam end.
**Content-Type**: N/A

#### Path Parameter
| Parameter | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Session to clear. |

#### Success Response (200)
```json
{ "detail": "Session 'exam_001_user_42' cleared." }
```

---

## Section 4 — Speech Detection Service

**Tag**: `Speech Detection`
**Router prefix**: `/analysis`
**Model**: Silero VAD (PyTorch Hub `snakers4/silero-vad`)
**State**: Per-session `SpeechSession` managed by `SpeechSessionManager`

---

### 4.1 POST `/analysis/speech-chunk`
**Purpose**: Analyse one audio chunk for speech and continuous speaking violations.
**Content-Type**: `application/json`

#### Request Body
| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Unique user/exam session ID. |
| `audio_chunk` | `string` | ✅ | Base64-encoded raw float32 PCM bytes (mono, 16 kHz). Chunk length: 1–2 seconds recommended. |
| `sample_rate` | `integer` | ❌ | Must be exactly `16000`. Default: `16000`. |

#### Success Response (200)
```json
{
  "session_id": "exam_001_user_42",
  "verdict": {
    "timestamp": "2026-04-26T01:39:20.123456+00:00",
    "flag": "SPEAKING",
    "speech_probability": 0.9241,
    "speaking_duration": 3.12,
    "evidence": "Speech detected (VAD confidence: 0.92). Continuous duration so far: 3.1s."
  }
}
```

**Flag values**:
| Flag | Condition | `speaking_duration` |
|:---|:---|:---|
| `SILENCE` | VAD prob ≤ 0.5 and not in active window | `0.0` |
| `SPEAKING` | VAD prob > 0.5, continuous speech < 5 s | Accumulated seconds |
| `SPEAKING` | Short silence (< 1.5 s) within speech window | Accumulated seconds |
| `SPEECH_VIOLATION` | Continuous speech ≥ 5.0 s | `0.0` (timer reset after trigger) |

#### Error Responses
| Code | Condition |
|:---|:---|
| 422 | `audio_chunk` is empty |
| 422 | `sample_rate` is not 16000 |
| 422 | Audio chunk too short (<512 samples = <32 ms) |
| 500 | Unexpected VAD inference error |

#### Business Logic & Thresholds
| Parameter | Value | Set In |
|:---|:---|:---|
| VAD confidence threshold | `0.5` | `speech_localMain.py` L32 |
| Violation trigger | `5.0` continuous seconds | `speech_localMain.py` L33 |
| Pause reset window | `1.5` seconds of silence | `speech_localMain.py` L34 |
| VAD window size | 512 samples per Silero pass | `speech_localMain.py` L39 |
| Max probability across windows | YES — batch forward pass | `_infer_windowed()` |

**Strike counting**: The `SPEECH_VIOLATION` flag is **reported** by the AI service; counting violations (e.g., 5 strikes = cheater) is handled by the main .NET backend.

---

### 4.2 DELETE `/analysis/speech-chunk/{session_id}`
**Purpose**: Free `SpeechSession` state from memory at exam end.
**Content-Type**: N/A

#### Path Parameter
| Parameter | Type | Required | Description |
|:---|:---|:---|:---|
| `session_id` | `string` | ✅ | Session to clear. |

#### Success Response (200)
```json
{ "detail": "Session 'exam_001_user_42' cleared." }
```

---

## Section 5 — System

### 5.1 GET `/health`
**Purpose**: Verify the API server is reachable and running.
**Content-Type**: N/A

#### Success Response (200)
```json
{ "status": "ok" }
```

---

## Section 6 — Common Error Codes

| HTTP Code | Meaning | When |
|:---|:---|:---|
| 200 | OK | Successful response (all endpoints) |
| 400 | Bad Request | Non-image file sent to OWL-ViT |
| 422 | Unprocessable Entity | Missing/invalid field, corrupt image, empty audio, wrong sample rate |
| 500 | Internal Server Error | Unhandled exception in model inference |

---

## Section 7 — Global Notes

### Modal Deployment Config
| Parameter | Value |
|:---|:---|
| GPU | NVIDIA L4 (configurable via `MODAL_GPU` env var) |
| Scale-down window | 600 seconds (containers idle for 10 min before teardown) |
| Thread pool (AI inference) | 4 workers (`ThreadPoolExecutor`) |

### Session Cleanup (Required)
Always call cleanup endpoints when an exam ends:
- `POST /analysis/unenroll` — remove face embedding from Redis
- `DELETE /analysis/gaze-frames/{session_id}` — free gaze state
- `DELETE /analysis/speech-chunk/{session_id}` — free speech state

Failure to clean up will cause Redis storage to grow and in-memory state to accumulate until container restart.

### Frame/Audio Encoding
- Images: base64-encoded JPEG or PNG. Data-URIs (`data:image/jpeg;base64,...`) are auto-stripped.
- Audio: base64-encoded raw **float32 PCM** bytes (not WAV/MP3). Must be 16 kHz mono.

### Recommended Timeouts (Client-Side)
| Endpoint Type | Recommended Timeout |
|:---|:---|
| Single frame / audio chunk | 10 seconds |
| Video (7-second clip) | 30 seconds |
| Enrollment | 15 seconds |

### Retry Behavior
No retry logic is implemented server-side. Clients should implement exponential backoff for 500 errors. 422 errors are client-side data issues and should not be retried without fixing the payload.

---

*Documentation generated: 2026-04-26 — EduAI Proctoring System v1.0.0*
