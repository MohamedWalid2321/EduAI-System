# Lumina — LUMINA Ecosystem

**An AI-powered Learning Management System with integrated exam proctoring.**

Lumina is a full-stack LMS made up of two repositories that work together as one product:

| Part | Repository | Stack |
|---|---|---|
| **API / Backend** + **AI services** + **Proctoring desktop app** | [`LUMINA-Ecosystem`](https://github.com/MohamedWalid2321/LUMINA-Ecosystem) | .NET 8, Python (FastAPI), Electron |
| **Web frontend (Lumina)** | [`LMS-Project`](https://github.com/youssefabobaker/LMS-Project) | Angular 17 |

> The frontend is wired into the backend repo as the git submodule `Lumina_Web_Platform`.

- **LMS product name:** **Lumina**
- **Proctoring product name:** **Lumina AI Proctoring** (a.k.a. **Lumina Proctor**)
- **Live demo:** [lumina-lms-site.vercel.app](https://lumina-lms-site.vercel.app/)

---

## 1. What this project does

Lumina lets institutions run courses, assignments, and quizzes online, and lets students take **proctored exams** monitored by AI models that watch for gaze-away behavior, unauthorized objects, multiple faces, voice/speech activity, and identity mismatches — flagging suspicious activity for instructor review instead of relying on a human watching a webcam feed.

The system is split into four cooperating services:

1. **Lumina Web (frontend)** – Angular SPA used by admins, instructors, and students.
2. **Lumina API (backend)** – .NET 8 Web API that owns all academic data, auth, payments, and business rules.
3. **AI Proctoring service** – a Python/FastAPI service (deployable on [Modal](https://modal.com) with GPU) that runs the actual computer-vision / audio models.
4. **Lumina AI Proctoring desktop app** – an Electron app students install to take exams in a locked-down window, which talks to a local Python bridge and forwards video/audio events to the AI service.

```
                     ┌────────────────────────────┐
                     │      Lumina Web (Angular)  │  admins / instructors / students
                     └──────────────┬─────────────┘
                                    │ REST / JWT
                                    ▼
                     ┌────────────────────────────┐
                     │      Lumina API (.NET 8)   │  courses, quizzes, users,
                     │  Onion / Clean Architecture │  payments, notifications,
                     └───────┬─────────┬──────────┘  cheating reports, risk scores
                             │         │
                 SQL Server  │         │  Redis (cache / Hangfire jobs)
                             ▼         ▼
                     ┌────────────────────────────┐
                     │   Lumina AI Proctoring      │  gaze · objects · faces
                     │   (Python / FastAPI/Modal)  │  speech · anti-spoofing
                     └──────────────▲─────────────┘
                                    │ HTTPS
                     ┌──────────────┴─────────────┐
                     │ Lumina Proctoring Desktop   │  Electron shell + local
                     │ App (Electron + Python      │  Python bridge (port 5050),
                     │ bridge)                     │  exam lockdown, risk feed
                     └────────────────────────────┘
```

---

## 2. Repository layout

### `LUMINA-Ecosystem` (backend + AI)

```
LUMINA-Ecosystem/
├── AI/                          # Cloud/GPU proctoring service (FastAPI + Modal)
│   ├── Models/
│   │   ├── EyeGazeDetection/     # Gaze tracking (MediaPipe face landmarker)
│   │   ├── FaceDetection/        # SCRFD face detector
│   │   ├── FaceAntiSpoofing/     # MiniFASNet liveness check
│   │   ├── Face_Recognition_Service/  # ArcFace ONNX + Redis-cached enrollment
│   │   ├── objectDetectionYolo/  # YOLO-based object/phone detection
│   │   ├── objectDetectionOWL_VIT/    # Open-vocabulary object detection
│   │   └── SpeechDetection/      # Voice-activity / speech detection
│   ├── routes/                   # face, gaze, object, speech route modules
│   ├── schemas/
│   └── main.py                   # FastAPI app + Modal deployment entrypoint
│
├── Backend/Edu_Ai_API_Solution/  # Lumina API — .NET 8, Onion/Clean Architecture
│   ├── Core/
│   │   ├── DomainLayer/          # Entities, enums, domain exceptions
│   │   ├── ServiceAbstractionLayer/  # Service interfaces
│   │   └── ServiceLayer/         # Business logic implementation
│   ├── Infrastructure/
│   │   ├── persistenceLayer/     # EF Core, repositories, unit of work
│   │   └── PresentationLayer/    # Cross-cutting presentation concerns
│   ├── Shared/                   # DTOs, error models, constants
│   └── Edu_Ai_API/               # ASP.NET Core Web API host, Controllers, middleware
│
├── Proctor_Desktop_App/          # Lumina AI Proctoring — Electron desktop client
│   ├── frontend/                 # Electron renderer (exam, login, ID-verification,
│   │                              #   AI-readiness check, results, session report pages)
│   ├── python_bridge/            # Local Python service (port 5050) that
│   │                              #   orchestrates the exam session, lockdown,
│   │                              #   and calls into the cloud AI service
│   └── AI/                       # Bundled copy of the AI models used offline/locally
│
└── Lumina_Web_Platform/          # git submodule → LMS-Project (frontend)
```

### `LMS-Project` (frontend)

```
LMS-Project/
├── src/app/
│   ├── core/                     # guards, interceptors, core services
│   ├── features/
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── department-management/
│   │   ├── role-management/
│   │   ├── user-management/
│   │   ├── user-profile/
│   │   ├── course-management/
│   │   ├── content/
│   │   ├── lectures/
│   │   ├── assignments/
│   │   ├── quizzes/
│   │   ├── payment/
│   │   ├── notification/
│   │   ├── desktop-guide/        # onboarding into the Proctoring desktop app
│   │   └── landing/
│   ├── models/
│   └── shared/components/
└── specs/                        # spec-driven feature specs (spec-kit style)
```

---

## 3. Features

### Lumina (LMS)

- **Auth & roles** — JWT-based authentication, role-based access (Admin / Instructor / Student) via `RolesController`, `AccountController`, `AuthunticationController`.
- **Academic structure** — Departments, academic years, semesters (Fall/Spring/Summer) and course lifecycle (`Drafted → Published → Archived`).
- **Course management** — course creation, enrollment (Active / Completed / Dropped), content and lecture management.
- **Assignments** — creation, file attachments, student submissions, grading (`AssignmentController`, `AssignmentSubmissionController`).
- **Quizzes** — multiple-choice and true/false questions, timed attempts, auto-scoring (`QuizController`, `QuestionController`, `QuizAttemptsController`).
- **Payments** — tuition/book/activity fees via the **Paymob** gateway (`PaymentController`, `FeeController`).
- **Notifications** — in-app/email notifications (`NotificationController`, MailKit + Hangfire background jobs).
- **Contact / support** — `ContactController` for inbound inquiries.

### Lumina AI Proctoring

- **Identity verification** — face enrollment and recognition (SCRFD detector + ArcFace ONNX embeddings), with **anti-spoofing** (MiniFASNet) to reject photos/screens.
- **Gaze tracking** — MediaPipe-based eye-gaze detection to flag looking away from the screen.
- **Object detection** — YOLO and OWL-ViT models to detect phones, extra people, notes, etc. in the exam feed.
- **Speech / voice-activity detection** — flags talking during a locked-down exam.
- **Risk analysis** — signals from all models are aggregated into a per-session risk score (`RiskAnalysisController`, `python_bridge/risk_estimator.py`).
- **Cheating reports** — instructors review flagged incidents (`CheatingReportController`).
- **Exam lockdown desktop app** — students install **Lumina AI Proctoring** (Electron) which locks down the desktop, runs an AI-readiness check, verifies identity, and streams events to the cloud AI service during the exam, then produces a session report.

---

## 4. Tech stack

| Layer | Technology |
|---|---|
| Frontend | Angular 17, Bootstrap 5, ng-bootstrap, SweetAlert2, `jwt-decode` |
| Backend API | ASP.NET Core 8 Web API, Onion/Clean Architecture (Domain / ServiceAbstraction / Service / Persistence / Presentation) |
| Data | Entity Framework Core 8, SQL Server |
| Auth | ASP.NET Core Identity + JWT Bearer |
| Caching / jobs | StackExchange.Redis, Hangfire (background jobs + dashboard) |
| Mapping / logging | Mapster, Serilog (compact JSON formatting) |
| Email | MailKit + HTML email templates |
| Payments | Paymob payment gateway |
| API docs | Swashbuckle (Swagger) |
| AI service | Python, FastAPI, deployable on Modal (GPU), Upstash Redis for face-enrollment cache |
| CV / ML models | Ultralytics YOLO, OWL-ViT, MediaPipe, SCRFD, ArcFace (ONNX Runtime), MiniFASNet, PyTorch/TensorFlow |
| Desktop app | Electron 33, Node.js, local Python bridge (Flask/FastAPI-style HTTP service) |

---

## 5. Getting started

You will generally run **four** things locally: the SQL database, the .NET API, the Angular app, and (optionally) the AI service / desktop app.

### 5.1 Clone

```bash
git clone --recurse-submodules https://github.com/MohamedWalid2321/LUMINA-Ecosystem.git
cd LUMINA-Ecosystem
# if you forgot --recurse-submodules:
git submodule update --init --recursive
```

The `LMS-Project` frontend can also be cloned standalone:

```bash
git clone https://github.com/youssefabobaker/LMS-Project.git
```

### 5.2 Backend — Lumina API (.NET 8)

```bash
cd Backend/Edu_Ai_API_Solution
dotnet restore
```

Configure `Edu_Ai_API/appsettings.json` (or `dotnet user-secrets`) with:
- SQL Server connection string
- JWT signing key / issuer / audience
- Redis connection string
- SMTP settings (MailKit)
- Paymob API credentials

Apply migrations and run:

```bash
dotnet ef database update --project Infrastructure/persistenceLayer --startup-project Edu_Ai_API
dotnet run --project Edu_Ai_API
```

Swagger UI will be available at the host's `/swagger` endpoint once running.

### 5.3 Frontend — Lumina Web (Angular 17)

> A hosted build is live at **https://lumina-lms-site.vercel.app/** — useful for a quick look without running anything locally. Note it's wired to whichever backend/API instance the team has deployed, not your local API.

```bash
cd Lumina_Web_Platform   # or the standalone LMS-Project clone
npm install
ng serve
```

Visit `http://localhost:4200`. Point the app's API base URL (in `src/environments/`) at your running Lumina API instance.

### 5.4 AI Proctoring service (Python / FastAPI)

```bash
cd AI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variables:
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` — face enrollment cache
- `FACE_ENROLLMENT_TTL_SECONDS` (optional, default `10800`)

Run locally:

```bash
uvicorn main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Or deploy to Modal (GPU-backed, models preloaded/warm):

```bash
modal secret create eduai-upstash-redis \
    UPSTASH_REDIS_REST_URL=... \
    UPSTASH_REDIS_REST_TOKEN=...
python -m modal deploy main.py
```

### 5.5 Lumina AI Proctoring desktop app (Electron)

```bash
cd Proctor_Desktop_App
npm install
pip install -r python_bridge/requirements.txt
cp config.example.json config.json
# edit config.json: set "baseUrl" to your Lumina API endpoint (must be https://)
npm start
```

Verify the local Python bridge is up:

```bash
curl http://127.0.0.1:5050/ping
# → {"status":"ok","version":"1.0.0","timestamp":"..."}
```

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| `BRIDGE_FAILED` on startup | Python not on PATH, or port 5050 in use | Check `python --version`; change `pythonPort` in `config.json` |
| `CONFIG_ERROR` | Missing/malformed `config.json` | Re-copy from `config.example.json` |
| `INSECURE_PROTOCOL` | `baseUrl` uses `http://` | Switch to `https://` |
| Two app windows open | Single-instance lock failed | Close all Lumina processes and relaunch |

---

## 6. API surface (Lumina API controllers)

| Controller | Responsibility |
|---|---|
| `AuthunticationController`, `AccountController` | Login, registration, tokens |
| `RolesController`, `UsersController` | Role-based access, user management |
| `DepartmentController`, `AcademicYearController` | Academic org structure |
| `CourseController` | Course CRUD, publishing, enrollment |
| `ContentController`, `LectureController` | Course content and lectures |
| `AssignmentController`, `AssignmentSubmissionController` | Assignments and submissions/grading |
| `QuizController`, `QuestionController`, `QuizAttemptsController` | Quiz authoring and attempts |
| `FeeController`, `PaymentController` | Tuition/fees and Paymob payments |
| `NotificationController` | Notifications/announcements |
| `CheatingReportController`, `RiskAnalysisController` | Proctoring incident review and risk scoring |
| `ContactController` | Contact/support messages |

Full request/response schemas are available via Swagger once the API is running.

---

## 7. Security & privacy notes

Because Lumina processes student identity data (face enrollment) and exam session recordings/events:

- All exam-session traffic between the desktop app and the AI service must use HTTPS (`config.json` enforces this).
- Face enrollment vectors are cached in Redis with a TTL (default 3 hours) rather than stored indefinitely.
- Access to cheating reports and risk analysis should be restricted to authorized instructor/admin roles.
- Treat JWT signing keys, Paymob credentials, SMTP credentials, and Upstash/Redis tokens as secrets — never commit them; use `dotnet user-secrets`, `.env` files, or Modal secrets.

---

## 8. Contributors

- [@MohamedWalid2321](https://github.com/MohamedWalid2321)
- [@youssefabobaker](https://github.com/youssefabobaker)
- [@Honda1010](https://github.com/Honda1010)
- [@youssif-mohamed1](https://github.com/youssif-mohamed1)
- [@3La20300](https://github.com/3La20300)
- [@Youssef-marawan](https://github.com/Youssef-marawan)

## 9. License

Add your chosen license here (e.g. MIT). If a `LICENSE` file already exists in either repository, reference it directly instead.
