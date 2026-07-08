# Lumina AI Proctoring

AI-powered proctoring desktop application built with Electron + Python.

## Prerequisites

- Node.js 20 LTS
- Python 3.11+
- pip

## Setup

**1. Install Node dependencies**
```sh
npm install
```

**2. Install Python dependencies**
```sh
pip install -r python_bridge/requirements.txt
```

**3. Create your config file**
```sh
cp config.example.json config.json
```
Edit `config.json` and set `baseUrl` to your LMS API endpoint (must be `https://`).

**4. Start the app**
```sh
npm start
```

**5. Verify the bridge**

While the app is running:
```sh
curl http://127.0.0.1:5050/ping
# → {"status":"ok","version":"1.0.0","timestamp":"..."}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Unable to Start" — `BRIDGE_FAILED` | Python not on PATH or port 5050 in use | Run `python --version`; change `pythonPort` in config.json |
| "Unable to Start" — `CONFIG_ERROR` | Missing / malformed config.json | re-copy `config.example.json` |
| "Unable to Start" — `INSECURE_PROTOCOL` | `baseUrl` uses `http://` | Change to `https://` |
| Two windows open | Single-instance lock failed | Close all Lumina processes and relaunch |

## Scripts

| Command | Purpose |
|---------|---------|
| `npm start` | Launch in development mode |
| `npm run build` | Package for distribution (electron-builder) |

See `specs/001-foundation/quickstart.md` for full local development guide.
