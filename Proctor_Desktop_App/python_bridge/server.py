"""
server.py — Flask HTTP bridge for Lumina AI.

Binds exclusively to 127.0.0.1 (loopback) so the bridge is never exposed to
the local network. Electron communicates via HTTP on localhost only.

Usage (spawned by Electron main process):
    python server.py --port 5050 --config /path/to/config.json

GET /ping
    Returns: {"status": "ok", "version": "1.0.0", "timestamp": "<ISO-UTC>"}
    Electron polls this endpoint up to 20 times at 500 ms intervals (10 s total)
    to determine when the bridge is ready. See contracts/ping.md.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from flask import Flask, jsonify
from flask_cors import CORS

from auth import auth_bp
from exam import exam_bp
from lockdown import lockdown_bp
from config import load_config, ConfigError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRIDGE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Apply CORS to all routes.
# Even though the bridge only binds to 127.0.0.1, CORS headers are required
# because Electron renderer pages use the file:// origin.
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(exam_bp)
app.register_blueprint(lockdown_bp)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():
    """Health-check endpoint polled by Electron during startup."""
    return jsonify(
        {
            "status": "ok",
            "version": BRIDGE_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    ), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lumina AI Python bridge")
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to listen on (default: 5050)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Absolute path to config.json (default: project root)",
    )
    args = parser.parse_args()

    # Resolve config path: explicit arg → project root fallback
    if args.config:
        config_path = args.config
    else:
        # When running from python_bridge/, config.json is one level up
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

    # Load and validate config — exits with code 1 on any ConfigError
    try:
        cfg = load_config(config_path)
    except ConfigError:
        # ConfigError.emit_stderr() was already called inside load_config
        sys.exit(1)

    # Store validated base_url so auth.py can access it via current_app.config
    app.config["BASE_URL"] = cfg.base_url

    # Use port from CLI arg if provided, otherwise from config
    port = args.port if args.port != 5050 or not hasattr(cfg, "python_port") else cfg.python_port

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
