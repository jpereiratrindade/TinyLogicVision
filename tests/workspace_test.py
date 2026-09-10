#!/usr/bin/env python3
"""Verify isolated workspace selection for the local Web server."""

import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "web" / "server.py"


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tinyvision-workspace-test-") as temporary:
        workspace = Path(temporary) / "isolated workspace"
        port = available_port()
        process = subprocess.Popen(
            [sys.executable, str(SERVER), "--workspace", str(workspace),
             "--port", str(port), "--no-browser"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            status = None
            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    break
                except Exception:
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
            if status is None:
                output, error = process.communicate(timeout=2)
                raise AssertionError(f"workspace server failed to start\n{output}\n{error}")
            assert Path(status["runtime_root"]) == workspace.resolve()
            manifest = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
            assert manifest["schema"] == "tinylogicvision.workspace/v1"
            for directory in ("uploads", "datasets", "models", "runs", "manifests"):
                assert (workspace / directory).is_dir()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
    print("TinyLogicVision isolated workspace test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
