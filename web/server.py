#!/usr/bin/env python3
"""
TinyLogicVision Local Web Application Server v0.1
Orchestrates patch extraction, dataset authoring, model training,
classification, and evaluation using the TinyLogicVision CLI authority.
"""

import argparse
import csv
import datetime
import hashlib
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

# Optional Pillow support with fallback
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNTIME_ROOT = REPO_ROOT / ".tinyvision"

UPLOADS_DIR = RUNTIME_ROOT / "uploads"
DATASETS_DIR = RUNTIME_ROOT / "datasets"
MODELS_DIR = RUNTIME_ROOT / "models"
RUNS_DIR = RUNTIME_ROOT / "runs"
MANIFESTS_DIR = RUNTIME_ROOT / "manifests"

for d in (UPLOADS_DIR, DATASETS_DIR, MODELS_DIR, RUNS_DIR, MANIFESTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

RUNS_LOCK = threading.Lock()
ACTIVE_RUNS = {}


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    return cleaned if cleaned else "unnamed"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_png_patch(pixels_rgb: bytes, width: int, height: int, output_path: Path):
    """Save raw RGB pixels as PNG. Uses PIL if available or pure standard library."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PIL:
        img = PILImage.frombytes("RGB", (width, height), pixels_rgb)
        img.save(output_path, format="PNG")
        return

    import struct
    import zlib

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        start = y * width * 3
        raw.extend(pixels_rgb[start : start + width * 3])

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    with open(output_path, "wb") as f:
        f.write(png)


def load_image_dimensions_and_rgb(image_path: Path):
    """Returns (width, height, raw_rgb_bytes)"""
    if HAS_PIL:
        with PILImage.open(image_path) as img:
            rgb_img = img.convert("RGB")
            return rgb_img.width, rgb_img.height, rgb_img.tobytes()

    # Fallback to loading via Python's standard libraries or basic headers
    with open(image_path, "rb") as f:
        data = f.read()

    # Parse PNG header
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        import struct
        w, h = struct.unpack(">II", data[16:24])
        # Return dummy or basic RGB if PIL is missing
        return w, h, None

    return 0, 0, None


def is_safe_path(target_path: Path, allowed_roots=None) -> bool:
    if allowed_roots is None:
        allowed_roots = [RUNTIME_ROOT, REPO_ROOT]
    try:
        resolved = target_path.resolve()
        for root in allowed_roots:
            if root.resolve() in resolved.parents or resolved == root.resolve():
                return True
        return False
    except Exception:
        return False


class TinyVisionRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep server log clean
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"success": False, "error": str(message)}, status=status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.serve_file(STATIC_DIR / "index.html", "text/html")
        elif path == "/style.css":
            self.serve_file(STATIC_DIR / "style.css", "text/css")
        elif path == "/app.js":
            self.serve_file(STATIC_DIR / "app.js", "application/javascript")
        elif path == "/api/status":
            self.handle_api_status()
        elif path == "/api/image":
            self.handle_api_image(query)
        elif path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/") :]
            self.handle_api_run_status(run_id)
        elif path == "/api/datasets":
            self.handle_api_list_datasets()
        elif path == "/api/models":
            self.handle_api_list_models()
        else:
            self.send_error_json("Not found", 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            self.handle_api_upload()
        elif path == "/api/datasets/create":
            self.handle_api_create_dataset()
        elif path == "/api/train":
            self.handle_api_train()
        elif path == "/api/classify":
            self.handle_api_classify()
        elif path == "/api/evaluate":
            self.handle_api_evaluate()
        else:
            self.send_error_json("Endpoint not found", 404)

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.is_file():
            self.send_error_json("File not found", 404)
            return
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_api_status(self):
        uploads = []
        for p in sorted(UPLOADS_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                w, h, _ = load_image_dimensions_and_rgb(p)
                uploads.append({
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": compute_file_sha256(p),
                    "width": w,
                    "height": h,
                })

        datasets = self.get_datasets_list()
        models = self.get_models_list()

        self.send_json({
            "success": True,
            "runtime_root": str(RUNTIME_ROOT),
            "has_pil": HAS_PIL,
            "uploads": uploads,
            "datasets": datasets,
            "models": models,
        })

    def handle_api_image(self, query):
        req_path = query.get("path", [""])[0]
        if not req_path:
            self.send_error_json("Missing path", 400)
            return
        p = Path(req_path)
        if not is_safe_path(p):
            self.send_error_json("Forbidden path", 403)
            return
        if not p.is_file():
            self.send_error_json("Image not found", 404)
            return
        ext = p.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        self.serve_file(p, mime)

    def handle_api_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 or content_length > 100 * 1024 * 1024:
            self.send_error_json("Invalid upload size", 400)
            return

        body = self.rfile.read(content_length)

        # Handle multipart/form-data
        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[-1].strip()
            parts = body.split(("--" + boundary).encode("utf-8"))
            file_data = None
            filename = "upload.png"
            for part in parts:
                if b'filename="' in part:
                    header_part, _, data_part = part.partition(b"\r\n\r\n")
                    header_str = header_part.decode("utf-8", errors="ignore")
                    m = re.search(r'filename="([^"]+)"', header_str)
                    if m:
                        filename = os.path.basename(m.group(1))
                    if data_part.endswith(b"\r\n"):
                        data_part = data_part[:-2]
                    file_data = data_part
                    break
            if not file_data:
                self.send_error_json("No file found in multipart upload", 400)
                return
        else:
            file_data = body
            filename = "upload.png"

        ext = Path(filename).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg"):
            ext = ".png"

        sha = compute_sha256(file_data)
        safe_base = sanitize_name(Path(filename).stem)
        saved_filename = f"{sha[:12]}_{safe_base}{ext}"
        target_path = UPLOADS_DIR / saved_filename

        with open(target_path, "wb") as f:
            f.write(file_data)

        w, h, _ = load_image_dimensions_and_rgb(target_path)

        self.send_json({
            "success": True,
            "filename": saved_filename,
            "path": str(target_path),
            "sha256": sha,
            "width": w,
            "height": h,
            "size_bytes": len(file_data),
        })

    def handle_api_create_dataset(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
            return

        dataset_name = sanitize_name(req.get("dataset_name", "dataset"))
        image_path = Path(req.get("image_path", ""))
        is_sentinel_10m = bool(req.get("is_sentinel_10m", False))
        patches = req.get("patches", [])

        if not is_safe_path(image_path) or not image_path.is_file():
            self.send_error_json("Invalid source image path", 400)
            return

        if not patches:
            self.send_error_json("No patches to extract", 400)
            return

        w, h, rgb_bytes = load_image_dimensions_and_rgb(image_path)
        if rgb_bytes is None:
            self.send_error_json("Failed to decode source image pixels", 400)
            return

        source_sha256 = compute_file_sha256(image_path)
        dataset_dir = DATASETS_DIR / dataset_name
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)

        manifest_rows = []
        counts = {"train": {}, "dev": {}, "probe": {}}
        patch_records = []
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        patch_index = 0
        for p in patches:
            cls_name = sanitize_name(p.get("class", "unclassified"))
            split = p.get("split", "train").lower()
            if split not in ("train", "dev", "probe"):
                split = "train"

            x = int(p.get("x", 0))
            y = int(p.get("y", 0))
            roi_id = str(p.get("roi_id", "roi_0"))

            if x < 0 or y < 0 or x + 8 > w or y + 8 > h:
                continue

            # Extract 8x8 pixels
            patch_rgb = bytearray()
            for py in range(y, y + 8):
                row_start = (py * w + x) * 3
                patch_rgb.extend(rgb_bytes[row_start : row_start + 8 * 3])

            patch_bytes = bytes(patch_rgb)
            patch_sha = compute_sha256(patch_bytes)
            patch_filename = f"patch_{patch_index:05d}_{patch_sha[:8]}.png"
            patch_out_path = dataset_dir / split / cls_name / patch_filename

            save_png_patch(patch_bytes, 8, 8, patch_out_path)

            counts[split][cls_name] = counts[split].get(cls_name, 0) + 1
            patch_index += 1

            manifest_rows.append({
                "dataset_id": dataset_name,
                "source_image": str(image_path.name),
                "source_sha256": source_sha256,
                "source_width": w,
                "source_height": h,
                "is_sentinel_10m": is_sentinel_10m,
                "resolution_note": "1px=10m (80x80m patch)" if is_sentinel_10m else "DISPLAY ONLY — no 10m guarantee",
                "class": cls_name,
                "split": split,
                "roi_id": roi_id,
                "patch_x": x,
                "patch_y": y,
                "patch_width": 8,
                "patch_height": 8,
                "patch_filename": patch_filename,
                "patch_sha256": patch_sha,
                "timestamp": timestamp,
            })

            patch_records.append({
                "filename": patch_filename,
                "class": cls_name,
                "split": split,
                "roi_id": roi_id,
                "x": x,
                "y": y,
                "sha256": patch_sha,
            })

        # Write manifest.csv
        manifest_path = dataset_dir / "manifest.csv"
        fieldnames = [
            "dataset_id", "source_image", "source_sha256", "source_width", "source_height",
            "is_sentinel_10m", "resolution_note", "class", "split", "roi_id",
            "patch_x", "patch_y", "patch_width", "patch_height", "patch_filename",
            "patch_sha256", "timestamp"
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        # Write dataset.json
        meta = {
            "dataset_id": dataset_name,
            "created_at": timestamp,
            "source_image": {
                "name": image_path.name,
                "path": str(image_path),
                "sha256": source_sha256,
                "width": w,
                "height": h,
            },
            "is_sentinel_10m": is_sentinel_10m,
            "total_patches": patch_index,
            "counts": counts,
            "patches": patch_records,
        }
        with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save copy to manifests/
        with open(MANIFESTS_DIR / f"{dataset_name}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.send_json({
            "success": True,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "total_patches": patch_index,
            "counts": counts,
            "manifest_csv": str(manifest_path),
            "dataset_json": str(dataset_dir / "dataset.json"),
        })

    def handle_api_train(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
            return

        dataset_name = sanitize_name(req.get("dataset_name", ""))
        model_name = sanitize_name(req.get("model_name", f"{dataset_name}_model"))
        if not model_name.endswith(".tlv"):
            model_filename = f"{model_name}.tlv"
        else:
            model_filename = model_name

        dataset_dir = DATASETS_DIR / dataset_name
        if not dataset_dir.is_dir() or not (dataset_dir / "train").is_dir():
            self.send_error_json(f"Dataset '{dataset_name}' not found or missing train split", 400)
            return

        model_path = MODELS_DIR / model_filename
        cli_bin = REPO_ROOT / "bin" / "tinyvision"

        run_id = f"run_{int(time.time())}_{dataset_name}"
        run_log_path = RUNS_DIR / f"{run_id}.log"

        cmd = [str(cli_bin), "train", str(dataset_dir), str(model_path)]

        # Execute training subprocess synchronously or background
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_content = proc.stdout
        with open(run_log_path, "w", encoding="utf-8") as f:
            f.write(log_content)

        success = (proc.returncode == 0)

        # Parse training output
        parsed_metrics = {
            "success": success,
            "returncode": proc.returncode,
            "model_path": str(model_path) if success else None,
            "log": log_content,
            "epochs": [],
            "architecture": None,
            "parameter_count": None,
            "final_train_loss": None,
            "final_train_acc": None,
            "final_dev_loss": None,
            "final_dev_acc": None,
        }

        for line in log_content.splitlines():
            if line.startswith("architecture="):
                parsed_metrics["architecture"] = line.split("=", 1)[-1]
            elif line.startswith("parameter_count="):
                parsed_metrics["parameter_count"] = int(line.split("=", 1)[-1])
            elif line.startswith("final_train_loss="):
                parsed_metrics["final_train_loss"] = float(line.split("=", 1)[-1])
            elif line.startswith("final_train_accuracy="):
                parsed_metrics["final_train_acc"] = line.split("=", 1)[-1]
            elif line.startswith("final_dev_loss="):
                parsed_metrics["final_dev_loss"] = float(line.split("=", 1)[-1])
            elif line.startswith("final_dev_accuracy="):
                parsed_metrics["final_dev_acc"] = line.split("=", 1)[-1]
            elif line.startswith("epoch="):
                # parse milestone: epoch= 30 train_acc=100.00% dev_acc=100.00%
                m = re.search(r"epoch=\s*(\d+)\s+train_acc=([0-9.]+)%\s+dev_acc=([0-9.]+)%", line)
                if m:
                    parsed_metrics["epochs"].append({
                        "epoch": int(m.group(1)),
                        "train_acc": float(m.group(2)),
                        "dev_acc": float(m.group(3)),
                    })

        self.send_json(parsed_metrics, status=200 if success else 400)

    def handle_api_classify(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
            return

        model_name = req.get("model_name", "")
        if not model_name.endswith(".tlv"):
            model_path = MODELS_DIR / f"{sanitize_name(model_name)}.tlv"
        else:
            model_path = MODELS_DIR / model_name

        image_path = Path(req.get("image_path", ""))

        if not model_path.is_file():
            self.send_error_json(f"Model file not found: {model_path.name}", 400)
            return

        if not is_safe_path(image_path) or not image_path.is_file():
            self.send_error_json("Image file not found or path forbidden", 400)
            return

        cli_bin = REPO_ROOT / "bin" / "tinyvision"
        cmd = [str(cli_bin), "classify", str(model_path), str(image_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            self.send_error_json(f"Classification failed: {proc.stderr or proc.stdout}", 400)
            return

        predicted = None
        score = None
        probabilities = {}
        for line in proc.stdout.splitlines():
            if line.startswith("predicted="):
                predicted = line.split("=", 1)[-1].strip()
            elif line.startswith("score="):
                score = float(line.split("=", 1)[-1].strip())
            elif line.startswith("probability[") and "]=" in line:
                cls_part = line[len("probability[") : line.index("]=")]
                val = float(line.split("]=", 1)[-1].strip())
                probabilities[cls_part] = val

        self.send_json({
            "success": True,
            "predicted": predicted,
            "score": score,
            "probabilities": probabilities,
            "raw_output": proc.stdout,
        })

    def handle_api_evaluate(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
            return

        model_name = req.get("model_name", "")
        if not model_name.endswith(".tlv"):
            model_path = MODELS_DIR / f"{sanitize_name(model_name)}.tlv"
        else:
            model_path = MODELS_DIR / model_name

        dataset_name = sanitize_name(req.get("dataset_name", ""))
        split = req.get("split", "dev").lower()
        if split not in ("dev", "probe"):
            self.send_error_json("Split must be 'dev' or 'probe'", 400)
            return

        split_dir = DATASETS_DIR / dataset_name / split
        if not split_dir.is_dir():
            self.send_error_json(f"Split directory '{split_dir}' not found", 400)
            return

        if not model_path.is_file():
            self.send_error_json(f"Model file '{model_path.name}' not found", 400)
            return

        cli_bin = REPO_ROOT / "bin" / "tinyvision"
        cmd = [str(cli_bin), "evaluate", str(model_path), str(split_dir)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            self.send_error_json(f"Evaluation failed: {proc.stderr or proc.stdout}", 400)
            return

        lines = proc.stdout.splitlines()
        samples = 0
        loss = 0.0
        accuracy = 0.0
        mean_prob = 0.0
        confusion = {"classes": [], "matrix": []}

        in_confusion = False
        for line in lines:
            if line.startswith("samples="):
                samples = int(line.split("=", 1)[-1])
            elif line.startswith("loss="):
                loss = float(line.split("=", 1)[-1])
            elif line.startswith("accuracy="):
                accuracy = float(line.split("=", 1)[-1].replace("%", ""))
            elif line.startswith("mean_true_probability="):
                mean_prob = float(line.split("=", 1)[-1])
            elif "confusion rows=actual cols=predicted" in line:
                in_confusion = True
            elif in_confusion:
                parts = line.split()
                if not parts:
                    continue
                if not confusion["classes"]:
                    # header row
                    confusion["classes"] = parts
                else:
                    row_name = parts[0]
                    row_counts = [int(x) for x in parts[1:]]
                    confusion["matrix"].append({"actual": row_name, "counts": row_counts})

        self.send_json({
            "success": True,
            "split": split,
            "samples": samples,
            "loss": loss,
            "accuracy": accuracy,
            "mean_true_probability": mean_prob,
            "confusion": confusion,
            "raw_output": proc.stdout,
        })

    def get_datasets_list(self):
        datasets = []
        for d in sorted(DATASETS_DIR.glob("*")):
            if d.is_dir() and (d / "dataset.json").is_file():
                try:
                    with open(d / "dataset.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    datasets.append(meta)
                except Exception:
                    pass
        return datasets

    def get_models_list(self):
        models = []
        for m in sorted(MODELS_DIR.glob("*.tlv")):
            if m.is_file():
                models.append({
                    "name": m.name,
                    "path": str(m),
                    "size_bytes": m.stat().st_size,
                    "updated_at": datetime.datetime.fromtimestamp(
                        m.stat().st_mtime, datetime.timezone.utc
                    ).isoformat(),
                })
        return models

    def handle_api_list_datasets(self):
        self.send_json({"success": True, "datasets": self.get_datasets_list()})

    def handle_api_list_models(self):
        self.send_json({"success": True, "models": self.get_models_list()})


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="TinyLogicVision Local Web GUI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (strictly 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    # Enforce loopback binding only
    if args.host not in ("127.0.0.1", "localhost"):
        print("Error: TinyLogicVision Web GUI is strictly local and binds only to 127.0.0.1", file=sys.stderr)
        sys.exit(1)

    host = "127.0.0.1"
    port = args.port

    server_address = (host, port)
    try:
        httpd = ThreadedHTTPServer(server_address, TinyVisionRequestHandler)
    except OSError as e:
        # Try next port if busy
        print(f"Port {port} busy, trying {port + 1}...")
        port = port + 1
        server_address = (host, port)
        httpd = ThreadedHTTPServer(server_address, TinyVisionRequestHandler)

    url = f"http://{host}:{port}/"
    print("=" * 60)
    print("  TinyLogicVision Local Web GUI v0.1")
    print(f"  Running locally at: {url}")
    print("  Workspace root:     .tinyvision/")
    print("  Bound interface:    127.0.0.1 (Strictly local, no external network)")
    print("=" * 60)
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down TinyLogicVision web server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
