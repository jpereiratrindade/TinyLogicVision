#!/usr/bin/env python3
"""
TinyLogicVision Local Web Application Server v0.1
Orchestrates patch extraction, dataset authoring, model training,
classification, and evaluation using the TinyLogicVision CLI authority.
"""

import uuid
import argparse
import csv
import datetime
import hashlib
import http.server
import json
import os
import re
import socketserver
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

# Optional Pillow and GDAL support
try:
    from PIL import Image as PILImage
    PILImage.MAX_IMAGE_PIXELS = None  # Allow Sentinel-2 full granules without DecompressionBombError
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_BIN = REPO_ROOT / "bin" / "tinyvision"
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
JOBS_LOCK = threading.Lock()
JOBS = {}
APPROVED_PATHS_LOCK = threading.Lock()
APPROVED_LOCAL_PATHS = set()

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jp2", ".j2k", ".tif", ".tiff"}


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


def get_image_dimensions(image_path: Path):
    """Fast extraction of image dimensions without loading all pixel data into memory."""
    if HAS_GDAL:
        try:
            ds = gdal.Open(str(image_path))
            if ds:
                return ds.RasterXSize, ds.RasterYSize
        except Exception:
            pass

    if HAS_PIL:
        try:
            with PILImage.open(image_path) as img:
                return img.size[0], img.size[1]
        except Exception:
            pass

    try:
        with open(image_path, "rb") as f:
            header = f.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                return struct.unpack(">II", header[16:24])
    except Exception:
        pass

    return 0, 0


def get_gdal_raster_signature(image_path: Path):
    """Return the alignment-critical raster metadata without inventing GEO values."""
    if not HAS_GDAL:
        return None
    dataset = gdal.Open(str(image_path))
    if dataset is None:
        return None
    projection = dataset.GetProjection() or None
    geotransform = dataset.GetGeoTransform(can_return_null=True)
    spatial_ref = dataset.GetSpatialRef()
    authority_name = spatial_ref.GetAuthorityName(None) if spatial_ref else None
    authority_code = spatial_ref.GetAuthorityCode(None) if spatial_ref else None
    crs_label = f"{authority_name}:{authority_code}" if authority_name and authority_code else projection
    signature = {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "projection": projection,
        "geotransform": tuple(geotransform) if geotransform is not None else None,
        "crs_label": crs_label,
    }
    dataset = None
    return signature


def read_band_2d(image_path: Path, max_dim: int = 2048):
    """Reads 2D single-band float32 raster with optional downsampled preview."""
    import numpy as np

    if HAS_GDAL:
        try:
            ds = gdal.Open(str(image_path))
            if ds is not None:
                w, h = ds.RasterXSize, ds.RasterYSize
                if max_dim and max(w, h) > max_dim:
                    scale = max_dim / float(max(w, h))
                    pw = max(1, int(w * scale))
                    ph = max(1, int(h * scale))
                else:
                    pw, ph = w, h
                band = ds.GetRasterBand(1)
                arr = band.ReadAsArray(buf_xsize=pw, buf_ysize=ph)
                if arr is not None:
                    return w, h, arr.astype(np.float32), pw, ph
        except Exception:
            pass

    if HAS_PIL:
        try:
            with PILImage.open(image_path) as img:
                w, h = img.size
                if max_dim and max(w, h) > max_dim:
                    scale = max_dim / float(max(w, h))
                    pw = max(1, int(w * scale))
                    ph = max(1, int(h * scale))
                    img_res = img.resize((pw, ph), PILImage.BILINEAR)
                    return w, h, np.array(img_res, dtype=np.float32), pw, ph
                else:
                    return w, h, np.array(img, dtype=np.float32), w, h
        except Exception:
            pass

    return 0, 0, None, 0, 0


def save_tvp_patch(values_float: list or bytes, width: int, height: int, channels: int, output_path: Path):
    """Save multichannel patch as TVP (TinyLogicVision Patch) binary format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import struct
    import numpy as np

    hdr = struct.pack("<8sIHHHH", b"TLV_PAT\x00", 1, width, height, channels, 1) # dtype 1 = float32
    if isinstance(values_float, (list, tuple)):
        raw_floats = np.array(values_float, dtype=np.float32).tobytes()
    elif isinstance(values_float, np.ndarray):
        raw_floats = values_float.astype(np.float32).tobytes()
    else:
        raw_floats = bytes(values_float)

    with open(output_path, "wb") as f:
        f.write(hdr)
        f.write(raw_floats)


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
    """Returns (width, height, raw_rgb_bytes) with support for PNG, JPEG, JP2 (JPEG2000), and GeoTIFF."""
    w, h, arr, pw, ph = read_band_2d(image_path, max_dim=None)
    if arr is not None:
        import numpy as np
        if arr.ndim == 2:
            val_max = float(np.percentile(arr, 98)) if arr.size > 0 else 1.0
            scale = 255.0 / max(val_max, 2500.0) if val_max > 255 else 1.0
            arr_8u = np.clip(arr * scale, 0, 255).astype(np.uint8)
            rgb_arr = np.dstack([arr_8u, arr_8u, arr_8u])
            return w, h, rgb_arr.tobytes()

    if HAS_PIL:
        try:
            with PILImage.open(image_path) as img:
                w, h = img.size
                if img.mode == "RGB":
                    return w, h, img.tobytes()
                elif img.mode in ("I;16", "I", "L", "F", "1"):
                    import numpy as np
                    arr = np.array(img, dtype=np.float32)
                    val_max = float(arr.max()) if arr.size > 0 else 1.0
                    scale = 255.0 / max(val_max, 3500.0) if val_max > 255 else 1.0
                    arr_8u = np.clip(arr * scale, 0, 255).astype(np.uint8)
                    rgb_arr = np.dstack([arr_8u, arr_8u, arr_8u])
                    return w, h, rgb_arr.tobytes()
                else:
                    rgb_img = img.convert("RGB")
                    return w, h, rgb_img.tobytes()
        except Exception:
            pass

    # Fallback to loading via Python's standard libraries or basic headers
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            import struct
            w, h = struct.unpack(">II", data[16:24])
            return w, h, None
    except Exception:
        pass

    return 0, 0, None


def approve_local_path(target_path: Path):
    with APPROVED_PATHS_LOCK:
        APPROVED_LOCAL_PATHS.add(target_path.resolve())


def is_safe_path(target_path: Path, allowed_roots=None) -> bool:
    try:
        resolved = target_path.resolve()
        if allowed_roots is None:
            allowed_roots = [RUNTIME_ROOT, REPO_ROOT]
            with APPROVED_PATHS_LOCK:
                allowed_roots.extend(APPROVED_LOCAL_PATHS)
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
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

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
            rest = path[len("/api/runs/") :]
            parts = rest.split("/", 1)
            run_id = sanitize_name(parts[0])
            if len(parts) > 1 and parts[1]:
                sub = parts[1]
                if sub.startswith("inspect"):
                    self.handle_api_run_inspect(run_id, query)
                else:
                    self.handle_api_run_artifact(run_id, sub)
            else:
                self.handle_api_run_status(run_id)
        elif path == "/api/datasets":
            self.handle_api_list_datasets()
        elif path == "/api/models":
            self.handle_api_list_models()
        elif path == "/api/jobs":
            self.handle_api_list_jobs()
        elif path.startswith("/api/jobs/"):
            job_id = sanitize_name(path[len("/api/jobs/") :])
            self.handle_api_get_job(job_id)
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
        elif path == "/api/dense_map":
            self.handle_api_dense_map()
        elif path == "/api/evaluate":
            self.handle_api_evaluate()
        elif path == "/api/jobs":
            self.handle_api_create_job()
        elif path in ("/api/sentinel/open", "/api/source/sentinel_open"):
            self.handle_api_sentinel_open()
        elif path == "/api/source/open_local":
            self.handle_api_source_open_local()
        elif path == "/api/source/browse_local":
            self.handle_api_source_browse_local()
        else:
            self.send_error_json("Endpoint not found", 404)

    def serve_file(self, file_path: Path, content_type: str, download_filename: str = None):
        if not file_path.is_file():
            self.send_error_json("File not found", 404)
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            if download_filename:
                self.send_header("Content-Disposition", f'attachment; filename="{download_filename}"')
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def handle_api_status(self):
        uploads = []
        for p in sorted(UPLOADS_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".jp2", ".j2k", ".tif", ".tiff"):
                w, h = get_image_dimensions(p)
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
            "has_gdal": HAS_GDAL,
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
        if ext == ".png":
            self.serve_file(p, "image/png")
        elif ext in (".jpg", ".jpeg"):
            self.serve_file(p, "image/jpeg")
        elif ext in (".jp2", ".j2k", ".tif", ".tiff"):
            # Convert single band or GeoTIFF/JP2 to preview PNG
            w, h, arr, pw, ph = read_band_2d(p, max_dim=2048)
            if arr is not None:
                import numpy as np
                import io
                val_max = float(np.percentile(arr, 98)) if arr.size > 0 else 1.0
                scale = 255.0 / max(val_max, 2500.0) if val_max > 255 else 1.0
                arr_8u = np.clip(arr * scale, 0, 255).astype(np.uint8)
                rgb_arr = np.dstack([arr_8u, arr_8u, arr_8u])
                if HAS_PIL:
                    out_img = PILImage.fromarray(rgb_arr)
                    buf = io.BytesIO()
                    out_img.save(buf, format="PNG")
                    png_bytes = buf.getvalue()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png_bytes)))
                    self.end_headers()
                    self.wfile.write(png_bytes)
                    return
            self.serve_file(p, "application/octet-stream")
        else:
            self.send_error_json("Unsupported image type", 415)

    def handle_api_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        # Support up to 2GB uploads for large Sentinel granules
        if content_length <= 0 or content_length > 2 * 1024 * 1024 * 1024:
            self.send_error_json("Invalid upload size (max 2GB)", 400)
            return

        body = self.rfile.read(content_length)

        # Handle multipart/form-data
        if "multipart/form-data" in content_type:
            boundary = ""
            for part_hdr in content_type.split(";"):
                part_hdr = part_hdr.strip()
                if part_hdr.lower().startswith("boundary="):
                    boundary = part_hdr[len("boundary="):].strip().strip('"')
                    break
            if not boundary:
                boundary = content_type.split("boundary=")[-1].strip().strip('"')

            parts = body.split(("--" + boundary).encode("utf-8"))
            saved_files = []
            for part in parts:
                if b'filename="' in part:
                    header_part, _, data_part = part.partition(b"\r\n\r\n")
                    header_str = header_part.decode("utf-8", errors="ignore")
                    m = re.search(r'filename="([^"]+)"', header_str)
                    if m:
                        orig_fn = os.path.basename(m.group(1))
                        if data_part.endswith(b"\r\n"):
                            data_part = data_part[:-2]
                        if len(data_part) > 0:
                            sha = compute_sha256(data_part)
                            ext = Path(orig_fn).suffix.lower()
                            if ext not in (".png", ".jpg", ".jpeg", ".jp2", ".j2k", ".tif", ".tiff"):
                                ext = ".png"
                            safe_base = sanitize_name(Path(orig_fn).stem)
                            saved_filename = f"{safe_base}_{sha[:8]}{ext}"
                            target_path = UPLOADS_DIR / saved_filename
                            with open(target_path, "wb") as f:
                                f.write(data_part)
                            w, h = get_image_dimensions(target_path)
                            saved_files.append({
                                "filename": saved_filename,
                                "original_name": orig_fn,
                                "path": str(target_path),
                                "sha256": sha,
                                "width": w,
                                "height": h,
                                "size_bytes": len(data_part),
                            })
            if not saved_files:
                self.send_error_json("No files found in multipart upload", 400)
                return

            first = saved_files[0]
            self.send_json({
                "success": True,
                "filename": first["filename"],
                "path": first["path"],
                "sha256": first["sha256"],
                "width": first["width"],
                "height": first["height"],
                "size_bytes": first["size_bytes"],
                "files": saved_files,
            })
            return
        else:
            file_data = body
            filename = "upload.png"
            ext = Path(filename).suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".jp2", ".j2k", ".tif", ".tiff"):
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

        # Enforce ROI-level split integrity: 1 ROI -> 1 Split
        roi_to_split = {}
        for p in patches:
            roi_id = str(p.get("roi_id", "roi_0"))
            split_val = str(p.get("split", "train")).lower()
            if roi_id in roi_to_split and roi_to_split[roi_id] != split_val:
                self.send_error_json(
                    f"Violação de integridade de split por ROI: ROI '{roi_id}' possui patches distribuídos em múltiplos splits ('{roi_to_split[roi_id]}' e '{split_val}'). A regra científica exige 1 ROI = 1 Split.",
                    400,
                )
                return
            roi_to_split[roi_id] = split_val

        # Enforce spatial patch disjointness across different splits
        for i in range(len(patches)):
            p1 = patches[i]
            s1 = str(p1.get("split", "train")).lower()
            x1 = int(p1.get("x", 0))
            y1 = int(p1.get("y", 0))
            for j in range(i + 1, len(patches)):
                p2 = patches[j]
                s2 = str(p2.get("split", "train")).lower()
                if s1 == s2:
                    continue
                x2 = int(p2.get("x", 0))
                y2 = int(p2.get("y", 0))
                if abs(x1 - x2) < 8 and abs(y1 - y2) < 8:
                    self.send_error_json(
                        f"Violação de integridade espacial: patch em split '{s1}' em ({x1}, {y1}) sobrepõe patch em split '{s2}' em ({x2}, {y2}). Patches pertencentes a splits diferentes não podem sobrepor espacialmente.",
                        400,
                    )
                    return

        manifest_rows = []
        counts = {"train": {}, "dev": {}, "probe": {}}
        patch_records = []
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        bands_input = req.get("bands", {}) or {}
        normalized_bands = {}
        if isinstance(bands_input, dict):
            for b_key in ("b2", "b3", "b4", "b8"):
                val = bands_input.get(b_key)
                if isinstance(val, dict):
                    normalized_bands[b_key] = val.get("path", "")
                elif isinstance(val, str):
                    normalized_bands[b_key] = val
                else:
                    normalized_bands[b_key] = ""

        has_sentinel_bands = is_sentinel_10m and bool(normalized_bands) and all(Path(normalized_bands.get(b, "")).is_file() for b in ("b2", "b3", "b4", "b8"))

        preview_w = float(req.get("preview_width", w) or w)
        preview_h = float(req.get("preview_height", h) or h)
        native_w = float(req.get("native_width", w) or w)
        native_h = float(req.get("native_height", h) or h)

        scale_x = native_w / max(preview_w, 1.0)
        scale_y = native_h / max(preview_h, 1.0)

        ds_bands = {}
        if has_sentinel_bands and HAS_GDAL:
            try:
                for b_key in ("b2", "b3", "b4", "b8"):
                    ds_bands[b_key] = gdal.Open(str(Path(normalized_bands[b_key])))
            except Exception:
                ds_bands = {}

        import numpy as np
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

            split_dir = dataset_dir / split / cls_name
            split_dir.mkdir(parents=True, exist_ok=True)

            if has_sentinel_bands and ds_bands:
                # 1. Reproject preview ROI coordinates to native 10m raster coordinates
                nat_x = min(max(0, int(round(x * scale_x))), int(native_w) - 8)
                nat_y = min(max(0, int(round(y * scale_y))), int(native_h) - 8)

                # 2. Extract exact 8x8 pixels from native 10m Sentinel-2 bands
                b2_arr = ds_bands["b2"].GetRasterBand(1).ReadAsArray(nat_x, nat_y, 8, 8).astype(np.float32) * 0.0001
                b3_arr = ds_bands["b3"].GetRasterBand(1).ReadAsArray(nat_x, nat_y, 8, 8).astype(np.float32) * 0.0001
                b4_arr = ds_bands["b4"].GetRasterBand(1).ReadAsArray(nat_x, nat_y, 8, 8).astype(np.float32) * 0.0001
                b8_arr = ds_bands["b8"].GetRasterBand(1).ReadAsArray(nat_x, nat_y, 8, 8).astype(np.float32) * 0.0001

                # 3. Interleaved 256 float32 values (8x8x4)
                multichannel_8x8x4 = np.dstack([b2_arr, b3_arr, b4_arr, b8_arr]).astype(np.float32)
                patch_raw_bytes = multichannel_8x8x4.tobytes()
                patch_sha = compute_sha256(patch_raw_bytes)
                patch_base = f"patch_{patch_index:05d}_{patch_sha[:8]}"

                # Save .tvp (multichannel 256 inputs binary patch)
                tvp_out_path = split_dir / f"{patch_base}.tvp"
                save_tvp_patch(patch_raw_bytes, 8, 8, 4, tvp_out_path)

                # Save .png (8x8 True Color composite for preview/inspection)
                r_8u = np.clip(b4_arr * 2500.0, 0, 255).astype(np.uint8)
                g_8u = np.clip(b3_arr * 2500.0, 0, 255).astype(np.uint8)
                b_8u = np.clip(b2_arr * 2500.0, 0, 255).astype(np.uint8)
                rgb_composite = np.dstack([r_8u, g_8u, b_8u]).tobytes()
                png_out_path = split_dir / f"{patch_base}.png"
                save_png_patch(rgb_composite, 8, 8, png_out_path)

                patch_filename = f"{patch_base}.tvp"
                patch_nat_x = nat_x
                patch_nat_y = nat_y
            else:
                # Extract 8x8 pixels RGB
                patch_rgb = bytearray()
                for py in range(y, y + 8):
                    row_start = (py * w + x) * 3
                    patch_rgb.extend(rgb_bytes[row_start : row_start + 8 * 3])

                patch_bytes = bytes(patch_rgb)
                patch_sha = compute_sha256(patch_bytes)
                patch_filename = f"patch_{patch_index:05d}_{patch_sha[:8]}.png"
                patch_out_path = split_dir / patch_filename

                save_png_patch(patch_bytes, 8, 8, patch_out_path)
                patch_nat_x = x
                patch_nat_y = y

            counts[split][cls_name] = counts[split].get(cls_name, 0) + 1
            patch_index += 1

            manifest_rows.append({
                "dataset_id": dataset_name,
                "source_image": str(image_path.name),
                "source_sha256": source_sha256,
                "source_width": int(native_w) if has_sentinel_bands else w,
                "source_height": int(native_h) if has_sentinel_bands else h,
                "is_sentinel_10m": is_sentinel_10m,
                "modality": "SENTINEL2_MULTIBAND (256 inputs)" if has_sentinel_bands else "RGB (192 inputs)",
                "resolution_note": "1px=10m (80x80m native patch)" if has_sentinel_bands else "DISPLAY ONLY — no 10m guarantee",
                "class": cls_name,
                "split": split,
                "roi_id": roi_id,
                "patch_x": patch_nat_x,
                "patch_y": patch_nat_y,
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
                "x": patch_nat_x,
                "y": patch_nat_y,
                "sha256": patch_sha,
            })

        # Write manifest.csv
        manifest_path = dataset_dir / "manifest.csv"
        fieldnames = [
            "dataset_id", "source_image", "source_sha256", "source_width", "source_height",
            "is_sentinel_10m", "modality", "resolution_note", "class", "split", "roi_id",
            "patch_x", "patch_y", "patch_width", "patch_height", "patch_filename",
            "patch_sha256", "timestamp"
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
        if not dataset_dir.is_dir():
            self.send_error_json(f"Dataset '{dataset_name}' não encontrado em .tinyvision/datasets/", 400)
            return

        if not (dataset_dir / "train").is_dir():
            self.send_error_json(f"Dataset '{dataset_name}' não possui o split 'train/'.", 400)
            return

        if not (dataset_dir / "dev").is_dir():
            self.send_error_json(f"Dataset '{dataset_name}' não possui o split 'dev/'. O treinamento do TinyLogicVision exige amostras em 'train/' e 'dev/' para monitorar a generalização.", 400)
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

        model_name = sanitize_name(Path(str(req.get("model_name", ""))).stem)
        model_path = MODELS_DIR / f"{model_name}.tlv"

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

        model_name = sanitize_name(Path(str(req.get("model_name", ""))).stem)
        model_path = MODELS_DIR / f"{model_name}.tlv"

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
                schema = None
                try:
                    with open(m, "r", encoding="utf-8") as model_file:
                        magic_line = model_file.readline().split()
                        header = model_file.readline().split()
                    if len(magic_line) == 2 and magic_line[0] == "TINYLOGICVISION_APP_MODEL":
                        version = int(magic_line[1])
                        if version == 1 and len(header) >= 5:
                            schema = {"width": int(header[0]), "height": int(header[1]), "channels": 3, "modality": "RGB"}
                        elif version == 2 and len(header) >= 7:
                            schema = {"width": int(header[0]), "height": int(header[1]), "channels": int(header[2]), "modality": header[3]}
                        if schema:
                            schema["input_size"] = schema["width"] * schema["height"] * schema["channels"]
                except (OSError, ValueError):
                    schema = None
                models.append({
                    "name": m.name,
                    "path": str(m),
                    "size_bytes": m.stat().st_size,
                    "updated_at": datetime.datetime.fromtimestamp(
                        m.stat().st_mtime, datetime.timezone.utc
                    ).isoformat(),
                    "schema": schema,
                })
        return models

    def handle_api_dense_map(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {e}", 400)
            return

        model_name = sanitize_name(Path(str(req.get("model_name", ""))).stem)
        model_path = MODELS_DIR / f"{model_name}.tlv"

        image_path = Path(req.get("image_path", ""))
        stride = int(req.get("stride", 1))
        confidence = float(req.get("confidence", 0.0))
        margin = float(req.get("margin", 0.0))
        threads = int(req.get("threads", 0))
        is_sentinel = bool(req.get("is_sentinel_10m", False))
        bands = req.get("bands") or {}

        if not model_path.is_file():
            self.send_error_json(f"Model file not found: {model_path.name}", 400)
            return

        if not is_safe_path(image_path) or not image_path.is_file():
            self.send_error_json("Image file not found or path forbidden", 400)
            return

        if stride not in (1, 2, 4, 8):
            self.send_error_json(f"Invalid stride {stride}, must be 1, 2, 4, or 8", 400)
            return

        run_id = f"map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_dir = RUNS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cli_bin = REPO_ROOT / "bin" / "tinyvision"
        cmd = [
            str(cli_bin), "map",
            str(model_path),
            str(image_path),
            str(out_dir),
            "--stride", str(stride),
            "--confidence", str(confidence),
            "--margin", str(margin),
        ]
        if threads > 0:
            cmd.extend(["--threads", str(threads)])
        if bands:
            normalized_bands = {}
            for band_name in ("b2", "b3", "b4", "b8"):
                band_value = bands.get(band_name)
                if isinstance(band_value, dict):
                    band_value = band_value.get("path", "")
                band_path = Path(str(band_value or ""))
                if not is_safe_path(band_path) or not band_path.is_file():
                    self.send_error_json(f"Sentinel band {band_name.upper()} not found or forbidden", 400)
                    return
                normalized_bands[band_name] = band_path
            cmd.extend([
                "--band-b2", str(normalized_bands["b2"]),
                "--band-b3", str(normalized_bands["b3"]),
                "--band-b4", str(normalized_bands["b4"]),
                "--band-b8", str(normalized_bands["b8"]),
            ])
            is_sentinel = True
        if is_sentinel:
            cmd.append("--sentinel-10m")

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            self.send_error_json(f"Dense map classification failed: {proc.stderr or proc.stdout}", 400)
            return

        run_json_path = out_dir / "run.json"
        metadata = {}
        if run_json_path.is_file():
            try:
                with open(run_json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        self.send_json({
            "success": True,
            "run_id": run_id,
            "metadata": metadata,
            "artifacts": {
                "class_map": f"/api/runs/{run_id}/class_map.png",
                "confidence": f"/api/runs/{run_id}/confidence.png",
                "margin": f"/api/runs/{run_id}/margin.png",
                "overlay": f"/api/runs/{run_id}/overlay.png",
                "run_json": f"/api/runs/{run_id}/run.json",
                "classification_csv": f"/api/runs/{run_id}/classification.csv",
            },
            "raw_output": proc.stdout,
        })

    def handle_api_run_artifact(self, run_id: str, filename: str):
        allowed = {
            "class_map.png": "image/png",
            "confidence.png": "image/png",
            "margin.png": "image/png",
            "overlay.png": "image/png",
            "run.json": "application/json",
            "classification.csv": "text/csv; charset=utf-8",
            "class_map.tif": "image/tiff",
            "confidence.tif": "image/tiff",
            "provenance.json": "application/json",
            "decisions.bin": "application/octet-stream",
            "classification_h3.csv": "text/csv; charset=utf-8",
        }
        if filename not in allowed:
            self.send_error_json("Artifact not allowed or not found", 404)
            return
        target_file = RUNS_DIR / sanitize_name(run_id) / filename
        if not target_file.is_file():
            self.send_error_json("File not found", 404)
            return
        self.serve_file(target_file, allowed[filename])

    def handle_api_run_status(self, run_id: str):
        clean_id = sanitize_name(run_id)
        run_file = RUNS_DIR / clean_id / "run.json"
        if not run_file.is_file():
            self.send_error_json("Run not found", 404)
            return
        with open(run_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.send_json({"success": True, "run_id": clean_id, "metadata": meta})

    def handle_api_run_inspect(self, run_id: str, query: dict):
        clean_id = sanitize_name(run_id)
        csv_file = RUNS_DIR / clean_id / "classification.csv"
        run_file = RUNS_DIR / clean_id / "run.json"
        if not csv_file.is_file() or not run_file.is_file():
            self.send_error_json("Run not found", 404)
            return

        try:
            x = int(query.get("x", ["0"])[0])
            y = int(query.get("y", ["0"])[0])
        except ValueError:
            self.send_error_json("Invalid coordinates", 400)
            return

        with open(run_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        stride = meta.get("stride", 1)
        w = meta.get("source_width", 0)
        h = meta.get("source_height", 0)

        # Snap to grid origin
        snap_x = (x // stride) * stride
        snap_y = (y // stride) * stride

        nx = (w - 8) // stride + 1
        ny = (h - 8) // stride + 1

        gx = snap_x // stride
        gy = snap_y // stride

        if gx < 0 or gx >= nx or gy < 0 or gy >= ny:
            self.send_error_json("Coordinates out of decision bounds", 400)
            return

        target_row = gy * nx + gx
        row_data = None

        bin_file = RUNS_DIR / clean_id / "decisions.bin"
        if bin_file.is_file():
            try:
                with open(bin_file, "rb") as bf:
                    hdr = bf.read(64)
                    if len(hdr) == 64 and hdr[:8] == b"TLV_DEC\x00":
                        rec_size = struct.unpack("<I", hdr[28:32])[0]
                        bf.seek(64 + target_row * rec_size)
                        raw = bf.read(rec_size)
                        if len(raw) >= 32:
                            bgx, bgy, box, boy, pred_idx, sec_idx, prob, sec_prob, margin, is_unc = struct.unpack("<IIIIHHfffB", raw[:29])
                            classes = meta.get("classes", [])
                            pred_name = classes[pred_idx] if pred_idx < len(classes) else ""
                            sec_name = classes[sec_idx] if len(classes) > 1 and sec_idx < len(classes) else ""
                            row_data = {
                                "grid_x": str(bgx),
                                "grid_y": str(bgy),
                                "origin_x": str(box),
                                "origin_y": str(boy),
                                "center_x": f"{box + 3.5:.1f}",
                                "center_y": f"{boy + 3.5:.1f}",
                                "display_x": str(box + 4),
                                "display_y": str(boy + 4),
                                "predicted_class": pred_name,
                                "probability": f"{prob:.6f}",
                                "second_class": sec_name,
                                "second_probability": f"{sec_prob:.6f}" if len(classes) > 1 else "",
                                "margin": f"{margin:.6f}",
                                "status": "UNCERTAIN" if is_unc else "CLASSIFIED"
                            }
            except Exception:
                row_data = None

        if row_data is None and csv_file.is_file():
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if idx == target_row:
                        row_data = row
                        break

        if row_data is None:
            self.send_error_json("Decision point not found", 404)
            return

        self.send_json({"success": True, "decision": row_data})

    def handle_api_list_datasets(self):
        self.send_json({"success": True, "datasets": self.get_datasets_list()})

    def handle_api_list_models(self):
        self.send_json({"success": True, "models": self.get_models_list()})

    def handle_api_create_job(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error_json("Invalid JSON payload", 400)
            return

        job_type = data.get("type")
        job_id = f"job-{uuid.uuid4().hex[:8]}"

        if job_type == "train":
            dataset_id = sanitize_name(data.get("dataset_id", ""))
            model_name = sanitize_name(data.get("model_name", f"model_{job_id}"))
            dataset_dir = DATASETS_DIR / dataset_id
            if not dataset_dir.is_dir():
                self.send_error_json("Dataset not found", 404)
                return

            model_path = MODELS_DIR / f"{model_name}.tlv"
            cmd = [str(CLI_BIN), "train", str(dataset_dir), str(model_path)]

        elif job_type == "dense_map":
            model_id = sanitize_name(data.get("model_id", ""))
            image_name = data.get("image_name", "")
            stride = int(data.get("stride", 1))
            conf = float(data.get("confidence_threshold", 0.0))
            margin = float(data.get("margin_threshold", 0.0))
            threads = int(data.get("threads", 0))

            model_path = MODELS_DIR / f"{model_id}.tlv"
            if not model_path.is_file():
                self.send_error_json("Model not found", 404)
                return

            image_path = UPLOADS_DIR / image_name
            if not image_path.is_file():
                self.send_error_json("Source image not found", 404)
                return

            run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            run_dir = RUNS_DIR / run_id
            cmd = [str(CLI_BIN), "map", str(model_path), str(image_path), str(run_dir),
                   "--stride", str(stride),
                   "--confidence", str(conf),
                   "--margin", str(margin)]
            if threads > 0:
                cmd.extend(["--threads", str(threads)])
        else:
            self.send_error_json("Unsupported job type", 400)
            return

        job_record = {
            "job_id": job_id,
            "type": job_type,
            "state": "running",
            "progress": 0.0,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "finished_at": None,
            "command": cmd,
            "exit_code": None,
            "log": "",
            "error": None
        }

        with JOBS_LOCK:
            JOBS[job_id] = job_record

        def worker(jid, command):
            try:
                proc = subprocess.run(command, capture_output=True, text=True)
                with JOBS_LOCK:
                    j = JOBS[jid]
                    j["exit_code"] = proc.returncode
                    j["log"] = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                    j["state"] = "completed" if proc.returncode == 0 else "failed"
                    j["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    j["progress"] = 1.0
            except Exception as exc:
                with JOBS_LOCK:
                    j = JOBS[jid]
                    j["state"] = "failed"
                    j["error"] = str(exc)
                    j["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        t = threading.Thread(target=worker, args=(job_id, cmd), daemon=True)
        t.start()

        self.send_json({"success": True, "job_id": job_id, "state": "running"})

    def handle_api_get_job(self, job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self.send_error_json("Job not found", 404)
            return
        self.send_json({"success": True, "job": job})

    def handle_api_list_jobs(self):
        with JOBS_LOCK:
            jobs_list = list(JOBS.values())
        self.send_json({"success": True, "jobs": jobs_list})

    def handle_api_source_open_local(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error_json("Invalid JSON payload", 400)
            return

        raw_path = req.get("path", "").strip()
        p = Path(raw_path)
        if not p.exists():
            self.send_error_json(f"Caminho não encontrado no sistema: {raw_path}", 404)
            return

        if p.is_dir():
            approve_local_path(p)
            self.open_sentinel_folder(p)
            return

        if p.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            self.send_error_json("Tipo de arquivo não suportado como fonte de imagem.", 415)
            return

        approve_local_path(p)
        w, h, _ = load_image_dimensions_and_rgb(p)
        sha = compute_file_sha256(p)
        self.send_json({
            "success": True,
            "filename": p.name,
            "path": str(p.resolve()),
            "sha256": sha,
            "width": w,
            "height": h,
            "size_bytes": p.stat().st_size,
        })

    def handle_api_source_browse_local(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8")) if content_length > 0 else {}
        except Exception:
            req = {}

        raw_path = req.get("path", "").strip() or str(Path.home())
        p = Path(raw_path).resolve()
        if not p.is_dir():
            p = p.parent
        with APPROVED_PATHS_LOCK:
            browse_roots = [RUNTIME_ROOT, REPO_ROOT, Path.home(), *APPROVED_LOCAL_PATHS]
        if not is_safe_path(p, browse_roots):
            self.send_error_json("Diretório fora das raízes locais autorizadas.", 403)
            return

        entries = []
        try:
            for item in sorted(p.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    entries.append({"name": item.name, "path": str(item), "is_dir": True})
                elif item.suffix.lower() in (".png", ".jpg", ".jpeg", ".jp2", ".j2k", ".tif", ".tiff"):
                    entries.append({"name": item.name, "path": str(item), "is_dir": False, "size": item.stat().st_size})
        except Exception as e:
            self.send_error_json(f"Erro ao listar diretório: {e}", 400)
            return

        self.send_json({
            "success": True,
            "current_path": str(p),
            "parent_path": str(p.parent) if p.parent != p else None,
            "entries": entries,
        })

    def handle_api_sentinel_open(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error_json("Invalid JSON payload", 400)
            return

        folder_path = req.get("folder_path", "").strip()
        bands_input = req.get("bands", {})

        if folder_path:
            p = Path(folder_path).resolve()
            if not p.is_dir():
                self.send_error_json(f"Diretório não encontrado: {folder_path}", 404)
                return
            approve_local_path(p)
            self.open_sentinel_folder(p)
            return

        if bands_input:
            b2_p = Path(bands_input.get("b2", "")).resolve()
            b3_p = Path(bands_input.get("b3", "")).resolve()
            b4_p = Path(bands_input.get("b4", "")).resolve()
            b8_p = Path(bands_input.get("b8", "")).resolve()

            for bp, name in [(b2_p, "B2"), (b3_p, "B3"), (b4_p, "B4"), (b8_p, "B8")]:
                if not bp.is_file():
                    self.send_error_json(f"Arquivo da banda {name} não encontrado: {bp}", 404)
                    return
                if bp.suffix.lower() not in (".jp2", ".j2k", ".tif", ".tiff"):
                    self.send_error_json(f"Tipo de arquivo inválido para a banda {name}: {bp.suffix}", 415)
                    return
                approve_local_path(bp)

            self.process_sentinel_band_set(b2_p, b3_p, b4_p, b8_p)
            return

        self.send_error_json("Especifique folder_path ou as 4 bandas (b2, b3, b4, b8)", 400)

    def open_sentinel_folder(self, directory: Path):
        b2_re = re.compile(r"(^|[_.-])(B02|B2|B02_10m|B2_10m)\.(jp2|tif|tiff)$", re.IGNORECASE)
        b3_re = re.compile(r"(^|[_.-])(B03|B3|B03_10m|B3_10m)\.(jp2|tif|tiff)$", re.IGNORECASE)
        b4_re = re.compile(r"(^|[_.-])(B04|B4|B04_10m|B4_10m)\.(jp2|tif|tiff)$", re.IGNORECASE)
        b8_re = re.compile(r"(^|[_.-])(B08|B8|B08_10m|B8_10m)\.(jp2|tif|tiff)$", re.IGNORECASE)
        tci_re = re.compile(r"(^|[_.-])(TCI|TCI_10m)\.(jp2|tif|tiff)$", re.IGNORECASE)

        b2_matches, b3_matches, b4_matches, b8_matches, tci_matches = [], [], [], [], []
        for root, _, files in os.walk(directory):
            for f in files:
                fp = Path(root) / f
                p_str = str(fp)
                if "R20m" in p_str or "R60m" in p_str or "20m" in f or "60m" in f:
                    continue

                if b2_re.search(f):
                    b2_matches.append(fp)
                elif b3_re.search(f):
                    b3_matches.append(fp)
                elif b4_re.search(f):
                    b4_matches.append(fp)
                elif b8_re.search(f):
                    b8_matches.append(fp)
                elif tci_re.search(f):
                    tci_matches.append(fp)

        if not (b2_matches and b3_matches and b4_matches and b8_matches):
            missing = []
            if not b2_matches: missing.append("B2 (B02/B02_10m)")
            if not b3_matches: missing.append("B3 (B03/B03_10m)")
            if not b4_matches: missing.append("B4 (B04/B04_10m)")
            if not b8_matches: missing.append("B8 (B08/B08_10m)")
            self.send_error_json(f"Bandas 10m não encontradas na pasta: {', '.join(missing)}", 400)
            return

        tci_path = tci_matches[0] if tci_matches else None
        self.process_sentinel_band_set(b2_matches[0], b3_matches[0], b4_matches[0], b8_matches[0], tci_path)

    def process_sentinel_band_set(self, b2_path: Path, b3_path: Path, b4_path: Path, b8_path: Path, tci_path: Path = None):
        try:
            import numpy as np
            preview_max_dim = 2048

            w4, h4, arr4, pw, ph = read_band_2d(b4_path, max_dim=preview_max_dim)
            w3, h3, arr3, _, _ = read_band_2d(b3_path, max_dim=preview_max_dim)
            w2, h2, arr2, _, _ = read_band_2d(b2_path, max_dim=preview_max_dim)
            w8, h8, arr8, _, _ = read_band_2d(b8_path, max_dim=preview_max_dim)

            if arr4 is None or arr3 is None or arr2 is None or arr8 is None:
                self.send_error_json("Falha ao decodificar matrizes raster das bandas Sentinel-2.", 500)
                return

            w, h = w4, h4
            if (w3, h3) != (w, h) or (w2, h2) != (w, h) or (w8, h8) != (w, h):
                self.send_error_json(f"Dimensões incompatíveis entre bandas Sentinel: B4=({w}x{h}), B3=({w3}x{h3}), B2=({w2}x{h2}), B8=({w8}x{h8})", 400)
                return

            band_paths = {"B2": b2_path, "B3": b3_path, "B4": b4_path, "B8": b8_path}
            band_signatures = {name: get_gdal_raster_signature(path) for name, path in band_paths.items()}
            if HAS_GDAL and any(signature is None for signature in band_signatures.values()):
                self.send_error_json("Falha ao ler metadados GDAL de todas as quatro bandas Sentinel-2.", 400)
                return
            if HAS_GDAL:
                base_signature = band_signatures["B2"]
                base_has_geo = bool(base_signature["projection"] and base_signature["geotransform"])
                for band_name, signature in band_signatures.items():
                    has_geo = bool(signature["projection"] and signature["geotransform"])
                    if has_geo != base_has_geo:
                        self.send_error_json(f"Georreferenciamento incompatível entre B2 e {band_name}.", 400)
                        return
                    if base_has_geo:
                        if signature["projection"] != base_signature["projection"]:
                            self.send_error_json(f"CRS incompatível entre B2 e {band_name}.", 400)
                            return
                        if any(abs(a - b) > 1e-7 for a, b in zip(signature["geotransform"], base_signature["geotransform"])):
                            self.send_error_json(f"Geotransform incompatível entre B2 e {band_name}.", 400)
                            return

            prev_hash = compute_sha256(f"{b2_path}_{b3_path}_{b4_path}_{b8_path}".encode("utf-8"))[:12]

            # 1. True Color Preview (B4, B3, B2 or TCI)
            if tci_path and tci_path.is_file():
                # Read TCI directly
                _, _, tci_arr, _, _ = read_band_2d(tci_path, max_dim=preview_max_dim)
                if tci_arr is not None and tci_arr.ndim == 3:
                    tc_img = PILImage.fromarray(np.clip(tci_arr, 0, 255).astype(np.uint8)) if HAS_PIL else None
                else:
                    tc_img = None
            else:
                tc_img = None

            if tc_img is None:
                p_max = max(float(np.percentile(arr4, 98)), float(np.percentile(arr3, 98)), float(np.percentile(arr2, 98)), 2000.0)
                scale = 255.0 / max(p_max, 1.0)
                r = np.clip(arr4 * scale, 0, 255).astype(np.uint8)
                g = np.clip(arr3 * scale, 0, 255).astype(np.uint8)
                b = np.clip(arr2 * scale, 0, 255).astype(np.uint8)
                tc_composite = np.dstack([r, g, b])
                tc_img = PILImage.fromarray(tc_composite) if HAS_PIL else None

            preview_tc_path = UPLOADS_DIR / f"s2_tc_{prev_hash}.png"
            if tc_img and HAS_PIL:
                tc_img.save(preview_tc_path, format="PNG")
            else:
                save_png_patch(tc_composite.tobytes() if 'tc_composite' in locals() else b'', pw, ph, preview_tc_path)

            # 2. False Color NIR (B8, B4, B3 - Near Infrared / Vegetation in red)
            p8_max = max(float(np.percentile(arr8, 98)), float(np.percentile(arr4, 98)), float(np.percentile(arr3, 98)), 2500.0)
            scale_fc = 255.0 / max(p8_max, 1.0)
            fc_r = np.clip(arr8 * scale_fc, 0, 255).astype(np.uint8)
            fc_g = np.clip(arr4 * scale_fc, 0, 255).astype(np.uint8)
            fc_b = np.clip(arr3 * scale_fc, 0, 255).astype(np.uint8)
            fc_composite = np.dstack([fc_r, fc_g, fc_b])
            preview_fc_path = UPLOADS_DIR / f"s2_fc_nir_{prev_hash}.png"
            if HAS_PIL:
                PILImage.fromarray(fc_composite).save(preview_fc_path, format="PNG")
            else:
                save_png_patch(fc_composite.tobytes(), pw, ph, preview_fc_path)

            # 3. False Color Green (B4, B8, B2 - Vegetation in vivid green)
            fc_g_r = np.clip(arr4 * scale_fc, 0, 255).astype(np.uint8)
            fc_g_g = np.clip(arr8 * scale_fc, 0, 255).astype(np.uint8)
            fc_g_b = np.clip(arr2 * scale_fc, 0, 255).astype(np.uint8)
            fc_g_composite = np.dstack([fc_g_r, fc_g_g, fc_g_b])
            preview_fc_green_path = UPLOADS_DIR / f"s2_fc_green_{prev_hash}.png"
            if HAS_PIL:
                PILImage.fromarray(fc_g_composite).save(preview_fc_green_path, format="PNG")
            else:
                save_png_patch(fc_g_composite.tobytes(), pw, ph, preview_fc_green_path)

            # 4. NDVI ((B8 - B4) / (B8 + B4))
            denom = arr8 + arr4
            denom[denom == 0] = 1.0
            ndvi = (arr8 - arr4) / denom
            # Map -0.2..0.8 to RGB colormap (brown to yellow to deep green)
            ndvi_norm = np.clip((ndvi + 0.2) / 1.0, 0.0, 1.0)
            ndvi_r = (255 * (1.0 - ndvi_norm * 0.8)).astype(np.uint8)
            ndvi_g = (255 * (ndvi_norm * 0.9 + 0.1)).astype(np.uint8)
            ndvi_b = (50 * (1.0 - ndvi_norm)).astype(np.uint8)
            ndvi_composite = np.dstack([ndvi_r, ndvi_g, ndvi_b])
            preview_ndvi_path = UPLOADS_DIR / f"s2_ndvi_{prev_hash}.png"
            if HAS_PIL:
                PILImage.fromarray(ndvi_composite).save(preview_ndvi_path, format="PNG")
            else:
                save_png_patch(ndvi_composite.tobytes(), pw, ph, preview_ndvi_path)

            crs_desc = None
            crs_wkt = None
            geo_transform = None
            pixel_size = None
            if HAS_GDAL and band_signatures["B2"]:
                signature = band_signatures["B2"]
                crs_desc = signature["crs_label"]
                crs_wkt = signature["projection"]
                if signature["geotransform"] is not None:
                    geo_transform = list(signature["geotransform"])
                    pixel_size = (
                        (geo_transform[1] ** 2 + geo_transform[4] ** 2) ** 0.5,
                        (geo_transform[2] ** 2 + geo_transform[5] ** 2) ** 0.5,
                    )

            descriptor = {
                "success": True,
                "modality": "SENTINEL2_MULTIBAND",
                "is_sentinel_10m": True,
                "width": w,
                "height": h,
                "preview_width": pw,
                "preview_height": ph,
                "preview_path": str(preview_tc_path),
                "preview_url": f"/api/image?path={preview_tc_path}",
                "previews": {
                    "true_color": f"/api/image?path={preview_tc_path}",
                    "false_color_nir": f"/api/image?path={preview_fc_path}",
                    "false_color_green": f"/api/image?path={preview_fc_green_path}",
                    "ndvi": f"/api/image?path={preview_ndvi_path}",
                },
                "bands": {
                    "b2": {"path": str(b2_path), "filename": b2_path.name, "size_bytes": b2_path.stat().st_size},
                    "b3": {"path": str(b3_path), "filename": b3_path.name, "size_bytes": b3_path.stat().st_size},
                    "b4": {"path": str(b4_path), "filename": b4_path.name, "size_bytes": b4_path.stat().st_size},
                    "b8": {"path": str(b8_path), "filename": b8_path.name, "size_bytes": b8_path.stat().st_size},
                },
                "tci_path": str(tci_path) if tci_path else None,
                "crs": crs_desc,
                "crs_wkt": crs_wkt,
                "geotransform": geo_transform,
                "pixel_size_m": pixel_size[0] if pixel_size else None,
            }

            self.send_json(descriptor)

        except Exception as e:
            self.send_error_json(f"Erro ao processar bandas Sentinel-2: {e}", 500)


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
