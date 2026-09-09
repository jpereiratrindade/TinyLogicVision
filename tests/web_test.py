#!/usr/bin/env python3
"""
Automated Test Suite for TinyLogicVision Web GUI & Dataset Authoring v0.1
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_BIN = REPO_ROOT / "bin" / "tinyvision"
SERVER_SCRIPT = REPO_ROOT / "web" / "server.py"

def create_test_image(path: Path, width: int = 32, height: int = 32) -> str:
    """Create a synthetic RGB PNG test image with distinctive color blocks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            if y < height // 3:
                # Forest green
                raw.extend([30, 180, 45])
            elif y < (2 * height) // 3:
                # Grassland yellow
                raw.extend([200, 180, 50])
            else:
                # Soil brown
                raw.extend([170, 100, 45])
    
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    
    with open(path, "wb") as f:
        f.write(png_bytes)
    
    return hashlib.sha256(png_bytes).hexdigest()

def test_web_help():
    print("Testing 'tinyvision web --help' and 'server.py --help'...")
    res = subprocess.run([sys.executable, str(SERVER_SCRIPT), "--help"], capture_output=True, text=True)
    assert res.returncode == 0, "server.py --help failed"
    assert "TinyLogicVision" in res.stdout or "usage" in res.stdout.lower()

def main():
    test_web_help()

    test_port = 8765
    server_process = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--port", str(test_port), "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    base_url = f"http://127.0.0.1:{test_port}"
    print(f"Starting test server at {base_url}...")

    # Wait for server startup
    started = False
    for _ in range(30):
        try:
            req = urllib.request.urlopen(f"{base_url}/api/status", timeout=1)
            if req.status == 200:
                started = True
                break
        except Exception:
            time.sleep(0.1)

    if not started:
        server_process.kill()
        out, err = server_process.communicate()
        print("Server stdout:", out)
        print("Server stderr:", err)
        raise RuntimeError("Server failed to start")

    try:
        # 1. Test Static files
        print("Testing static file delivery...")
        with urllib.request.urlopen(f"{base_url}/") as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "TinyLogicVision" in html
            assert "Preparar Dataset" in html

        with urllib.request.urlopen(f"{base_url}/style.css") as res:
            assert res.status == 200
            assert "TinyLogicVision" in res.read().decode("utf-8")

        with urllib.request.urlopen(f"{base_url}/app.js") as res:
            assert res.status == 200
            assert "TinyLogicVision" in res.read().decode("utf-8")

        # 2. Test Path Traversal Protection
        print("Testing path traversal security...")
        try:
            urllib.request.urlopen(f"{base_url}/api/image?path=../../../../etc/passwd", timeout=1)
            raise AssertionError("Path traversal was not blocked!")
        except urllib.error.HTTPError as e:
            assert e.code in (400, 403, 404), f"Unexpected HTTP status for path traversal: {e.code}"

        # 3. Test Image Upload
        print("Testing image upload...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_img = Path(tmp_dir) / "source_sentinel.png"
            img_sha = create_test_image(test_img, width=32, height=32)

            with open(test_img, "rb") as f:
                img_data = f.read()

            req = urllib.request.Request(
                f"{base_url}/api/upload",
                data=img_data,
                headers={"Content-Type": "image/png", "Content-Length": str(len(img_data))},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                upload_resp = json.loads(res.read().decode("utf-8"))
                assert upload_resp["success"] is True
                assert upload_resp["sha256"] == img_sha
                uploaded_path = upload_resp["path"]

            # 4. Test Dataset Creation with Patches & Manifest
            print("Testing 8x8 patch extraction and dataset authoring...")
            dataset_name = f"test_ds_{int(time.time())}"
            patches = [
                # Forest patches (y=0..8)
                {"roi_id": "roi_f1", "class": "floresta", "split": "train", "x": 0, "y": 0},
                {"roi_id": "roi_f1", "class": "floresta", "split": "train", "x": 8, "y": 0},
                {"roi_id": "roi_f1", "class": "floresta", "split": "dev", "x": 16, "y": 0},
                {"roi_id": "roi_f1", "class": "floresta", "split": "probe", "x": 24, "y": 0},
                # Grassland patches (y=10..18)
                {"roi_id": "roi_g1", "class": "campo", "split": "train", "x": 0, "y": 10},
                {"roi_id": "roi_g1", "class": "campo", "split": "train", "x": 8, "y": 10},
                {"roi_id": "roi_g1", "class": "campo", "split": "dev", "x": 16, "y": 10},
                {"roi_id": "roi_g1", "class": "campo", "split": "probe", "x": 24, "y": 10},
                # Soil patches (y=22..30)
                {"roi_id": "roi_s1", "class": "solo", "split": "train", "x": 0, "y": 22},
                {"roi_id": "roi_s1", "class": "solo", "split": "train", "x": 8, "y": 22},
                {"roi_id": "roi_s1", "class": "solo", "split": "dev", "x": 16, "y": 22},
                {"roi_id": "roi_s1", "class": "solo", "split": "probe", "x": 24, "y": 22},
            ]

            create_payload = json.dumps({
                "dataset_name": dataset_name,
                "image_path": uploaded_path,
                "is_sentinel_10m": True,
                "patches": patches
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/api/datasets/create",
                data=create_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                ds_resp = json.loads(res.read().decode("utf-8"))
                assert ds_resp["success"] is True
                assert ds_resp["total_patches"] == 12
                assert ds_resp["counts"]["train"]["floresta"] == 2
                assert ds_resp["counts"]["dev"]["floresta"] == 1
                assert ds_resp["counts"]["probe"]["floresta"] == 1

            # Verify manifest.csv and dataset.json exist and are well-formed
            ds_dir = REPO_ROOT / ".tinyvision" / "datasets" / dataset_name
            manifest_csv = ds_dir / "manifest.csv"
            dataset_json = ds_dir / "dataset.json"
            assert manifest_csv.is_file(), "manifest.csv missing"
            assert dataset_json.is_file(), "dataset.json missing"

            with open(dataset_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                assert meta["is_sentinel_10m"] is True
                assert len(meta["patches"]) == 12

            # 5. Test Training through API (runs ./bin/tinyvision train)
            print("Testing model training via web API...")
            model_name = f"model_{dataset_name}"
            train_payload = json.dumps({
                "dataset_name": dataset_name,
                "model_name": model_name
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/api/train",
                data=train_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                train_resp = json.loads(res.read().decode("utf-8"))
                assert train_resp["success"] is True
                assert train_resp["parameter_count"] == 4707  # 3 classes: 192->24->3

            # 6. Test Classification via API (runs ./bin/tinyvision classify)
            print("Testing model classification via web API...")
            classify_patch_path = str(ds_dir / "probe" / "floresta" / list((ds_dir / "probe" / "floresta").glob("*.png"))[0].name)
            classify_payload = json.dumps({
                "model_name": model_name,
                "image_path": str(ds_dir / "probe" / "floresta" / classify_patch_path)
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/api/classify",
                data=classify_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                cls_resp = json.loads(res.read().decode("utf-8"))
                assert cls_resp["success"] is True
                assert cls_resp["predicted"] == "floresta"
                assert cls_resp["score"] > 0.5

            # 7. Test Evaluation for DEV and PROBE via API
            print("Testing split evaluation via web API...")
            for split in ("dev", "probe"):
                eval_payload = json.dumps({
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "split": split
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{base_url}/api/evaluate",
                    data=eval_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    assert res.status == 200
                    eval_resp = json.loads(res.read().decode("utf-8"))
                    assert eval_resp["success"] is True
                    assert eval_resp["samples"] == 3
                    assert eval_resp["accuracy"] >= 0.0

            # Clean up test dataset and model from .tinyvision
            shutil.rmtree(ds_dir, ignore_errors=True)
            model_file = REPO_ROOT / ".tinyvision" / "models" / f"{model_name}.tlv"
            if model_file.exists():
                model_file.unlink()

        print("TinyLogicVision Web GUI test suite PASS")

    finally:
        server_process.kill()
        server_process.wait()

if __name__ == "__main__":
    main()
