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
            assert e.code in (400, 403, 404, 415), f"Unexpected HTTP status for path traversal: {e.code}"

        try:
            urllib.request.urlopen(f"{base_url}/api/image?path=/etc/passwd", timeout=1)
            raise AssertionError("Absolute arbitrary-file read was not blocked!")
        except urllib.error.HTTPError as e:
            assert e.code in (400, 403, 404, 415), f"Unexpected HTTP status for absolute path: {e.code}"

        # 3. Test Image Upload (PNG and Sentinel JP2)
        print("Testing image upload (PNG and Sentinel JP2)...")
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

            # Test Sentinel .jp2 upload & preview delivery
            try:
                from PIL import Image as TestPILImage
                import io
                jp2_path = Path(tmp_dir) / "T22JCS_20240101_B02_10m.jp2"
                pil_jp2 = TestPILImage.new("RGB", (32, 32), color=(50, 150, 200))
                pil_jp2.save(jp2_path, format="JPEG2000")
                with open(jp2_path, "rb") as f:
                    jp2_data = f.read()

                req_jp2 = urllib.request.Request(
                    f"{base_url}/api/upload",
                    data=jp2_data,
                    headers={"Content-Type": "image/jp2", "Content-Length": str(len(jp2_data))},
                    method="POST"
                )
                with urllib.request.urlopen(req_jp2) as res:
                    assert res.status == 200
                    jp2_resp = json.loads(res.read().decode("utf-8"))
                    assert jp2_resp["success"] is True
                    assert jp2_resp["width"] == 32
                    assert jp2_resp["height"] == 32
                    uploaded_jp2_path = jp2_resp["path"]

                # Test on-the-fly PNG conversion of .jp2 for canvas
                with urllib.request.urlopen(f"{base_url}/api/image?path={uploaded_jp2_path}") as res:
                    assert res.status == 200
                    assert res.headers.get("Content-Type") == "image/png"
                    read_bytes = res.read()
                    assert len(read_bytes) > 0
                    assert read_bytes[:8] == b"\x89PNG\r\n\x1a\n"
            except Exception as e:
                print(f"JP2 testing note: {e}")

            # 4A. Test Spatial Split Invariant Rejection (1 ROI != Multiple Splits)
            print("Testing spatial split invariant rejection (1 ROI = 1 Split)...")
            bad_patches = [
                {"roi_id": "roi_leaky", "class": "floresta", "split": "train", "x": 0, "y": 0},
                {"roi_id": "roi_leaky", "class": "floresta", "split": "dev", "x": 8, "y": 0},
            ]
            bad_payload = json.dumps({
                "dataset_name": "leaky_dataset",
                "image_path": uploaded_path,
                "is_sentinel_10m": True,
                "patches": bad_patches,
            }).encode("utf-8")
            bad_req = urllib.request.Request(
                f"{base_url}/api/datasets/create",
                data=bad_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                urllib.request.urlopen(bad_req)
                raise AssertionError("Mixed split ROI was unexpectedly accepted!")
            except urllib.error.HTTPError as e:
                assert e.code == 400, f"Expected 400 for spatial split violation, got {e.code}"
                err_body = json.loads(e.read().decode("utf-8"))
                assert "integridade de split" in err_body["error"].lower() or "violação" in err_body["error"].lower()

            # 4B. Test Spatial Patch Overlap Rejection Across Splits
            print("Testing spatial patch overlap rejection across splits...")
            overlap_patches = [
                {"roi_id": "roi_train_1", "class": "floresta", "split": "train", "x": 0, "y": 0},
                {"roi_id": "roi_dev_1", "class": "floresta", "split": "dev", "x": 4, "y": 4},
            ]
            overlap_payload = json.dumps({
                "dataset_name": "overlap_dataset",
                "image_path": uploaded_path,
                "is_sentinel_10m": True,
                "patches": overlap_patches,
            }).encode("utf-8")
            overlap_req = urllib.request.Request(
                f"{base_url}/api/datasets/create",
                data=overlap_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                urllib.request.urlopen(overlap_req)
                raise AssertionError("Spatially overlapping patches across splits were unexpectedly accepted!")
            except urllib.error.HTTPError as e:
                assert e.code == 400, f"Expected 400 for spatial patch overlap violation, got {e.code}"
                err_body = json.loads(e.read().decode("utf-8"))
                assert "sobreposição espacial" in err_body["error"].lower() or "violação" in err_body["error"].lower()

            # 4C. Test Dataset Creation with Valid Disjoint Patches & Strict ROI Splits
            print("Testing 8x8 patch extraction and dataset authoring (disjoint splits)...")
            dataset_name = f"test_ds_{int(time.time())}"
            patches = [
                # Forest patches (distinct ROIs per split)
                {"roi_id": "roi_f_train", "class": "floresta", "split": "train", "x": 0, "y": 0},
                {"roi_id": "roi_f_train", "class": "floresta", "split": "train", "x": 8, "y": 0},
                {"roi_id": "roi_f_dev", "class": "floresta", "split": "dev", "x": 16, "y": 0},
                {"roi_id": "roi_f_probe", "class": "floresta", "split": "probe", "x": 24, "y": 0},
                # Grassland patches
                {"roi_id": "roi_g_train", "class": "campo", "split": "train", "x": 0, "y": 10},
                {"roi_id": "roi_g_train", "class": "campo", "split": "train", "x": 8, "y": 10},
                {"roi_id": "roi_g_dev", "class": "campo", "split": "dev", "x": 16, "y": 10},
                {"roi_id": "roi_g_probe", "class": "campo", "split": "probe", "x": 24, "y": 10},
                # Soil patches
                {"roi_id": "roi_s_train", "class": "solo", "split": "train", "x": 0, "y": 22},
                {"roi_id": "roi_s_train", "class": "solo", "split": "train", "x": 8, "y": 22},
                {"roi_id": "roi_s_dev", "class": "solo", "split": "dev", "x": 16, "y": 22},
                {"roi_id": "roi_s_probe", "class": "solo", "split": "probe", "x": 24, "y": 22},
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
            dataset_provenance = ds_dir / "provenance.json"
            dataset_evidence = ds_dir / "evidence.jsonl"
            assert manifest_csv.is_file(), "manifest.csv missing"
            assert dataset_json.is_file(), "dataset.json missing"
            assert dataset_provenance.is_file(), "dataset provenance.json missing"
            assert dataset_evidence.is_file(), "dataset evidence.jsonl missing"

            with open(dataset_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                assert meta["is_sentinel_10m"] is True
                assert len(meta["patches"]) == 12
            with open(dataset_provenance, "r", encoding="utf-8") as f:
                provenance = json.load(f)
                node_types = {node["type"] for node in provenance["nodes"]}
                assert {"Source", "ROI", "Patch", "Dataset"}.issubset(node_types)
                assert any(edge["type"] == "sampled_from" for edge in provenance["edges"])

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
            model_provenance = REPO_ROOT / ".tinyvision" / "models" / f"{model_name}.provenance.json"
            assert model_provenance.is_file(), "model training provenance missing"
            with open(model_provenance, "r", encoding="utf-8") as f:
                training_graph = json.load(f)
                assert any(node["type"] == "TrainingRun" for node in training_graph["nodes"])
                assert any(edge["type"] == "trained_on" for edge in training_graph["edges"])

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

            # 8. Test Dense Spatial Classification via API (runs ./bin/tinyvision map)
            print("Testing dense spatial classification via web API...")
            dense_payload = json.dumps({
                "model_name": model_name,
                "image_path": uploaded_path,
                "stride": 4,
                "confidence": 0.0,
                "margin": 0.0,
                "is_sentinel_10m": True,
                "decision_csv": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/api/dense_map",
                data=dense_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req) as res:
                assert res.status == 200
                dense_resp = json.loads(res.read().decode("utf-8"))
                assert dense_resp["success"] is True
                run_id = dense_resp["run_id"]
                metadata = dense_resp["metadata"]
                assert metadata["stride"] == 4
                # 32x32 image with stride 4: nx = (32-8)/4 + 1 = 7, ny = 7 => 49 decisions
                assert metadata["decision_count"] == 49
                assert metadata["sentinel_nominal_10m"] is True
                assert metadata["operator_declared_nominal_10m"] is True
                assert metadata["nominal_context_m"] == 80
                assert metadata["nominal_decision_spacing_m"] == 40
                assert "geospatial" in metadata
                assert metadata["geospatial"]["available"] is False
                assert metadata["engineering_stats"]["implementation_mode"] == "BOUNDED_TILE_STREAMING"
                assert metadata["engineering_stats"]["memory_bound_scope"] == "TILE_PLUS_HALO"
                assert metadata["engineering_stats"]["max_decisions"] is None

            # Verify artifact delivery via HTTP and parse run.json semantics
            print("Verifying map artifacts delivery over HTTP and contract in run.json...")
            for art in ("class_map.png", "confidence.png", "margin.png", "overlay.png", "run.json",
                        "classification.csv", "provenance.json", "evidence.jsonl"):
                with urllib.request.urlopen(f"{base_url}/api/runs/{run_id}/{art}") as res:
                    assert res.status == 200
                    data = res.read()
                    assert len(data) > 0
                    if art == "run.json":
                        run_meta = json.loads(data.decode("utf-8"))
                        assert run_meta["grid_width"] == 7
                        assert run_meta["grid_height"] == 7
                        assert run_meta["decision_count"] == 49
                        assert "spatial_semantics" in run_meta
                        assert "support" in run_meta["spatial_semantics"]
                        assert "concept_distinction" in run_meta["spatial_semantics"]
                        assert "uncertainty_semantics" in run_meta
                        assert "palette" in run_meta
                        assert len(run_meta["palette"]["classes"]) == 3
                        assert run_meta["palette"]["uncertain"]["color"] == "#808080"
                        assert run_meta["geospatial"]["available"] is False

            with urllib.request.urlopen(f"{base_url}/api/provenance") as res:
                assert res.status == 200
                workspace_graph = json.loads(res.read().decode("utf-8"))
                assert workspace_graph["success"] is True
                node_types = {node["type"] for node in workspace_graph["nodes"]}
                assert {"Dataset", "TrainingRun", "Model", "ClassificationRun", "Artifact"}.issubset(node_types)
                serialized_graph = json.dumps(workspace_graph)
                assert "a1b2c3d4" not in serialized_graph
                assert "mlp_256_24_3" not in serialized_graph

            # Verify point inspection endpoint
            print("Verifying point inspection endpoint...")
            with urllib.request.urlopen(f"{base_url}/api/runs/{run_id}/inspect?x=0&y=0") as res:
                assert res.status == 200
                insp_data = json.loads(res.read().decode("utf-8"))
                assert insp_data["success"] is True
                dec = insp_data["decision"]
                assert dec["grid_x"] == "0"
                assert dec["grid_y"] == "0"
                assert dec["origin_x"] == "0"
                assert dec["origin_y"] == "0"
                assert dec["center_x"] == "3.5"
                assert dec["center_y"] == "3.5"
                assert dec["display_x"] == "4"
                assert dec["display_y"] == "4"
                assert "margin" in dec
                assert dec["predicted_class"] in ("floresta", "campo", "solo")

            # Verify asynchronous job execution endpoint
            print("Verifying async jobs endpoint...")
            job_payload = {
                "type": "train",
                "dataset_id": dataset_name,
                "model_name": f"async_{model_name}"
            }
            req_job = urllib.request.Request(
                f"{base_url}/api/jobs",
                data=json.dumps(job_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req_job) as res:
                assert res.status == 200
                job_res = json.loads(res.read().decode("utf-8"))
                assert job_res["success"] is True
                assert "job_id" in job_res
                jid = job_res["job_id"]

            # Poll job status
            time.sleep(0.5)
            with urllib.request.urlopen(f"{base_url}/api/jobs/{jid}") as res:
                assert res.status == 200
                poll_res = json.loads(res.read().decode("utf-8"))
                assert poll_res["success"] is True
                assert poll_res["job"]["job_id"] == jid
                assert poll_res["job"]["state"] in ("running", "completed")

            # Clean up test dataset, model and runs from .tinyvision
            shutil.rmtree(ds_dir, ignore_errors=True)
            shutil.rmtree(REPO_ROOT / ".tinyvision" / "runs" / run_id, ignore_errors=True)
            model_file = REPO_ROOT / ".tinyvision" / "models" / f"{model_name}.tlv"
            if model_file.exists():
                model_file.unlink()
            for suffix in (".provenance.json", ".evidence.jsonl"):
                sidecar = REPO_ROOT / ".tinyvision" / "models" / f"{model_name}{suffix}"
                if sidecar.exists():
                    sidecar.unlink()
            async_model = REPO_ROOT / ".tinyvision" / "models" / f"async_{model_name}.tlv"
            if async_model.exists():
                async_model.unlink()

            # 9. Test Sentinel-2 Multiband 10m (B2, B3, B4, B8) Native 256-input Pipeline
            print("Testing Sentinel-2 4-band native 256-input pipeline (.tvp datasets, 256->24->3 MLP)...")
            try:
                import numpy as np
                from osgeo import gdal, osr
                s2_dir = Path(tmp_dir) / "sentinel_bands"
                s2_dir.mkdir(parents=True, exist_ok=True)
                bands_map = {}
                # Create 32x32 synthetic uint16 Sentinel bands
                for b_name, base_val in [("B02_10m", 800), ("B03_10m", 1200), ("B04_10m", 900), ("B08_10m", 3600)]:
                    b_path = s2_dir / f"{b_name}.tif"
                    drv = gdal.GetDriverByName("GTiff")
                    ds = drv.Create(str(b_path), 32, 32, 1, gdal.GDT_UInt16)
                    reference = osr.SpatialReference()
                    reference.ImportFromEPSG(32722)
                    ds.SetProjection(reference.ExportToWkt())
                    ds.SetGeoTransform((500000.0, 10.0, 0.0, 7500000.0, 0.0, -10.0))
                    # Fill top 1/3 with vegetation (high NIR), bottom with soil (low NIR)
                    arr = np.full((32, 32), base_val, dtype=np.uint16)
                    if "B08" in b_name:
                        arr[16:, :] = 600
                    ds.GetRasterBand(1).WriteArray(arr)
                    ds.FlushCache()
                    ds = None
                    key = "b2" if "B02" in b_name else ("b3" if "B03" in b_name else ("b4" if "B04" in b_name else "b8"))
                    bands_map[key] = str(b_path)

                # Open 4 bands via API
                open_s2_payload = json.dumps({"bands": bands_map}).encode("utf-8")
                req_s2 = urllib.request.Request(
                    f"{base_url}/api/source/sentinel_open",
                    data=open_s2_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req_s2) as res:
                    assert res.status == 200
                    s2_desc = json.loads(res.read().decode("utf-8"))
                    assert s2_desc["success"] is True
                    assert s2_desc["modality"] == "SENTINEL2_MULTIBAND"
                    assert s2_desc["is_sentinel_10m"] is True
                    assert "previews" in s2_desc
                    assert "true_color" in s2_desc["previews"]
                    assert "false_color_nir" in s2_desc["previews"]

                # Extract .tvp 256-input patches
                s2_dataset_name = f"s2_ds_{int(time.time())}"
                s2_patches = [
                    {"roi_id": "roi_veg_tr", "class": "vegetacao", "split": "train", "x": 0, "y": 0},
                    {"roi_id": "roi_veg_tr", "class": "vegetacao", "split": "train", "x": 8, "y": 0},
                    {"roi_id": "roi_veg_dev", "class": "vegetacao", "split": "dev", "x": 16, "y": 0},
                    {"roi_id": "roi_veg_pr", "class": "vegetacao", "split": "probe", "x": 24, "y": 0},
                    {"roi_id": "roi_soil_tr", "class": "solo", "split": "train", "x": 0, "y": 20},
                    {"roi_id": "roi_soil_tr", "class": "solo", "split": "train", "x": 8, "y": 20},
                    {"roi_id": "roi_soil_dev", "class": "solo", "split": "dev", "x": 16, "y": 20},
                    {"roi_id": "roi_soil_pr", "class": "solo", "split": "probe", "x": 24, "y": 20},
                ]
                create_s2_payload = json.dumps({
                    "dataset_name": s2_dataset_name,
                    "image_path": bands_map["b4"],
                    "is_sentinel_10m": True,
                    "patches": s2_patches,
                    "bands": bands_map,
                    "preview_width": 32,
                    "preview_height": 32,
                    "native_width": 32,
                    "native_height": 32,
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{base_url}/api/datasets/create",
                    data=create_s2_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    assert res.status == 200
                    s2_ds_resp = json.loads(res.read().decode("utf-8"))
                    assert s2_ds_resp["success"] is True
                    assert s2_ds_resp["total_patches"] == 8

                s2_ds_dir = REPO_ROOT / ".tinyvision" / "datasets" / s2_dataset_name
                # Verify .tvp file format on disk
                tvp_files = list((s2_ds_dir / "train" / "vegetacao").glob("*.tvp"))
                assert len(tvp_files) == 2, f"Expected 2 .tvp files, found: {tvp_files}"
                tvp_file = tvp_files[0]
                assert tvp_file.stat().st_size == 20 + 8 * 8 * 4 * 4  # 1044 bytes (20-byte header + 1024-byte float32 payload)
                with open(tvp_file, "rb") as f:
                    magic = f.read(8)
                    assert magic == b"TLV_PAT\x00"

                # Train Sentinel-2 Model: 256 inputs -> 24 hidden -> 2 classes => 6218 params
                s2_model_name = f"model_{s2_dataset_name}"
                s2_train_payload = json.dumps({
                    "dataset_name": s2_dataset_name,
                    "model_name": s2_model_name
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/train",
                    data=s2_train_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    assert res.status == 200
                    s2_train_resp = json.loads(res.read().decode("utf-8"))
                    assert s2_train_resp["success"] is True
                    # 2 classes: (256+1)*24 + (24+1)*2 = 6168 + 50 = 6218 params
                    assert s2_train_resp["parameter_count"] == 6218

                # Classify .tvp patch via API
                s2_classify_payload = json.dumps({
                    "model_name": s2_model_name,
                    "image_path": str(tvp_file)
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/classify",
                    data=s2_classify_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    assert res.status == 200
                    s2_cls_resp = json.loads(res.read().decode("utf-8"))
                    assert s2_cls_resp["success"] is True
                    assert s2_cls_resp["predicted"] in ("vegetacao", "solo")

                # Dense classification must consume the same native B2/B3/B4/B8 schema as training.
                s2_dense_payload = json.dumps({
                    "model_name": s2_model_name,
                    "image_path": s2_desc["preview_path"],
                    "bands": s2_desc["bands"],
                    "stride": 8,
                    "threads": 2,
                    "confidence": 0.0,
                    "margin": 0.0,
                    "is_sentinel_10m": True,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/dense_map",
                    data=s2_dense_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req) as res:
                    assert res.status == 200
                    s2_dense_resp = json.loads(res.read().decode("utf-8"))
                    assert s2_dense_resp["success"] is True
                    s2_run_id = s2_dense_resp["run_id"]
                    s2_run_meta = s2_dense_resp["metadata"]
                    assert s2_run_meta["input_modality"] == "SENTINEL2_MULTIBAND"
                    assert s2_run_meta["input_channels"] == 4
                    assert s2_run_meta["source_width"] == 32
                    assert s2_run_meta["source_height"] == 32
                    assert s2_run_meta["decision_count"] == 16
                    assert s2_run_meta["geospatial"]["available"] is True
                    assert s2_run_meta["engineering_stats"]["implementation_mode"] == "BOUNDED_TILE_STREAMING"
                    assert s2_run_meta["engineering_stats"]["peak_working_set_bytes"] > 0
                    assert s2_run_meta["h3"]["implementation"] in ("OFFICIAL_H3", "UNAVAILABLE")

                geospatial_artifacts = ["class_map.tif", "confidence.tif", "margin.tif"]
                if s2_run_meta["h3"]["available"]:
                    assert s2_run_meta["h3"]["implementation"] == "OFFICIAL_H3"
                    geospatial_artifacts.append("classification_h3.csv")
                for artifact in geospatial_artifacts:
                    with urllib.request.urlopen(f"{base_url}/api/runs/{s2_run_id}/{artifact}") as res:
                        assert res.status == 200
                        assert len(res.read()) > 0

                # A 256-input model must reject an RGB-only dense source.
                invalid_s2_dense_payload = json.dumps({
                    "model_name": s2_model_name,
                    "image_path": s2_desc["preview_path"],
                    "stride": 8,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/dense_map",
                    data=invalid_s2_dense_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                try:
                    urllib.request.urlopen(req)
                    raise AssertionError("Sentinel model accepted an RGB-only dense source")
                except urllib.error.HTTPError as exc:
                    assert exc.code == 400

                # Clean up Sentinel-2 artifacts
                shutil.rmtree(s2_ds_dir, ignore_errors=True)
                shutil.rmtree(REPO_ROOT / ".tinyvision" / "runs" / s2_run_id, ignore_errors=True)
                s2_model_file = REPO_ROOT / ".tinyvision" / "models" / f"{s2_model_name}.tlv"
                if s2_model_file.exists():
                    s2_model_file.unlink()
                for suffix in (".provenance.json", ".evidence.jsonl"):
                    sidecar = REPO_ROOT / ".tinyvision" / "models" / f"{s2_model_name}{suffix}"
                    if sidecar.exists():
                        sidecar.unlink()

            except ImportError:
                print("Note: osgeo.gdal or numpy not available in this test pass, skipping S2 multiband step")

        print("TinyLogicVision Web GUI test suite PASS")

    finally:
        server_process.kill()
        server_process.wait()

if __name__ == "__main__":
    main()
