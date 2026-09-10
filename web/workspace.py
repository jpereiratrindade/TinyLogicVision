#!/usr/bin/env python3
"""TinyLogicVision workspace creation and validation."""

import argparse
import datetime
import json
import os
from pathlib import Path

WORKSPACE_SCHEMA = "tinylogicvision.workspace/v1"
WORKSPACE_DIRECTORIES = ("uploads", "datasets", "models", "runs", "manifests")


def resolve_workspace(path: str | Path | None, repo_root: Path) -> Path:
    value = Path(path).expanduser() if path else repo_root / ".tinyvision"
    return value.resolve()


def initialize_workspace(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"workspace path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in WORKSPACE_DIRECTORIES:
        (root / name).mkdir(exist_ok=True)

    manifest_path = root / "workspace.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid workspace manifest: {manifest_path}") from exc
        if manifest.get("schema") != WORKSPACE_SCHEMA:
            raise ValueError(f"unsupported workspace schema in {manifest_path}")
    else:
        manifest = {
            "schema": WORKSPACE_SCHEMA,
            "name": root.name,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        temporary = root / f".workspace.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyLogicVision workspace manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create or validate a workspace")
    init_parser.add_argument("path")
    args = parser.parse_args()
    if args.command == "init":
        root = initialize_workspace(args.path)
        print(f"TinyLogicVision workspace ready: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
