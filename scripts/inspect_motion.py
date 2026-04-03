#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "This script requires numpy to inspect motion files. "
            "Please run it in the RoboJuDo runtime environment."
        ) from exc
    return np


def _import_joblib():
    try:
        import joblib
    except ImportError:
        joblib = None
    return joblib


def _short_scalar(value: Any) -> str:
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def _array_summary(name: str, value: Any, indent: int = 0, preview_items: int = 6) -> None:
    np = _import_numpy()
    prefix = " " * indent
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    print(f"{prefix}{name}: type={type(value).__name__}, shape={shape}, dtype={dtype}")

    if not hasattr(value, "shape"):
        return

    try:
        arr = np.asarray(value)
    except Exception as exc:
        print(f"{prefix}  preview: <failed to convert to ndarray: {exc}>")
        return

    if arr.size == 0:
        print(f"{prefix}  preview: []")
        return

    flat = arr.reshape(-1)
    preview = ", ".join(_short_scalar(item) for item in flat[:preview_items].tolist())
    suffix = ", ..." if flat.size > preview_items else ""
    print(f"{prefix}  preview: [{preview}{suffix}]")


def _describe_item(name: str, value: Any, indent: int = 0, depth: int = 0, max_depth: int = 2) -> None:
    prefix = " " * indent

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        _array_summary(name, value, indent=indent)
        return

    if isinstance(value, Mapping):
        print(f"{prefix}{name}: dict(len={len(value)})")
        if depth >= max_depth:
            return
        for child_key, child_value in list(value.items()):
            _describe_item(str(child_key), child_value, indent=indent + 2, depth=depth + 1, max_depth=max_depth)
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}(len={len(value)})")
        if depth >= max_depth:
            return
        for index, child_value in enumerate(list(value)[:5]):
            _describe_item(f"[{index}]", child_value, indent=indent + 2, depth=depth + 1, max_depth=max_depth)
        if len(value) > 5:
            print(f"{prefix}  ...")
        return

    print(f"{prefix}{name}: type={type(value).__name__}, value={_short_scalar(value)}")


def _normalize_npz_like(data: Any) -> tuple[list[str], Any]:
    if hasattr(data, "files"):
        return list(data.files), data
    if isinstance(data, Mapping):
        return list(data.keys()), data
    return [], data


def check_beyondmimic_npz(path: Path) -> int:
    np = _import_numpy()
    data = np.load(path, allow_pickle=True)
    keys, container = _normalize_npz_like(data)

    required_keys = [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    optional_keys = ["hand_pose"]

    print(f"Check: BeyondMimic compatibility for {path}")

    print(f"Keys ({len(keys)}): {', '.join(keys)}")
    for key in keys:
        _describe_item(key, container[key], indent=2, max_depth=1)

    missing_keys = [key for key in required_keys if key not in keys]
    if missing_keys:
        print(f"Status: FAIL")
        print(f"Missing keys: {', '.join(missing_keys)}")
        return 1

    print("Status: PASS (required keys present)")

    for key in required_keys + [key for key in optional_keys if key in keys]:
        value = container[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"  {key}: shape={shape}, dtype={dtype}")
        if dtype == object:
            print(f"    warning: dtype=object, loading requires allow_pickle=True")

    try:
        joint_pos = np.asarray(container["joint_pos"])
        joint_vel = np.asarray(container["joint_vel"])
        body_pos_w = np.asarray(container["body_pos_w"])
        body_quat_w = np.asarray(container["body_quat_w"])
        body_lin_vel_w = np.asarray(container["body_lin_vel_w"])
        body_ang_vel_w = np.asarray(container["body_ang_vel_w"])

        num_frames = joint_pos.shape[0]
        if joint_vel.shape[0] != num_frames:
            print("  warning: joint_vel frame count does not match joint_pos")
        if body_pos_w.shape[0] != num_frames:
            print("  warning: body_pos_w frame count does not match joint_pos")
        if body_quat_w.shape[0] != num_frames:
            print("  warning: body_quat_w frame count does not match joint_pos")
        if body_lin_vel_w.shape[0] != num_frames:
            print("  warning: body_lin_vel_w frame count does not match joint_pos")
        if body_ang_vel_w.shape[0] != num_frames:
            print("  warning: body_ang_vel_w frame count does not match joint_pos")
    except Exception as exc:
        print(f"  warning: failed to run shape consistency checks: {exc}")

    return 0


def inspect_npz(path: Path) -> int:
    np = _import_numpy()
    data = np.load(path, allow_pickle=True)
    keys, container = _normalize_npz_like(data)
    print(f"File: {path}")
    print("Format: npz")
    print(f"Keys ({len(keys)}): {', '.join(keys)}")
    for key in keys:
        value = container[key]
        _describe_item(key, value, indent=2)
    return 0


def inspect_pkl(path: Path) -> int:
    joblib = _import_joblib()

    loaders = []
    if joblib is not None:
        loaders.append(("joblib", lambda p: joblib.load(p)))
    loaders.append(("pickle", lambda p: pickle.load(p.open("rb"))))

    errors: list[str] = []
    obj = None
    used_loader = None

    for loader_name, loader in loaders:
        try:
            obj = loader(path)
            used_loader = loader_name
            break
        except Exception as exc:
            errors.append(f"{loader_name}: {exc}")

    if used_loader is None:
        print(f"File: {path}")
        print("Format: pkl")
        print("Failed to load file with available loaders:")
        for err in errors:
            print(f"  - {err}")
        if joblib is None:
            print("Hint: install joblib if this file was saved with joblib.")
        return 1

    print(f"File: {path}")
    print("Format: pkl")
    print(f"Loader: {used_loader}")
    _describe_item("root", obj, indent=0)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RoboJuDo motion files (.npz / .pkl).")
    parser.add_argument("path", type=Path, help="Path to the motion file")
    parser.add_argument(
        "-c",
        "--check-beyondmimic",
        action="store_true",
        help="Check whether a .npz file matches the BeyondMimic motion keys",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    path = args.path

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    suffix = path.suffix.lower()
    if suffix == ".npz":
        if args.check_beyondmimic:
            return check_beyondmimic_npz(path)
        return inspect_npz(path)
    if suffix == ".pkl":
        if args.check_beyondmimic:
            print("BeyondMimic check only supports .npz files.", file=sys.stderr)
            return 2
        return inspect_pkl(path)

    print(f"Unsupported file type: {path.suffix}", file=sys.stderr)
    print("Supported extensions: .npz, .pkl", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
