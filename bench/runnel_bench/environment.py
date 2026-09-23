"""Privacy-safe macOS benchmark environment capture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SENSITIVE_KEYS = {
    "activationlockstatus",
    "machinename",
    "modelnumber",
    "platformuuid",
    "provisioningudid",
    "serial",
    "serialnumber",
    "udid",
}
SENSITIVE_KEY_SUBSTRINGS = ("serialnumber", "platformuuid", "provisioningudid")


def _run(argv: list[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "available": False, "error": str(exc)}
    return {
        "argv": argv,
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _json_command(argv: list[str], timeout: float = 30.0) -> Any | None:
    result = _run(argv, timeout=timeout)
    if not result.get("available"):
        return None
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return None


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in SENSITIVE_KEYS or any(
                fragment in normalized for fragment in SENSITIVE_KEY_SUBSTRINGS
            ):
                continue
            clean[key] = _scrub(item)
        return clean
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\(id=\d+\)", "(id=<redacted>)", value)
    return value


def _package_versions() -> dict[str, str | None]:
    names = (
        "mlx",
        "mlx-lm",
        "transformers",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
        "psutil",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(repo_root: Path) -> dict[str, Any]:
    result = _run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], timeout=10.0
    )
    dirty = _run(
        ["git", "-C", str(repo_root), "status", "--porcelain"], timeout=10.0
    )
    return {
        "commit": result.get("stdout") if result.get("available") else None,
        "dirty": bool(dirty.get("stdout")) if dirty.get("available") else None,
    }


def _memory_pressure() -> dict[str, Any]:
    raw = _run(["memory_pressure"], timeout=10.0)
    text = raw.get("stdout", "")
    match = re.search(r"free percentage:\s*(\d+)%", text)
    return {
        "free_percent": int(match.group(1)) if match else None,
        "command": raw,
        "vm_stat": _run(["vm_stat"], timeout=10.0),
        "swap": _run(["sysctl", "-n", "vm.swapusage"], timeout=10.0),
    }


def _power() -> dict[str, Any]:
    return {
        "state": _run(["pmset", "-g", "ps"], timeout=10.0),
        "thermal": _run(["pmset", "-g", "therm"], timeout=10.0),
        "battery": _run(["pmset", "-g", "battery"], timeout=10.0),
    }


def _process_settings() -> dict[str, Any]:
    names = (
        "hw.model",
        "hw.memsize",
        "hw.ncpu",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.perflevel0.logicalcpu",
        "hw.perflevel1.logicalcpu",
        "hw.perflevel2.logicalcpu",
        "hw.optional.armv8_1_atomics",
    )
    sysctl: dict[str, str | None] = {}
    for name in names:
        result = _run(["sysctl", "-n", name], timeout=5.0)
        sysctl[name] = result.get("stdout") if result.get("available") else None
    selected_env = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
            "MTL_DEBUG_LAYER",
            "MTL_SHADER_VALIDATION",
        )
    }
    return {
        "sysctl": sysctl,
        "ulimit": _run(["sh", "-c", "ulimit -a"], timeout=5.0),
        "selected_environment": selected_env,
    }


def capture_environment(repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = (repo_root or Path.cwd()).resolve()
    hardware = _scrub(_json_command(["system_profiler", "SPHardwareDataType", "-json"]))
    displays = _scrub(_json_command(["system_profiler", "SPDisplaysDataType", "-json"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": _git(repo_root),
        "python": {
            "version": sys.version,
            "executable_name": Path(sys.executable).name,
            "packages": _package_versions(),
        },
        "macos": {
            "sw_vers": _run(["sw_vers"]),
            "xcode": _run(["xcodebuild", "-version"]),
            "sdk": _run(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
            "sdk_build": _run(["xcrun", "--sdk", "macosx", "--show-sdk-build-version"]),
            "metal": _run(["xcrun", "metal", "-v"]),
            "clang": _run(["clang", "--version"]),
        },
        "hardware": hardware,
        "displays": displays,
        "memory": _memory_pressure(),
        "power": _power(),
        "process": _process_settings(),
        "counters": {
            "gpu_counters": "not captured by this stdlib probe",
            "powermetrics": "not run: requires elevated privileges and is opt-in",
        },
    }


def write_json(data: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Runnel benchmark environment")
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="Repository root"
    )
    args = parser.parse_args(argv)
    data = capture_environment(args.repo_root)
    if args.output:
        write_json(data, args.output)
    else:
        print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
