#!/usr/bin/env python3
"""Apply reproducible, secret-safe oMLX model settings for local benchmarks.

The API key is read from the local oMLX settings only for the login request and
is never printed or persisted by this script.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class OMLXError(RuntimeError):
    pass


class OMLXAdmin:
    def __init__(self, base_url: str, omlx_home: Path) -> None:
        settings = json.loads((omlx_home / "settings.json").read_text(encoding="utf-8"))
        api_key = settings.get("auth", {}).get("api_key")
        if not api_key:
            raise OMLXError("oMLX API key is not configured")
        self.base_url = base_url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )
        login = urllib.request.Request(
            f"{self.base_url}/admin/api/login",
            data=json.dumps({"api_key": api_key, "remember": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(login, timeout=30) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            raise OMLXError(f"oMLX admin login failed: HTTP {exc.code}") from exc

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise OMLXError(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc

    def apply(self, model: str, settings: dict[str, Any], reload: bool) -> Any:
        encoded = urllib.parse.quote(model, safe="")
        if reload:
            try:
                self.call("POST", f"/admin/api/models/{encoded}/unload")
            except OMLXError as exc:
                if "HTTP 400" not in str(exc):
                    raise
        effective = self.call(
            "PUT", f"/admin/api/models/{encoded}/settings", settings
        )
        if reload:
            self.call("POST", f"/admin/api/models/{encoded}/load")
        return effective

    def clear_caches(self) -> dict[str, Any]:
        return {
            "ssd": self.call("POST", "/admin/api/ssd-cache/clear", {}),
            "hot": self.call("POST", "/admin/api/hot-cache/clear", {}),
        }


def load_settings(path: Path, model: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "models" in data:
        try:
            return data["models"][model]
        except KeyError as exc:
            raise OMLXError(f"model not present in {path.name}: {model}") from exc
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an oMLX benchmark profile")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--omlx-home", type=Path, default=Path.home() / ".omlx")
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--effective-output", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.settings, args.model)
    client = OMLXAdmin(args.base_url, args.omlx_home.expanduser())
    if args.clear_cache:
        client.clear_caches()
    effective = client.apply(args.model, settings, reload=not args.no_reload)
    effective_settings = effective.get("settings", {}) if isinstance(effective, dict) else {}
    mismatches = {
        key: {"requested": value, "effective": effective_settings.get(key)}
        for key, value in settings.items()
        if effective_settings.get(key) != value
    }
    if mismatches:
        raise OMLXError(f"effective settings differ: {mismatches}")
    if args.effective_output:
        args.effective_output.parent.mkdir(parents=True, exist_ok=True)
        args.effective_output.write_text(
            json.dumps(effective_settings, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"applied {len(settings)} settings to {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
