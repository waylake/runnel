"""Deterministic public coding-agent workload definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

PYTHON_FILES = {
    "src/cache.py": '''from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Entry(Generic[T]):
    value: T
    hits: int = 0


class LRUCache(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[str, Entry[T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        entry.hits += 1
        self._items.move_to_end(key)
        return entry.value

    def put(self, key: str, value: T) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = Entry(value)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
''',
    "src/repository.py": '''from __future__ import annotations

from pathlib import Path

from .cache import LRUCache


class RepositoryIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.contents: LRUCache[str] = LRUCache(128)

    def read(self, relative: str) -> str:
        cached = self.contents.get(relative)
        if cached is not None:
            return cached
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("path escapes repository")
        text = path.read_text(encoding="utf-8")
        self.contents.put(relative, text)
        return text
''',
    "src/service.py": '''from __future__ import annotations

from dataclasses import dataclass

from .repository import RepositoryIndex


@dataclass
class Finding:
    path: str
    line: int
    message: str


class ReviewService:
    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    def find_todos(self, paths: list[str]) -> list[Finding]:
        findings: list[Finding] = []
        for path in paths:
            for number, line in enumerate(self.index.read(path).splitlines(), 1):
                if "TODO" in line:
                    findings.append(Finding(path, number, line.strip()))
        return findings
''',
    "tests/test_cache.py": '''from src.cache import LRUCache


def test_evicts_least_recently_used() -> None:
    cache = LRUCache[int](2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
''',
}

TYPESCRIPT_FILES = {
    "src/index.ts": '''export interface SourceFile {
  path: string;
  text: string;
  revision: number;
}

export class PrefixIndex {
  private readonly files = new Map<string, SourceFile>();

  upsert(file: SourceFile): void {
    this.files.set(file.path, file);
  }

  longestPrefix(left: string, right: string): number {
    const limit = Math.min(left.length, right.length);
    for (let length = limit; length > 0; length -= 1) {
      if (left.slice(0, length) === right.slice(0, length)) return length;
    }
    return 0;
  }
}
''',
    "src/tool.ts": '''export type ToolResult = { ok: true; value: string } | { ok: false; error: string };

export async function runTool(name: string, input: string): Promise<ToolResult> {
  if (!name) return { ok: false, error: "missing tool name" };
  return { ok: true, value: `${name}:${input.trim()}` };
}
''',
}

SYSTEM_PROMPT = """You are a careful coding agent. Inspect the supplied repository before answering. Preserve unrelated behavior, make the smallest coherent patch, and explain correctness and tests concisely."""

TASK_PROMPT = """Analyze the supplied repository. Identify one correctness bug, explain why it happens, and propose a minimal patch with a regression test. Do not modify unrelated files."""


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    files: dict[str, str]
    system_prompt: str
    turns: tuple[str, ...]

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "files": self.files,
                "system_prompt": self.system_prompt,
                "turns": self.turns,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def source_text(self) -> str:
        blocks = []
        for path, text in sorted(self.files.items()):
            language = "typescript" if path.endswith(".ts") else "python"
            blocks.append(f"--- FILE: {path} ({language}) ---\n{text.rstrip()}\n")
        return "\n".join(blocks)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "description": self.description,
            "sha256": self.digest(),
            "file_names": sorted(self.files),
            "source_bytes": len(self.source_text().encode("utf-8")),
            "turn_count": len(self.turns),
        }


PYTHON_WORKLOAD = Workload(
    name="python-multi-file-v1",
    description="Four-file Python repository, analysis-to-patch agent loop",
    files=PYTHON_FILES,
    system_prompt=SYSTEM_PROMPT,
    turns=(
        TASK_PROMPT,
        "Implement the minimal fix and regression test you proposed. Return a unified diff.",
        "The test fails with a KeyError after eviction. Diagnose the failure and revise the patch.",
    ),
)

TYPESCRIPT_WORKLOAD = Workload(
    name="typescript-tools-v1",
    description="TypeScript source-index and tool-result insertion loop",
    files=TYPESCRIPT_FILES,
    system_prompt=SYSTEM_PROMPT,
    turns=(
        "Review the TypeScript files and explain one likely edge-case bug.",
        "Add a focused regression test and return the patch.",
        "A tool returned an empty string. Continue using that result without repeating the repository analysis.",
    ),
)

WORKLOADS = {
    workload.name: workload
    for workload in (PYTHON_WORKLOAD, TYPESCRIPT_WORKLOAD)
}


def get_workload(name: str) -> Workload:
    try:
        return WORKLOADS[name]
    except KeyError as exc:
        raise KeyError(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}") from exc
