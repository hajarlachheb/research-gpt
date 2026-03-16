"""Prompt template manager with versioning support."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from monitoring.logger import get_logger

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class PromptVersion:
    name: str
    version: int
    content: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class PromptManager:
    """Manages prompt templates with versioning and A/B testing support."""

    def __init__(self, templates_dir: Path | None = None):
        self._dir = templates_dir or TEMPLATES_DIR
        self._cache: dict[str, str] = {}
        self._versions: dict[str, list[PromptVersion]] = {}
        self._active_versions: dict[str, int] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Discover and load all template files."""
        if not self._dir.exists():
            logger.warning("templates_dir_missing", path=str(self._dir))
            return

        for f in sorted(self._dir.glob("*.txt")):
            stem = f.stem  # e.g., "answer_v1"
            parts = stem.rsplit("_v", 1)
            name = parts[0]
            version = int(parts[1]) if len(parts) == 2 else 1

            content = f.read_text(encoding="utf-8").strip()

            if name not in self._versions:
                self._versions[name] = []
            self._versions[name].append(PromptVersion(
                name=name, version=version, content=content,
            ))
            self._active_versions.setdefault(name, version)
            self._cache[f"{name}_v{version}"] = content

        logger.info("prompts_loaded", count=len(self._cache))

    def get_prompt(self, name: str, version: int | None = None) -> str:
        """Retrieve a prompt template by name and optional version."""
        v = version or self._active_versions.get(name, 1)
        key = f"{name}_v{v}"
        if key not in self._cache:
            raise KeyError(f"Prompt '{key}' not found. Available: {list(self._cache.keys())}")
        return self._cache[key]

    def set_active_version(self, name: str, version: int) -> None:
        """Switch the active version for a prompt template."""
        key = f"{name}_v{version}"
        if key not in self._cache:
            raise KeyError(f"Prompt version '{key}' does not exist")
        self._active_versions[name] = version
        logger.info("prompt_version_changed", name=name, version=version)

    def list_prompts(self) -> dict[str, list[int]]:
        """List all prompt names and their available versions."""
        result: dict[str, list[int]] = {}
        for name, versions in self._versions.items():
            result[name] = sorted(v.version for v in versions)
        return result

    def add_version(self, name: str, content: str) -> int:
        """Add a new version of a prompt and persist it."""
        existing = self._versions.get(name, [])
        new_version = max((v.version for v in existing), default=0) + 1

        pv = PromptVersion(name=name, version=new_version, content=content)
        self._versions.setdefault(name, []).append(pv)
        key = f"{name}_v{new_version}"
        self._cache[key] = content

        filepath = self._dir / f"{name}_v{new_version}.txt"
        filepath.write_text(content, encoding="utf-8")
        logger.info("prompt_version_added", name=name, version=new_version)

        return new_version
