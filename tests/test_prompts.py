"""Tests for prompt manager and versioning."""

from __future__ import annotations

import os
import tempfile
import shutil
import pytest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_pm")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from prompts.manager import PromptManager


@pytest.fixture
def temp_templates():
    d = Path(tempfile.mkdtemp())
    (d / "answer_v1.txt").write_text("Answer template v1", encoding="utf-8")
    (d / "answer_v2.txt").write_text("Answer template v2", encoding="utf-8")
    (d / "rewrite_v1.txt").write_text("Rewrite template v1", encoding="utf-8")
    yield d
    shutil.rmtree(d)


class TestPromptManager:
    def test_load_templates(self, temp_templates):
        pm = PromptManager(temp_templates)
        prompts = pm.list_prompts()
        assert "answer" in prompts
        assert 1 in prompts["answer"]
        assert 2 in prompts["answer"]
        assert "rewrite" in prompts

    def test_get_prompt_default_version(self, temp_templates):
        pm = PromptManager(temp_templates)
        content = pm.get_prompt("answer")
        assert "Answer template" in content

    def test_get_prompt_specific_version(self, temp_templates):
        pm = PromptManager(temp_templates)
        v2 = pm.get_prompt("answer", version=2)
        assert v2 == "Answer template v2"

    def test_set_active_version(self, temp_templates):
        pm = PromptManager(temp_templates)
        pm.set_active_version("answer", 2)
        content = pm.get_prompt("answer")
        assert content == "Answer template v2"

    def test_add_version(self, temp_templates):
        pm = PromptManager(temp_templates)
        new_v = pm.add_version("answer", "Answer template v3")
        assert new_v == 3
        content = pm.get_prompt("answer", version=3)
        assert content == "Answer template v3"
        assert (temp_templates / "answer_v3.txt").exists()

    def test_get_missing_prompt_raises(self, temp_templates):
        pm = PromptManager(temp_templates)
        with pytest.raises(KeyError):
            pm.get_prompt("nonexistent")

    def test_set_invalid_version_raises(self, temp_templates):
        pm = PromptManager(temp_templates)
        with pytest.raises(KeyError):
            pm.set_active_version("answer", 99)
