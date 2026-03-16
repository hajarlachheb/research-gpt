"""Clean and normalize extracted text from research papers."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Apply a chain of cleaning steps to raw extracted text."""
    text = _normalize_unicode(text)
    text = _fix_hyphenation(text)
    text = _collapse_whitespace(text)
    text = _remove_header_footer_artifacts(text)
    text = _strip_references_numbers(text)
    return text.strip()


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text


def _fix_hyphenation(text: str) -> str:
    """Rejoin words split across lines by hyphens."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _remove_header_footer_artifacts(text: str) -> str:
    """Remove common PDF extraction artifacts like page numbers."""
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*preprint.*$", "", text, flags=re.MULTILINE)
    return text


def _strip_references_numbers(text: str) -> str:
    """Lighten inline citation markers like [1], [2,3]."""
    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", "", text)
