"""Extract text from PDF research papers with section detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class ExtractedSection:
    title: str
    text: str
    page_numbers: list[int] = field(default_factory=list)


HEADING_PATTERNS = [
    re.compile(r"^\d+\.?\s+[A-Z]"),              # "1. Introduction", "2 Related Work"
    re.compile(r"^(?:Abstract|Introduction|Related Work|Methodology|Method|"
               r"Methods|Experiments|Results|Discussion|Conclusion|"
               r"Conclusions|References|Acknowledgments|Appendix)",
               re.IGNORECASE),
]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 120:
        return False
    return any(p.match(stripped) for p in HEADING_PATTERNS)


def extract_pdf(file_path: str | Path) -> list[ExtractedSection]:
    """Extract text from a PDF and split into sections."""
    doc = fitz.open(str(file_path))
    sections: list[ExtractedSection] = []
    current_title = "Untitled"
    current_lines: list[str] = []
    current_pages: list[int] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _is_heading(line):
                if current_lines:
                    sections.append(ExtractedSection(
                        title=current_title,
                        text="\n".join(current_lines),
                        page_numbers=sorted(set(current_pages)),
                    ))
                current_title = line
                current_lines = []
                current_pages = [page_num + 1]
            else:
                current_lines.append(line)
                if page_num + 1 not in current_pages:
                    current_pages.append(page_num + 1)

    if current_lines:
        sections.append(ExtractedSection(
            title=current_title,
            text="\n".join(current_lines),
            page_numbers=sorted(set(current_pages)),
        ))

    doc.close()
    return sections


