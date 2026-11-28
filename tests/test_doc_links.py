"""Relative links in the docs must point at a file in this repository."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _pages() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _target(page: Path, raw: str) -> tuple[Path, str]:
    body = raw.strip().split()[0]
    path_text, _, anchor = body.partition("#")
    if path_text == "":
        return page, anchor
    return (page.parent / path_text).resolve(), anchor


def test_relative_doc_links_resolve():
    missing: list[str] = []
    for page in _pages():
        for match in LINK.finditer(page.read_text(encoding="utf-8")):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, anchor = _target(page, target)
            if not path.is_file():
                missing.append(f"{page.relative_to(ROOT)} -> {target}")
                continue
            if anchor and not _has_heading(path, anchor):
                missing.append(f"{page.relative_to(ROOT)} -> {target} (missing heading)")
    assert missing == []


def _has_heading(path: Path, anchor: str) -> bool:
    headings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            headings.append(_slug(title))
    return anchor in headings


def _slug(title: str) -> str:
    cleaned = title.lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return re.sub(r"\s+", "-", cleaned.strip())
