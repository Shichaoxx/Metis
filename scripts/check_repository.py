#!/usr/bin/env python3
"""Check repository structure, inline Markdown file links, and recipe syntax."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


DOC_DIRS = ("docs", "cookbooks", "examples", "recipes", "tests", "ops", "scripts")
RUNTIME_DIRS = {"data", "runs", "reports", "artifacts", "logs", ".local"}
REQUIRED = ("README.md", "CONTRIBUTING.md", "pyproject.toml", "src/cometa",
            "tests", "docs/README.md", "cookbooks/README.md", "examples/README.md",
            "recipes/README.md")
INLINE_LINK = re.compile(
    r"!?\[(?:\\.|[^\]\\\n])*\]\(\s*"
    r"(<[^>\n]*>|(?:\\.|[^\s()\\]+|\([^()\n]*\))+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def files_under(directory: Path, suffix: str):
    """Walk only real directories; never follow a directory or file symlink."""
    if directory.is_symlink() or not directory.is_dir():
        return
    for base, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        for name in sorted(names):
            path = Path(base) / name
            if path.suffix == suffix and not path.is_symlink():
                yield path


def inline_links(path: Path):
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = FENCE.match(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        for match in INLINE_LINK.finditer(line):
            yield number, match[1].removeprefix("<").removesuffix(">")


def reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only structure, inline Markdown file-link and recipe JSON checks.",
        epilog="Scans root Markdown and docs/cookbooks/examples/recipes/tests/ops/scripts. "
               "Skips fenced code, reference-style links, URLs, anchors and local runtime "
               "links (data/runs/reports/artifacts/logs/.local). Does not check network "
               "targets or heading anchors, follow symlinks, load models, or run tests.",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parent.parent,
                        help="Repository or unpacked source-distribution root")
    root = Path(os.path.abspath(parser.parse_args().root))
    errors = []
    for name in REQUIRED:
        if not (root / name).exists():
            errors.append(f"{name}: missing required repository entry")
    documents = sorted(p for p in root.glob("*.md") if not p.is_symlink())
    for name in DOC_DIRS:
        documents.extend(files_under(root / name, ".md"))
    checked = runtime = symlinks = 0
    for document in documents:
        try:
            for line, destination in inline_links(document):
                url = urlsplit(destination)
                if url.scheme or url.netloc or not url.path:
                    continue
                local = re.sub(r"\\([\\`*_{}\[\]()#+.!<> -])", r"\1", unquote(url.path))
                # Normalize before inspecting symlinks: data/../docs is documentation.
                target = Path(os.path.abspath(document.parent / local))
                try:
                    relative = target.relative_to(root)
                except ValueError:
                    errors.append(f"{document.relative_to(root)}:{line}: link leaves repository {destination}")
                    continue
                if relative.parts and relative.parts[0] in RUNTIME_DIRS:
                    runtime += 1
                    continue
                # Check parents first, before a lookup can follow an intermediate link.
                if any(p.is_symlink() for p in reversed((target, *target.parents)) if p != root):
                    symlinks += 1
                    continue
                checked += 1
                if not target.exists():
                    errors.append(f"{document.relative_to(root)}:{line}: missing link target {destination}")
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{document.relative_to(root)}: cannot check Markdown: {exc}")
    recipes = list(files_under(root / "recipes", ".json"))
    for recipe in recipes:
        try:
            json.loads(recipe.read_text(encoding="utf-8"), parse_constant=reject_constant)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{recipe.relative_to(root)}: invalid JSON: {exc}")
    try:
        import tomllib
    except ImportError:
        print("Skipped TOML parsing: Python 3.10 has no standard-library tomllib.")
    else:
        try:
            with (root / "pyproject.toml").open("rb") as stream:
                tomllib.load(stream)
        except (OSError, ValueError) as exc:
            errors.append(f"pyproject.toml: invalid TOML: {exc}")
    print(f"Checked {len(documents)} Markdown files, {checked} local links, "
          f"{len(recipes)} recipe JSON files, and required structure.")
    print(f"Skipped local runtime links: {runtime}; skipped other symlink links: {symlinks}.")
    if errors:
        for error in errors:
            print(error)
        print(f"FAILED: {len(errors)} issue(s).")
        return 1
    print("Repository checks passed (network targets and heading anchors not checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
