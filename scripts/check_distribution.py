#!/usr/bin/env python3
"""Check Metis wheel/sdist contents without extracting archives or loading models."""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath
import stat
import tarfile
import zipfile


PRIVATE_ROOTS = {
    "data", "runs", "reports", "artifacts", "logs", "ops", ".local",
    ".git", ".codex", ".agents", ".metis", ".cometa", ".venv", "venv",
}
PRIVATE_ENTRIES = {"AGENTS.md", "CORE.md", "HANDOFF.md", "JUPITER.md"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
REQUIRED_MODULES = {"metis/__init__.py", "metis/tasks/objectives.py", "metis/tasks/decisions.py"}


def member_parts(name: str) -> tuple[str, ...]:
    """Validate the original member name before any path normalization."""
    parts = tuple(name.rstrip("/").split("/"))
    if (not name or name.startswith("/") or "\\" in name
            or PureWindowsPath(name).drive or any(p in {"", ".", ".."} for p in parts)):
        raise ValueError("unsafe archive path")
    return parts


def excluded(parts: tuple[str, ...]) -> bool:
    """Distinguish runtime output roots from source modules and documentation."""
    if not parts:
        return False
    return (
        parts[0] in PRIVATE_ROOTS
        or (len(parts) == 1 and parts[0] in PRIVATE_ENTRIES)
        or parts[:2] in {("docs", "operations"), ("docs", "history")}
        or any(p in CACHE_DIRS or p == ".env" or p.startswith(".env.") for p in parts)
        or parts[-1].endswith((".pyc", ".pyo", ".aux", ".fls", ".fdb_latexmk", ".synctex.gz"))
    )


def check_distribution(path: Path, forbid_text: tuple[str, ...] = ()) -> list[str]:
    """Return content-policy errors; forbidden text values are never reported."""
    errors: list[str] = []
    files: set[str] = set()
    roots: set[str] = set()
    wheel = path.suffix == ".whl"

    def inspect(name: str, kind: str, read) -> None:
        try:
            parts = member_parts(name)
        except ValueError:
            errors.append(f"{name!r}: unsafe archive path")
            return
        if not wheel:
            roots.add(parts[0])
            parts = parts[1:]
            if not parts and kind != "directory":
                errors.append(f"{name!r}: source distribution requires a root directory")
        if kind not in {"file", "directory"}:
            errors.append(f"{name!r}: links and special entries are not allowed")
            return
        if excluded(parts):
            errors.append(f"{name!r}: local-only content or build cache")
        if kind == "directory":
            return
        files.add("/".join(parts))
        if forbid_text:
            try:
                content = read().decode("utf-8")
            except UnicodeDecodeError:
                return
            for index, pattern in enumerate(forbid_text, start=1):
                if pattern in content:
                    errors.append(f"{name!r}: forbidden text pattern #{index}")

    try:
        if wheel or zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    mode = stat.S_IFMT(member.external_attr >> 16)
                    if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
                        kind = "link or special entry"
                    else:
                        kind = "directory" if member.is_dir() else "file"
                    inspect(member.filename, kind, lambda m=member: archive.read(m))
        else:
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    kind = "directory" if member.isdir() else "file" if member.isfile() else "link or special entry"

                    def read_member(m=member):
                        stream = archive.extractfile(m)
                        if stream is None:
                            raise OSError("archive member is unreadable")
                        with stream:
                            return stream.read()

                    inspect(member.name, kind, read_member)
    except (OSError, EOFError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile):
        errors.append("cannot read distribution archive")
        return errors

    if not wheel and len(roots) != 1:
        errors.append("source distribution requires exactly one root directory")
    required = REQUIRED_MODULES if wheel else {"src/" + p for p in REQUIRED_MODULES} | {"README.md", "pyproject.toml"}
    for missing in sorted(required - files):
        errors.append(f"{missing}: missing required distribution file")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+", help="Wheel or source-distribution archive paths")
    parser.add_argument("--forbid-text", action="append", default=[], metavar="TEXT",
                        help="Reject this literal, case-sensitive string in UTF-8 members; repeatable")
    args = parser.parse_args(argv)
    if any(not value for value in args.forbid_text):
        parser.error("--forbid-text must be nonempty")
    failed = False
    for path in args.archives:
        errors = check_distribution(path, tuple(args.forbid_text))
        for error in errors:
            print(f"{path.name}: {error}")
        print(f"{path.name}: {'FAILED' if errors else 'passed'}")
        failed |= bool(errors)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
