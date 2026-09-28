"""Run lineage and integrity-checked portable inference artifacts."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import uuid
from pathlib import Path
from .schema import sha256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event(run_dir, name, **fields):
    path = Path(run_dir) / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": utc_now(), "event": name, **fields}, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    packages = {}
    for name in ("torch", "transformers", "accelerate", "peft", "cometa-local"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def create_run(root, project, config):
    if not project or any(x in project for x in ("/", "\\")) or project in {".", ".."}:
        raise ValueError("project must be a directory name, not a path")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = Path(root).resolve() / project / run_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "run.json", {"run_id": run_id, "project": project, "status": "running", "created_at": utc_now()})
    write_json(path / "config.resolved.json", config)
    write_json(path / "environment.json", environment())
    source_root = Path(__file__).resolve().parent
    write_json(path / "source.sha256.json", {
        str(item.relative_to(source_root)): sha256(item)
        for item in sorted(source_root.rglob("*.py"))
    })
    event(path, "run_started", run_id=run_id)
    return path


def finish_run(run_dir, *, status="completed", **fields):
    path = Path(run_dir) / "run.json"
    value = json.loads(path.read_text())
    value.update(status=status, updated_at=utc_now(), **fields)
    write_json(path, value)
    event(run_dir, "run_" + status, **fields)


def seal_artifact(path, *, task=None, source_run=None):
    path = Path(path).resolve()
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Portable exports must not contain symlinks")
        if item.is_file() and item.name != "artifact.json":
            files[item.relative_to(path).as_posix()] = sha256(item)
    if not files:
        raise ValueError("Cannot seal empty artifact")
    spec_path = path / "model_spec.json"
    family = json.loads(spec_path.read_text()).get("family") if spec_path.exists() else None
    artifact = {"format_version": "1.0", "kind": "task_model", "model_family": family, "created_at": utc_now(),
                "files": files, "task": task or {"kind": "ranking"}, "source_run": source_run,
                "environment": environment()}
    write_json(path / "artifact.json", artifact)
    return artifact


def verify_artifact(path):
    path = Path(path).resolve()
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Portable exports must not contain symlinks")
    artifact = json.loads((path / "artifact.json").read_text())
    if artifact.get("format_version") != "1.0" or not artifact.get("files"):
        raise ValueError("Unsupported or empty artifact manifest")
    for relative, digest in artifact["files"].items():
        item = (path / relative).resolve()
        if not item.is_relative_to(path) or not item.is_file() or sha256(item) != digest:
            raise ValueError(f"Artifact integrity error: {relative}")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "artifact.json"}
    if actual != set(artifact["files"]):
        raise ValueError("Artifact has unregistered files")
    return artifact
