"""Canonical data contracts, independent of any tokenizer or tensor library."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

TASK_KINDS = {"ranking", "multi_label", "single_choice"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(sample: dict, *, require_labels: bool = True,
                    task_kind: str = "ranking") -> dict:
    if task_kind not in TASK_KINDS:
        raise ValueError(f"Unsupported task kind: {task_kind}")
    if sample.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    for field in ("id", "task_id"):
        if not isinstance(sample.get(field), str) or not sample[field]:
            raise ValueError(f"{field} must be a non-empty string")
    inputs = sample.get("input")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("query"), str):
        raise ValueError("input.query must be a string")
    if "context" in inputs and not isinstance(inputs["context"], str):
        raise ValueError("input.context must be a string")
    candidates = sample.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be an object")
        if not isinstance(candidate.get("id"), str) or not candidate["id"]:
            raise ValueError("Each candidate needs a stable string id")
        if not isinstance(candidate.get("text"), str):
            raise ValueError("candidate.text must be a string")
        ids.append(candidate["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    supervision = sample.get("supervision")
    if supervision is None:
        if require_labels:
            raise ValueError("Supervision required")
        return sample
    if supervision.get("kind") != "candidate_labels":
        raise ValueError("Only explicit candidate_labels supervision is supported")
    if supervision.get("unjudged_policy", "ignore") not in {"ignore", "error"}:
        raise ValueError("Unjudged policy must be ignore or error; do not silently label 0")
    labels = supervision.get("labels")
    if not isinstance(labels, dict) or (require_labels and not labels):
        raise ValueError("Expected non-empty labels keyed by candidate ID")
    if set(labels) - set(ids):
        raise ValueError("Label refers to candidate outside this sample")
    for grade in labels.values():
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            raise ValueError("Labels must be numeric")
        if not math.isfinite(grade) or grade < 0:
            raise ValueError("Labels must be finite and non-negative")
        if task_kind != "ranking" and grade not in {0, 1}:
            raise ValueError("Classification targets must be binary")
    if supervision.get("unjudged_policy") == "error" and set(labels) != set(ids):
        raise ValueError("Fully judged sample is missing candidate labels")
    if task_kind == "single_choice" and (set(labels) != set(ids) or sum(labels.values()) != 1):
        raise ValueError("single_choice requires all labels and exactly one positive")
    return sample


def read_jsonl(path: str | Path, *, require_labels: bool = True,
               task_kind: str = "ranking") -> Iterator[dict]:
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = validate_sample(json.loads(line), require_labels=require_labels,
                                       task_kind=task_kind)
                if item["id"] in seen:
                    raise ValueError("Duplicate sample ID")
                seen.add(item["id"])
                yield item
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, samples) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")


def load_manifest(path: str | Path, *, verify: bool = True) -> dict:
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    if result.get("schema_version") != "1.0" or not result.get("dataset_id"):
        raise ValueError("Manifest needs schema_version=1.0 and dataset_id")
    if result.get("task_kind", "ranking") not in TASK_KINDS:
        raise ValueError("Unknown task_kind in manifest")
    if not isinstance(result.get("splits"), dict) or not result["splits"]:
        raise ValueError("Manifest requires splits")
    for name, spec in result["splits"].items():
        resolved = (path.parent / spec["path"]).resolve()
        if not resolved.is_file():
            raise ValueError(f"Missing split {name}: {resolved}")
        if verify and (not spec.get("sha256") or sha256(resolved) != spec["sha256"]):
            raise ValueError(f"Split checksum mismatch: {name}")
        spec["path"] = str(resolved)
        for auxiliary in ("qrels", "bm25", "retrieval"):
            key = auxiliary + "_path"
            if spec.get(key):
                spec[key] = str((path.parent / spec[key]).resolve())
                if not Path(spec[key]).is_file():
                    raise ValueError(f"Missing {auxiliary} for {name}")
                if verify and (not spec.get(auxiliary + "_sha256") or sha256(spec[key]) != spec[auxiliary + "_sha256"]):
                    raise ValueError(f"{auxiliary} checksum mismatch: {name}")
    result["_manifest_path"] = str(path)
    result["_manifest_sha256"] = sha256(path)
    return result


def validate_dataset(path: str | Path) -> dict:
    manifest = load_manifest(path)
    report = {"dataset_id": manifest["dataset_id"], "manifest_sha256": manifest["_manifest_sha256"], "splits": {}}
    ids_by_split = {}
    for split, spec in manifest["splits"].items():
        samples = list(read_jsonl(spec["path"], require_labels=split == "train",
                                 task_kind=manifest.get("task_kind", "ranking")))
        ids_by_split[split] = {x["id"] for x in samples}
        if "count" in spec and spec["count"] != len(samples):
            raise ValueError(f"Split count mismatch: {split}")
        report["splits"][split] = {"samples": len(samples), "candidates": sum(len(x["candidates"]) for x in samples),
            "all_negative": sum(bool(x.get("supervision", {}).get("labels", {})) and not any(x["supervision"]["labels"].values()) for x in samples),
            "without_judgments": sum(not x.get("supervision", {}).get("labels", {}) for x in samples),
            "unjudged": sum(len(x["candidates"]) - len(x.get("supervision", {}).get("labels", {})) for x in samples)}
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            if ids_by_split[left] & ids_by_split[right]:
                raise ValueError(f"Overlapping sample IDs: {left}/{right}")
    return report
