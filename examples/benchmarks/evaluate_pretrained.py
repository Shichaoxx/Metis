#!/usr/bin/env python3
"""Resumable, local-only Qwen reranker evaluation over a frozen manifest.

Example (weights must already exist locally):
  python evaluate_pretrained.py --manifest data/manifest.json --model /local/model \
    --output runs/qwen-test --device mps --dtype float16 --backend sdpa
Continue the exact run with the same arguments plus --resume. A --limit run is
always marked PILOT, never as a full held-out benchmark. No queries are dropped.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import uuid

import cometa
from cometa.api import decide
from cometa.artifacts import environment, write_json
from cometa.metrics import aggregate, load_qrels, ranking_metrics
from cometa.schema import load_manifest, read_jsonl, sha256

VERSION = "cometa-real-eval-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def emit(event, **fields):
    print(json.dumps({"time": now(), "event": event, **fields}, ensure_ascii=False,
                     allow_nan=False), flush=True)


def sync(torch, device):
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def memory(torch, device):
    kind = torch.device(device).type
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if kind == "cuda":
        result["cuda_session_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_session_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    elif kind == "mps":
        result["mps_sampled_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_sampled_driver_bytes"] = torch.mps.driver_allocated_memory()
        result["mps_measurement"] = "sampled after a query; not a peak allocation measurement"
    return result


def local_model_identity(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("--model must be an already downloaded local model directory containing config.json")
    files = {str(item.relative_to(root)): {"sha256": sha256(item), "bytes": item.stat().st_size}
             for item in sorted(root.rglob("*")) if item.is_file() and ".cache" not in item.relative_to(root).parts}
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("No local model weights found; this evaluator does not download models")
    return {"source": str(root), "files": files, "content_sha256": digest_json(files)}


def source_identity():
    root = Path(cometa.__file__).resolve().parent
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}


def device_identity(torch, device):
    kind = torch.device(device).type
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA is unavailable")
        return {"type": kind, "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)), "cuda_version": torch.version.cuda}
    if kind == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested MPS is unavailable")
        return {"type": kind, "machine": platform.machine(), "platform": platform.platform()}
    if kind != "cpu":
        raise ValueError("Only cpu, mps and cuda devices are supported")
    return {"type": kind, "machine": platform.machine(), "platform": platform.platform()}


@contextlib.contextmanager
def output_lock(directory):
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluator owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_completed(path, query_ids, *, repair_partial_tail):
    """Only incomplete, non-newline-terminated tails may be recovered and retried."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not repair_partial_tail:
            raise ValueError("Incomplete prediction tail; resume is required for recovery")
        boundary = data.rfind(b"\n") + 1
        backup = path.with_name("incomplete-tail-" + uuid.uuid4().hex + ".bin")
        backup.write_bytes(data[boundary:])
        with path.open("r+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        data = data[:boundary]
        emit("recovered_incomplete_tail", backup=str(backup))
    records = []
    for index, line in enumerate(data.splitlines()):
        record = json.loads(line)
        if index >= len(query_ids) or record.get("id") != query_ids[index] or record.get("ordinal") != index:
            raise ValueError("Saved predictions are not an exact ordered prefix of the requested queries")
        if record.get("status") != "ok" or not math.isfinite(record.get("latency_ms", float("nan"))) or record["latency_ms"] < 0:
            raise ValueError("Invalid saved prediction status or latency")
        records.append(record)
    return records


def finish(output, protocol, records, target_samples, qrels, k):
    if len(records) != len(protocol["sample_ids"]):
        raise ValueError("Cannot aggregate an incomplete evaluation")
    rows, truncated_queries, truncated_candidates, candidate_count = [], 0, 0, 0
    for sample, prediction in zip(target_samples, records):
        ids = [item["id"] for item in sample["candidates"]]
        if prediction["id"] != sample["id"] or [s["candidate_id"] for s in prediction["scores"]] != ids:
            raise ValueError("Saved candidate identity/order differs from the frozen input")
        scores = [item["score"] for item in prediction["scores"]]
        rows.append(ranking_metrics(ids, scores, qrels[sample["id"]], k=k))
        details = prediction.get("input_metadata", {})
        truncated_queries += int(details.get("input_truncated", False))
        truncated_candidates += sum(bool(value) for value in details.get("candidate_truncated", {}).values())
        candidate_count += len(ids)
    if len(rows) != len(records):
        raise ValueError("Input iterator ended before every prediction was verified")
    metrics = aggregate(rows)
    times = sorted(item["latency_ms"] for item in records)
    percentile = lambda fraction: times[min(len(times) - 1, max(0, math.ceil(fraction * len(times)) - 1))]
    metrics.update({"scope": protocol["scope"], "full_split_queries": protocol["full_split_queries"],
                    "candidates": candidate_count, "queries_with_truncation": truncated_queries,
                    "truncated_candidates": truncated_candidates,
                    "latency_ms": {"mean": sum(times) / len(times), "p50": percentile(0.5), "p95": percentile(0.95)},
                    "total_scoring_seconds": sum(times) / 1000,
                    "latency_scope": protocol["latency_scope"]})
    write_json(output / "metrics.json", metrics)
    protocol.update(status="completed", completed=len(records), updated_at=now(),
                    predictions_sha256=sha256(output / "predictions.jsonl"))
    write_json(output / "evaluation.json", protocol)
    title = "PILOT ONLY — not a full benchmark" if protocol["scope"] == "pilot" else "Full held-out pretrained reranker evaluation"
    lines = ["# " + title, "", f"Model: {protocol['model']}",
             f"Split: {protocol['split']}; scored {len(records)} of {protocol['full_split_queries']} queries.", "",
             "The frozen candidate set and full qrels are used. No failed or zero-recall queries are dropped.",
             "This is pretrained reranking, not a post-training improvement or a SOTA claim.", "",
             "| Metric | Value |", "|---|---:|"]
    lines += [f"| {key} | {value:.6f} |" for key, value in metrics.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    lines += ["", "Latency includes tokenization, all candidate forwards, CPU score transfer and device synchronization.",
              "Model load, explicit warmups, metric computation and disk writes are excluded; per-session warmups/load are logged separately.",
              "CUDA memory is a session peak; MPS memory samples are not peak measurements."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("completed", scope=protocol["scope"], output=str(output), metrics=metrics)


def run(args):
    import torch
    from cometa.model import QwenReranker
    if args.split == "train":
        raise ValueError("Use a held-out split; train candidates may contain gold injection")
    if args.k < 1 or args.threads < 1 or args.warmup_queries < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid k, threads, warmup count or pilot limit")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    manifest = load_manifest(args.manifest)
    spec = manifest["splits"][args.split]
    if not spec.get("qrels_path"):
        raise ValueError("This evaluator requires complete, separately hashed qrels")
    qrels = load_qrels(spec["qrels_path"])
    all_ids = [sample["id"] for sample in read_jsonl(spec["path"], require_labels=False)]
    if spec.get("count", len(all_ids)) != len(all_ids) or not all_ids:
        raise ValueError("Split count mismatch or empty split")
    query_ids = all_ids[:args.limit] if args.limit is not None else all_ids
    if any(qid not in qrels for qid in query_ids):
        raise ValueError("Missing full qrels for a requested query")
    def selected_samples():
        for ordinal, sample in enumerate(read_jsonl(spec["path"], require_labels=False)):
            if ordinal == len(query_ids):
                break
            yield sample
    emit("verifying_model_identity", model=args.model)
    model_identity = local_model_identity(args.model)
    parameters = {name: getattr(args, name) for name in ("device", "dtype", "backend", "layout", "max_length",
        "max_tree_tokens", "pair_batch_size", "threads", "warmup_queries", "revision", "k", "limit")}
    contract = {"version": VERSION, "manifest_sha256": manifest["_manifest_sha256"],
        "split": args.split, "split_sha256": spec["sha256"], "qrels_sha256": sha256(spec["qrels_path"]),
        "sample_ids": query_ids, "model": model_identity, "parameters": parameters,
        "runner_sha256": sha256(Path(__file__)), "cometa_source_sha256": source_identity(),
        "environment": environment(), "device_identity": device_identity(torch, args.device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
    contract_hash = digest_json(contract)
    output = Path(args.output).expanduser().resolve()
    existed = output.exists()
    if existed and not args.resume:
        raise FileExistsError("Output exists; use --resume only with the exact same protocol")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        if not existed and (output / "protocol.json").exists():
            raise RuntimeError("Output was initialized by another process; restart with --resume")
        if existed:
            stored = json.loads((output / "protocol.json").read_text())
            if stored != contract:
                raise ValueError("Resume protocol mismatch: inputs, model, code, environment or scoring parameters changed")
            protocol = json.loads((output / "evaluation.json").read_text())
        else:
            write_json(output / "protocol.json", contract)
            protocol = {"eval_id": uuid.uuid4().hex, "dataset_id": manifest["dataset_id"],
                "manifest_sha256": manifest["_manifest_sha256"], "split": args.split, "split_sha256": spec["sha256"],
                "qrels_sha256": sha256(spec["qrels_path"]), "model": model_identity["source"],
                "sample_ids": query_ids, "full_split_queries": len(all_ids), "scope": "pilot" if args.limit is not None else "full_split",
                "candidate_policy": "frozen_manifest_candidates_no_gold_injection", "gain": "linear_trec_eval", "k": args.k,
                "device": args.device, "dtype": args.dtype, "status": "initialized", "created_at": now(),
                "protocol_sha256": contract_hash, "environment": environment(),
                "latency_scope": "device_synchronized_per_query_tokenization_all_candidate_forward_and_cpu_scores_excludes_load_warmup_metrics_disk"}
            write_json(output / "evaluation.json", protocol)
        records = read_completed(output / "predictions.jsonl", query_ids, repair_partial_tail=args.resume)
        if protocol.get("status") == "completed" and protocol.get("predictions_sha256") != sha256(output / "predictions.jsonl"):
            raise ValueError("Completed prediction file hash mismatch")
        # Recheck saved candidate IDs and scores before resuming computation.
        for sample, record in zip(selected_samples(), records):
            ids = [candidate["id"] for candidate in sample["candidates"]]
            if [item["candidate_id"] for item in record["scores"]] != ids:
                raise ValueError("Saved candidate IDs/order mismatch")
            ranking_metrics(ids, [item["score"] for item in record["scores"]], qrels[sample["id"]], k=args.k)
        if len(records) == len(query_ids):
            finish(output, protocol, records, selected_samples(), qrels, args.k)
            return 0
        session = uuid.uuid4().hex
        protocol.update(status="loading_model", completed=len(records), updated_at=now())
        write_json(output / "evaluation.json", protocol)
        write_json(output / "progress.json", {"status": "loading_model", "completed": len(records),
            "total": len(query_ids), "scope": protocol["scope"], "updated_at": now()})
        active_query = None
        try:
            emit("loading_model", completed=len(records), total=len(query_ids), device=args.device, dtype=args.dtype)
            start = time.perf_counter()
            model = QwenReranker(model_identity["source"], layout=args.layout, backend=args.backend,
                device=args.device, dtype=args.dtype, revision=args.revision, max_length=args.max_length,
                max_tree_tokens=args.max_tree_tokens, pair_batch_size=args.pair_batch_size).eval()
            sync(torch, args.device)
            load_seconds = time.perf_counter() - start
            protocol["model_spec"] = {
                "source": model.source, "revision": model.revision, "layout": model.layout,
                "backend": model.backend, "max_length": model.max_length,
                "max_tree_tokens": model.max_tree_tokens, "pair_batch_size": model.pair_batch_size,
                "instruction": model.instruction, "yes_token_id": model.yes_id, "no_token_id": model.no_id,
                "compiler": model.compiler.spec(),
            }
            append_json(output / "sessions.jsonl", {"session": session, "time": now(), "event": "model_loaded",
                "model_load_seconds": load_seconds, "resumed_queries": len(records)})
            for index, sample in enumerate(selected_samples()):
                if index >= args.warmup_queries:
                    break
                sync(torch, args.device)
                start = time.perf_counter()
                model.score([sample])
                sync(torch, args.device)
                elapsed = 1000 * (time.perf_counter() - start)
                append_json(output / "warmups.jsonl", {"session": session, "id": sample["id"], "latency_ms": elapsed,
                    "included_in_aggregate_latency": False})
                emit("warmup", id=sample["id"], latency_ms=elapsed, excluded_from_scoring=True)
            if torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(args.device)
            protocol.update(status="running", updated_at=now())
            write_json(output / "evaluation.json", protocol)
            for ordinal, sample in enumerate(selected_samples()):
                if ordinal < len(records):
                    continue
                active_query = sample["id"]
                sync(torch, args.device)
                start = time.perf_counter()
                scores = model.score([sample])[0]
                sync(torch, args.device)
                elapsed = 1000 * (time.perf_counter() - start)
                result = decide(sample, scores, model_id=model_identity["source"])
                result.update(ordinal=ordinal, session=session, latency_ms=elapsed,
                    input_metadata=model.last_input_metadata[0], memory=memory(torch, args.device),
                    metrics=ranking_metrics([x["id"] for x in sample["candidates"]], scores, qrels[sample["id"]], k=args.k))
                append_json(output / "predictions.jsonl", result)
                records.append(result)
                progress = {"status": "running", "scope": protocol["scope"], "completed": len(records),
                    "total": len(query_ids), "last_query": active_query, "last_latency_ms": elapsed,
                    "total_scoring_seconds": sum(row["latency_ms"] for row in records) / 1000, "updated_at": now()}
                write_json(output / "progress.json", progress)
                emit("query_completed", **progress)
            finish(output, protocol, records, selected_samples(), qrels, args.k)
            write_json(output / "progress.json", {"status": "completed", "completed": len(records),
                "total": len(query_ids), "scope": protocol["scope"], "updated_at": now()})
            return 0
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            protocol.update(status=status, completed=len(records), failed_query=active_query, updated_at=now())
            write_json(output / "evaluation.json", protocol)
            write_json(output / "progress.json", {"status": status, "completed": len(records),
                "total": len(query_ids), "scope": protocol["scope"], "failed_query": active_query, "updated_at": now()})
            append_json(output / "failures.jsonl", {"session": session, "time": now(), "id": active_query,
                "completed": len(records), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
            emit(status, completed=len(records), total=len(query_ids), id=active_query,
                 error_type=type(exc).__name__, error=str(exc), resume_required=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True, help="Already downloaded local model directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--backend", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--layout", choices=["pairs", "tree"], default="pairs")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-tree-tokens", type=int)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--revision")
    parser.add_argument("--limit", type=int, help="Deterministic first-N pilot in a separate output directory")
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        emit("error", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
