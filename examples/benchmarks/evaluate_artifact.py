#!/usr/bin/env python3
"""Evaluate a sealed ranking artifact on a frozen held-out manifest.

No downloads or training. --limit is always a pilot. Resume requires the same
artifact, local base dependency, data, source, environment and scoring settings.
Input template, readout and token budget come from the artifact without overrides.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import os
import time
import traceback
import uuid

import evaluate_pretrained as common
from cometa.artifacts import environment, verify_artifact, write_json
from cometa.metrics import aggregate, load_qrels, ranking_metrics
from cometa.schema import load_manifest, read_jsonl, sha256

VERSION = "metis-artifact-eval-v1"
METRIC_DEFINITION = {
    "version": "full-qrels-linear-v1", "k": 10, "gain": "linear_trec_eval",
    "ndcg": "dcg@10 / ideal_dcg@10 over complete qrels",
    "map": "AP over all supplied candidates divided by all qrels positives",
    "mrr": "first positive reciprocal rank over all supplied candidates",
    "recall": "top10 hits / all qrels positives",
    "candidate_recall": "all candidate hits / all qrels positives",
    "unjudged": "zero evaluation gain, not a human negative judgment",
    "ties": "descending score then ascending candidate ID",
    "aggregation": "macro mean over every requested query including zero-recall queries",
}
LATENCY_SCOPE = "device_synchronized_per_query_tokenization_all_candidate_forward_and_cpu_scores_excludes_load_warmup_metrics_disk"


def autocast_context(torch, device, precision):
    if precision == "none":
        return nullcontext()
    if precision != "bf16" or torch.device(device).type != "cuda":
        raise ValueError("--precision bf16 requires CUDA; use none on CPU/MPS")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def artifact_identity(path):
    root = Path(path).expanduser().resolve()
    sealed = verify_artifact(root)
    if sealed.get("task", {}).get("kind") != "ranking":
        raise ValueError("This evaluator requires a sealed ranking artifact")
    spec = json.loads((root / "model_spec.json").read_text())
    inputs = json.loads((root / "input_spec.json").read_text())
    if type(inputs.get("max_length")) is not int or inputs["max_length"] < 1:
        raise ValueError("Artifact input_spec must declare a positive max_length")
    dependency = None
    if spec.get("tuning") == "lora":
        # The adapter alone is not the identity of a LoRA model. Force an
        # already local base and seal its contents in the evaluation contract.
        dependency = common.local_model_identity(spec["source"])
        expected = spec.get("base_dependency_sha256")
        if not expected or any(dependency["files"].get(name, {}).get("sha256") != value
                               for name, value in expected.items()):
            raise ValueError("LoRA artifact needs matching recorded local base file hashes")
    files = dict(sealed["files"])
    files["artifact.json"] = sha256(root / "artifact.json")
    return {"path": str(root), "files_sha256": files,
            "content_sha256": common.digest_json(files), "base_dependency": dependency,
            "model_spec": spec, "input_spec": inputs}


def validate_prediction(sample, record, qrels):
    ids = [item["id"] for item in sample["candidates"]]
    if (record.get("id") != sample["id"] or record.get("status") != "ok"
            or record.get("coverage") != 1.0
            or [item.get("candidate_id") for item in record.get("scores", [])] != ids):
        raise ValueError("Prediction must cover the exact ordered candidate IDs")
    values = [item["score"] for item in record["scores"]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError("Every candidate must have one finite numeric score")
    calculated = ranking_metrics(ids, values, qrels[sample["id"]], k=10)
    if "metrics" in record:
        for key, value in calculated.items():
            actual = record["metrics"].get(key)
            if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                    or not math.isfinite(actual)
                    or not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"Saved per-query metric differs from full qrels: {key}")
    return calculated


def finish(output, protocol, records, samples, qrels):
    if len(records) != len(samples) or len(records) != len(protocol["sample_ids"]):
        raise ValueError("No aggregate is allowed for an incomplete run")
    rows = [validate_prediction(sample, record, qrels) for sample, record in zip(samples, records)]
    times = sorted(record["latency_ms"] for record in records)
    percentile = lambda p: times[min(len(times) - 1, max(0, math.ceil(p * len(times)) - 1))]
    truncation_available = all(isinstance(row.get("input_metadata", {}).get("candidate_truncated"), dict)
                               for row in records)
    metrics = aggregate(rows)
    metrics.update(scope=protocol["scope"], full_split_queries=protocol["full_split_queries"],
                   candidates=sum(len(s["candidates"]) for s in samples),
                   latency_ms={"mean": math.fsum(times) / len(times), "p50": percentile(.5), "p95": percentile(.95)},
                   total_scoring_seconds=math.fsum(times) / 1000, latency_scope=LATENCY_SCOPE,
                   truncation_metadata_available=truncation_available,
                   queries_with_truncation=(sum(any(r["input_metadata"]["candidate_truncated"].values())
                                                for r in records) if truncation_available else None),
                   truncated_candidates=(sum(sum(bool(v) for v in r["input_metadata"]["candidate_truncated"].values())
                                             for r in records) if truncation_available else None))
    memory_keys = {key for r in records for key, value in r.get("memory", {}).items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)}
    metrics["memory"] = {key: max(r["memory"][key] for r in records if key in r.get("memory", {}))
                         for key in sorted(memory_keys)}
    metrics["memory_scope"] = "CUDA session peak after warmup, including resident model; maxima across resumed sessions. MPS values are sampled, not peaks."
    write_json(output / "metrics.json", metrics)
    protocol.update(status="completed", completed=len(records), updated_at=common.now(),
                    predictions_sha256=sha256(output / "predictions.jsonl"))
    write_json(output / "evaluation.json", protocol)
    write_json(output / "progress.json", {"status": "completed", "completed": len(records),
               "total": len(samples), "scope": protocol["scope"], "updated_at": common.now()})
    title = "PILOT — not a full benchmark" if protocol["scope"] == "pilot" else "Full held-out artifact evaluation"
    lines = [f"# {title}", "", f"Artifact: {protocol['model']}",
             f"Split: {protocol['split']}; {len(records)} / {protocol['full_split_queries']} queries.", "",
             "Every frozen candidate is scored. Full qrels include unretrieved positives; zero-recall queries remain.",
             "Input compiler/readout and model identity are recorded as exported; this is not a SOTA claim.", "",
             "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.6f} |" for key, value in metrics.items()
                 if isinstance(value, (int, float)) and not isinstance(value, bool))
    lines += ["", "Latency includes tokenization, all candidate forwards, device synchronization and CPU score transfer.",
              "Load and warmup are logged separately. Metrics and disk writes are outside scoring latency.",
              metrics["memory_scope"], "No incomplete run receives a success-only aggregate."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.emit("completed", output=str(output), scope=protocol["scope"], metrics=metrics)


def run(args):
    if args.split not in {"validation", "dev", "test"}:
        raise ValueError("Only held-out validation/dev/test is allowed")
    if args.threads < 1 or args.warmup_queries < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid threads, warmup count or limit")
    # A sealed artifact is evaluated with local files only; missing dependencies
    # fail explicitly instead of silently downloading a different model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from cometa.api import Predictor
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    # Weight dtype and execution precision are distinct. In particular,
    # --dtype float32 --precision bf16 preserves FP32 weights and uses BF16
    # autocast for forwards, matching the training checkpoint-selection path.
    if args.precision == "bf16":
        if torch.device(args.device).type != "cuda":
            raise ValueError("BF16 autocast evaluation requires a CUDA device")
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested CUDA device does not support BF16")
    manifest = load_manifest(args.manifest)
    if manifest.get("task_kind", "ranking") != "ranking":
        raise ValueError("Manifest task_kind must be ranking")
    split = manifest["splits"][args.split]
    if not split.get("qrels_path"):
        raise ValueError("A complete separately hashed qrels file is required")
    qrels = load_qrels(split["qrels_path"])
    all_samples = list(read_jsonl(split["path"], require_labels=False))
    if not all_samples or split.get("count") != len(all_samples):
        raise ValueError("Full manifest split count mismatch")
    if any(s["id"] not in qrels or s.get("metadata", {}).get("injected_positive_ids") for s in all_samples):
        raise ValueError("Every held-out query needs full qrels and no injected positive")
    samples = all_samples[:args.limit] if args.limit is not None else all_samples
    query_ids = [s["id"] for s in samples]
    identity = artifact_identity(args.artifact)
    parameters = {key: getattr(args, key) for key in
                  ("device", "dtype", "precision", "threads", "warmup_queries", "limit", "split")}
    contract = {"version": VERSION, "manifest_sha256": manifest["_manifest_sha256"],
                "split_sha256": split["sha256"], "qrels_sha256": sha256(split["qrels_path"]),
                "sample_ids": query_ids, "full_split_queries": len(all_samples),
                "candidate_ids_sha256": common.digest_json([[s["id"], [c["id"] for c in s["candidates"]]] for s in samples]),
                "artifact": identity, "parameters": parameters,
                "runner_sha256": sha256(Path(__file__)), "helper_sha256": sha256(Path(common.__file__)),
                "cometa_source_sha256": common.source_identity(), "metric_definition": METRIC_DEFINITION,
                "environment": environment(), "device_identity": common.device_identity(torch, args.device),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
    # JSON round-trip ensures saved JSON list/tuple and map semantics are exact.
    contract = json.loads(json.dumps(contract, allow_nan=False))
    contract_hash = common.digest_json(contract)
    output = Path(args.output).expanduser().resolve()
    existed = output.exists()
    if existed and not args.resume:
        raise FileExistsError("Output exists; --resume requires the identical run contract")
    output.mkdir(parents=True, exist_ok=True)
    with common.output_lock(output):
        if existed:
            stored = json.loads((output / "protocol.json").read_text())
            if stored != contract:
                raise ValueError("Resume mismatch: artifact/base/data/source/environment/scoring settings changed")
            protocol = json.loads((output / "evaluation.json").read_text())
            if protocol.get("protocol_sha256") != contract_hash or protocol.get("sample_ids") != query_ids:
                raise ValueError("Saved evaluation does not match the immutable contract")
        else:
            if (output / "protocol.json").exists():
                raise RuntimeError("Output was concurrently initialized")
            write_json(output / "protocol.json", contract)
            protocol = {"eval_id": uuid.uuid4().hex, "runner_version": VERSION,
                        "dataset_id": manifest["dataset_id"], "manifest_sha256": manifest["_manifest_sha256"],
                        "split": args.split, "split_sha256": split["sha256"], "qrels_sha256": sha256(split["qrels_path"]),
                        "model": identity["path"], "artifact_manifest_sha256": identity["files_sha256"]["artifact.json"],
                        "sample_ids": query_ids, "full_split_queries": len(all_samples),
                        "scope": "pilot" if args.limit is not None else "full_split",
                        "candidate_policy": "frozen_manifest_candidates_no_gold_injection",
                        "gain": "linear_trec_eval", "k": 10, "metric_definition": METRIC_DEFINITION,
                        "model_spec": identity["model_spec"], "input_spec": identity["input_spec"],
                        "device": args.device, "dtype": args.dtype, "precision": args.precision,
                        "environment": environment(),
                        "status": "initialized", "completed": 0, "created_at": common.now(),
                        "protocol_sha256": contract_hash, "latency_scope": LATENCY_SCOPE}
            write_json(output / "evaluation.json", protocol)
        expected_fields = {"manifest_sha256": contract["manifest_sha256"], "split_sha256": contract["split_sha256"],
                           "qrels_sha256": contract["qrels_sha256"], "split": args.split,
                           "dataset_id": manifest["dataset_id"], "protocol_sha256": contract_hash,
                           "artifact_manifest_sha256": identity["files_sha256"]["artifact.json"],
                           "model": identity["path"], "model_spec": identity["model_spec"],
                           "input_spec": identity["input_spec"], "device": args.device,
                           "dtype": args.dtype, "precision": args.precision,
                           "scope": "pilot" if args.limit is not None else "full_split",
                           "full_split_queries": len(all_samples), "metric_definition": METRIC_DEFINITION,
                           "gain": "linear_trec_eval", "k": 10, "sample_ids": query_ids,
                           "latency_scope": LATENCY_SCOPE,
                           "candidate_policy": "frozen_manifest_candidates_no_gold_injection"}
        if any(protocol.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("Saved evaluation metadata differs from the immutable run contract")
        records = common.read_completed(output / "predictions.jsonl", query_ids, repair_partial_tail=args.resume)
        if protocol.get("status") == "completed" and protocol.get("predictions_sha256") != sha256(output / "predictions.jsonl"):
            raise ValueError("Completed predictions hash mismatch")
        for sample, record in zip(samples, records):
            validate_prediction(sample, record, qrels)
        if len(records) == len(samples):
            finish(output, protocol, records, samples, qrels)
            return 0
        session = uuid.uuid4().hex
        active_query = None
        try:
            protocol.update(status="loading_model", completed=len(records))
            write_json(output / "evaluation.json", protocol)
            common.emit("loading_artifact", artifact=identity["path"], completed=len(records), total=len(samples))
            start = time.perf_counter()
            predictor = Predictor(identity["path"], device=args.device, dtype=args.dtype)
            predictor.model.eval()
            if predictor.model.compiler.spec() != identity["input_spec"]:
                raise ValueError("Loaded compiler differs from the sealed input protocol")
            common.sync(torch, args.device)
            common.append_json(output / "sessions.jsonl", {"session": session, "time": common.now(),
                "event": "artifact_loaded", "model_load_seconds": time.perf_counter() - start,
                "resumed_queries": len(records)})
            for sample in samples[:args.warmup_queries]:
                common.sync(torch, args.device)
                start = time.perf_counter()
                with autocast_context(torch, args.device, args.precision):
                    warmup = predictor.predict([sample])
                common.sync(torch, args.device)
                elapsed = (time.perf_counter() - start) * 1000
                if len(warmup) != 1:
                    raise ValueError("Warmup predictor lost the query")
                validate_prediction(sample, warmup[0], qrels)
                common.append_json(output / "warmups.jsonl", {"session": session, "id": sample["id"],
                    "latency_ms": elapsed, "included_in_aggregate_latency": False})
            if torch.device(args.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(args.device)
            protocol.update(status="running")
            write_json(output / "evaluation.json", protocol)
            for ordinal, sample in enumerate(samples):
                if ordinal < len(records):
                    continue
                active_query = sample["id"]
                common.sync(torch, args.device)
                start = time.perf_counter()
                with autocast_context(torch, args.device, args.precision):
                    batch = predictor.predict([sample])
                common.sync(torch, args.device)
                elapsed = (time.perf_counter() - start) * 1000
                if len(batch) != 1:
                    raise ValueError("Predictor lost the query")
                prediction = batch[0]
                prediction["metrics"] = validate_prediction(sample, prediction, qrels)
                prediction.update(ordinal=ordinal, session=session, latency_ms=elapsed,
                                  memory=common.memory(torch, args.device))
                common.append_json(output / "predictions.jsonl", prediction)
                records.append(prediction)
                progress = {"status": "running", "scope": protocol["scope"], "completed": len(records),
                            "total": len(samples), "last_query": active_query, "last_latency_ms": elapsed,
                            "updated_at": common.now()}
                write_json(output / "progress.json", progress)
                common.emit("query_completed", **progress)
            finish(output, protocol, records, samples, qrels)
            return 0
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            protocol.update(status=status, completed=len(records), failed_query=active_query, updated_at=common.now())
            write_json(output / "evaluation.json", protocol)
            write_json(output / "progress.json", {"status": status, "completed": len(records), "total": len(samples),
                       "scope": protocol["scope"], "failed_query": active_query, "updated_at": common.now()})
            common.append_json(output / "failures.jsonl", {"session": session, "id": active_query,
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
            common.emit(status, completed=len(records), error_type=type(exc).__name__, error=str(exc))
            return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", required=True, help="Sealed local export; local base required for LoRA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "dev", "test"], default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--precision", choices=["none", "bf16"], default="none",
                        help="Autocast policy, independent of stored/loaded weight dtype")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N queries; always explicitly PILOT")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        common.emit("error", error_type=type(exc).__name__, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
