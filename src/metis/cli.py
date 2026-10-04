"""Local CLI. All optional model imports occur only in model commands."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .api import Predictor, decide
from .artifacts import create_run, event, finish_run, seal_artifact, verify_artifact, write_json, environment
from .config import resolve_config
from .metrics import aggregate, load_qrels, ranking_metrics
from .registry import DatasetRegistry
from .schema import load_manifest, read_jsonl, sha256, validate_dataset, write_jsonl


def _json_print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _manifest_arg(value, registry):
    return Path(value) if Path(value).is_file() else DatasetRegistry(registry).resolve(value)


def run_evaluation(args):
    manifest = load_manifest(_manifest_arg(args.manifest, args.registry))
    if manifest.get("task_kind", "ranking") != "ranking":
        raise ValueError("This benchmark evaluator currently supports ranking; other task scores use predict")
    if args.split not in manifest["splits"]:
        raise ValueError(f"Unknown split: {args.split}")
    if args.split == "train":
        raise ValueError("Benchmark evaluation uses validation/test; training candidates may contain injected positives")
    if args.k < 1:
        raise ValueError("k must be positive")
    spec = manifest["splits"][args.split]
    samples = list(read_jsonl(spec["path"], require_labels=False))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("Empty evaluation split")
    qrels = load_qrels(spec["qrels_path"]) if spec.get("qrels_path") else {
        s["id"]: s["supervision"]["labels"] for s in samples}
    if any(s["id"] not in qrels for s in samples):
        raise ValueError("Missing complete qrels for an evaluation query")
    backend, bm25_runs = None, None
    model_id = args.baseline or args.artifact or args.model
    if args.baseline:
        score_path = spec.get("bm25_path") if args.baseline == "bm25" else spec.get("retrieval_path", spec.get("bm25_path"))
        if not score_path:
            raise ValueError("No frozen retrieval scores in manifest")
        bm25_runs = {}
        for line in Path(score_path).read_text().splitlines():
            row = json.loads(line)
            if row["id"] in bm25_runs:
                raise ValueError("Duplicate retrieval query ID")
            bm25_runs[row["id"]] = row["scores"]
    elif args.artifact:
        backend = Predictor(args.artifact, device=args.device, dtype=args.dtype)
    else:
        from .model import QwenReranker
        backend = QwenReranker(args.model, layout=args.layout, backend=args.backend,
                              device=args.device, dtype=args.dtype, max_length=args.max_length,
                              revision=args.revision, pair_batch_size=args.pair_batch_size,
                              max_tree_tokens=args.max_tree_tokens)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"eval_id": uuid.uuid4().hex, "dataset_id": manifest["dataset_id"],
                "manifest_sha256": manifest["_manifest_sha256"], "split": args.split,
                "split_sha256": spec["sha256"], "qrels_sha256": sha256(spec["qrels_path"]) if spec.get("qrels_path") else None,
                "model": str(model_id), "sample_ids": [s["id"] for s in samples],
                "candidate_policy": "frozen_manifest_candidates_no_gold_injection",
                "gain": "linear_trec_eval", "k": args.k, "latency_scope": "per_query_scoring_including_tokenization_not_model_load",
                "device": args.device, "dtype": args.dtype, "environment": environment(),
                "artifact_manifest_sha256": sha256(Path(args.artifact)/"artifact.json") if args.artifact else None,
                "status": "running"}
    if backend is not None:
        model = backend.model if args.artifact else backend
        protocol["model_spec"] = {key: getattr(model, key) for key in
            ("source", "revision", "layout", "backend", "max_length", "max_tree_tokens", "instruction", "pair_batch_size")}
        protocol["input_spec"] = model.compiler.spec()
    else:
        protocol["retrieval_sha256"] = sha256(score_path)
        protocol["latency_scope"] = "frozen_score_lookup_and_decision_only_excludes_retriever_execution"
    write_json(output / "evaluation.json", protocol)
    rows, times, failures = [], [], []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as sink:
        for sample in samples:
            ids = [c["id"] for c in sample["candidates"]]
            start = time.perf_counter()
            try:
                if bm25_runs is not None:
                    scores = [bm25_runs[sample["id"]][cid] for cid in ids]
                    prediction = decide(sample, scores, model_id=model_id)
                    prediction["score_semantics"] = "bm25" if args.baseline == "bm25" else "retriever_score"
                elif args.artifact:
                    prediction = backend.predict([sample])[0]
                    scores = [x["score"] for x in prediction["scores"]]
                else:
                    scores = backend.score([sample])[0]
                    prediction = decide(sample, scores, model_id=model_id)
                    metadata = getattr(backend, "last_input_metadata", [])
                    if metadata:
                        prediction["input_metadata"] = metadata[0]
                elapsed = 1000 * (time.perf_counter() - start)
                prediction["latency_ms"] = elapsed
                times.append(elapsed)
                metrics = ranking_metrics(ids, scores, qrels[sample["id"]], k=args.k)
                prediction["metrics"] = metrics
                rows.append(metrics)
                sink.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                sink.flush()
            except Exception as exc:
                failures.append({"id": sample["id"], "error_type": type(exc).__name__, "error": str(exc)})
    write_json(output / "failures.json", failures)
    if failures:
        protocol.update(status="failed", failures=len(failures), completed=len(rows))
        write_json(output / "evaluation.json", protocol)
        raise RuntimeError(f"{len(failures)} evaluation queries failed; no success-only aggregate reported. See {output}")
    metrics = aggregate(rows)
    ordered = sorted(times)
    metrics["latency_ms"] = {"mean": sum(times) / len(times), "p50": ordered[len(ordered)//2],
                             "p95": ordered[min(len(ordered)-1, int(len(ordered)*0.95))]}
    write_json(output / "metrics.json", metrics)
    protocol.update(status="completed", completed=len(rows))
    write_json(output / "evaluation.json", protocol)
    report = ["# Metis evaluation", "", f"Dataset: {manifest['dataset_id']}; split: {args.split}",
              f"Model: {model_id}", "", "Full qrels are used: unretrieved positives count as misses.",
              "This report is a measured run, not a claim of leaderboard superiority.", "", "| Metric | Value |", "|---|---:|"]
    report.extend(f"| {key} | {value:.6f} |" for key, value in metrics.items() if isinstance(value, (float, int)))
    (output / "report.md").write_text("\n".join(report) + "\n")
    return {"output": str(output), "metrics": metrics}


def compare_evaluations(baseline, candidate, *, metric="ndcg@10", max_drop=0.01):
    if max_drop < 0:
        raise ValueError("max_drop must be nonnegative")
    left, right = Path(baseline), Path(candidate)
    lp = json.loads((left / "evaluation.json").read_text())
    rp = json.loads((right / "evaluation.json").read_text())
    keys = ("manifest_sha256", "split", "split_sha256", "qrels_sha256", "sample_ids", "candidate_policy", "gain", "k")
    if lp.get("status") != "completed" or rp.get("status") != "completed":
        raise ValueError("Cannot compare incomplete evaluations")
    if any(lp.get(key) != rp.get(key) for key in keys):
        raise ValueError("Evaluation protocols differ; compare the same queries/candidates/qrels")
    if lp.get("input_spec") is not None and rp.get("input_spec") is not None and lp["input_spec"] != rp["input_spec"]:
        raise ValueError("Model input protocols differ; use the same prompt and truncation budget")
    a = json.loads((left / "metrics.json").read_text())[metric]
    b = json.loads((right / "metrics.json").read_text())[metric]
    return {"metric": metric, "baseline": a, "candidate": b, "delta": b-a,
            "max_absolute_drop": max_drop, "passed": b-a >= -max_drop,
            "selection_allowed": rp["split"] in {"validation", "dev"}}


def _serve(args):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    predictor = Predictor(args.artifact, device=args.device, dtype=args.dtype)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"status": "ready", "artifact": str(args.artifact)}
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            if self.path != "/v1/decisions":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Request must contain 1..2,000,000 bytes")
                body = json.loads(self.rfile.read(size))
                result = {"request_id": body.get("request_id", uuid.uuid4().hex),
                          "results": predictor.predict(body["samples"], **body.get("policy", {}))}
                code = 200
            except (ValueError, KeyError, TypeError) as exc:
                result, code = {"status": "invalid_request", "error": str(exc)}, 400
            except Exception as exc:
                result, code = {"status": "inference_error", "error": type(exc).__name__}, 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
    print(f"Local development server http://{args.host}:{args.port}; one worker")
    HTTPServer((args.host, args.port), Handler).serve_forever()


def parser():
    p = argparse.ArgumentParser(prog="metis", description="Local task-specific decision model kit")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-nfcorpus")
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--top-k", type=int, default=50)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--retrieval-run")
    retrieve = sub.add_parser("retrieve-nfcorpus")
    retrieve.add_argument("--data-dir", required=True); retrieve.add_argument("--output", required=True)
    retrieve.add_argument("--top-k", type=int, default=100)
    retrieve.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    retrieve.add_argument("--revision"); retrieve.add_argument("--device", default="cpu")
    retrieve.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    retrieve.add_argument("--batch-size", type=int, default=8)
    retrieve.add_argument("--max-length", type=int, default=8192)
    ds = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    reg = ds.add_parser("register")
    reg.add_argument("name"); reg.add_argument("manifest")
    reg.add_argument("--registry", default=".metis/datasets.json")
    reg.add_argument("--replace", action="store_true")
    listing = ds.add_parser("list")
    listing.add_argument("--registry", default=".metis/datasets.json")
    val = sub.add_parser("validate"); val.add_argument("manifest")
    val.add_argument("--registry", default=".metis/datasets.json")
    train = sub.add_parser("train"); train.add_argument("config")
    train.add_argument("--resume", default=None)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--manifest", required=True); ev.add_argument("--split", default="test")
    ev.add_argument("--registry", default=".metis/datasets.json")
    ev.add_argument("--output", required=True); ev.add_argument("--limit", type=int)
    ev.add_argument("--k", type=int, default=10)
    modes = ev.add_mutually_exclusive_group(required=True)
    modes.add_argument("--baseline", choices=["bm25", "retrieval"])
    modes.add_argument("--model"); modes.add_argument("--artifact")
    ev.add_argument("--layout", choices=["pairs", "tree"], default="pairs")
    ev.add_argument("--backend", choices=["eager", "sdpa", "flex"], default="eager")
    ev.add_argument("--max-length", type=int, default=4096)
    ev.add_argument("--max-tree-tokens", type=int)
    ev.add_argument("--pair-batch-size", type=int, default=8)
    ev.add_argument("--revision")
    for cmd in (ev,):
        cmd.add_argument("--device", default="cpu"); cmd.add_argument("--dtype", default="float32")
    pred = sub.add_parser("predict"); pred.add_argument("--artifact", required=True)
    pred.add_argument("--input", required=True); pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cpu"); pred.add_argument("--dtype", default="float32")
    pred.add_argument("--top-k", type=int, default=None)
    export = sub.add_parser("export"); export.add_argument("--artifact", required=True)
    export.add_argument("--output", required=True)
    comp = sub.add_parser("compare"); comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True); comp.add_argument("--metric", default="ndcg@10")
    comp.add_argument("--max-drop", type=float, default=0.01)
    serv = sub.add_parser("serve"); serv.add_argument("--artifact", required=True)
    serv.add_argument("--host", default="127.0.0.1"); serv.add_argument("--port", type=int, default=8080)
    serv.add_argument("--device", default="cpu"); serv.add_argument("--dtype", default="float32")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-nfcorpus":
            from .benchmarks import nfcorpus
            data_dir = nfcorpus.fetch(args.data_dir) if args.download else args.data_dir
            result = {"manifest": str(nfcorpus.prepare(data_dir, args.output, top_k=args.top_k, retrieval_run=args.retrieval_run))}
        elif args.command == "retrieve-nfcorpus":
            from .benchmarks.retrieval import build_dense_run
            result = {"retrieval_run": str(build_dense_run(args.data_dir, args.output, top_k=args.top_k,
                model_name_or_path=args.model, revision=args.revision, device=args.device, dtype=args.dtype,
                batch_size=args.batch_size, max_length=args.max_length))}
        elif args.command == "dataset":
            registry = DatasetRegistry(args.registry)
            result = registry.register(args.name, args.manifest, replace=args.replace) if args.action == "register" else registry.entries()
        elif args.command == "validate":
            result = validate_dataset(_manifest_arg(args.manifest, args.registry))
        elif args.command == "train":
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise ValueError("CLI v0.1 currently supports one process; distributed run coordination is not yet implemented")
            cfg, manifest = resolve_config(args.config, {"training": {"resume_from_checkpoint": args.resume}} if args.resume else None)
            validate_dataset(cfg["data"]["manifest"])
            run = create_run(cfg["output"]["root"], cfg["output"]["project"], cfg)
            write_json(run / "data.manifest.json", manifest)
            try:
                from .training import train
                exported = Path(train(cfg, run))
                # Keep both final and validation-selected exports independently loadable.
                for artifact_dir in sorted((run / "exports").iterdir()):
                    if artifact_dir.is_dir() and (artifact_dir / "model_spec.json").is_file():
                        seal_artifact(artifact_dir, task=cfg["task"], source_run=run.name)
                        verify_artifact(artifact_dir)
                finish_run(run, artifact=str(exported.relative_to(run)))
                result = {"run": str(run), "artifact": str(exported)}
            except Exception as exc:
                finish_run(run, status="failed", error_type=type(exc).__name__, error=str(exc))
                raise
        elif args.command == "evaluate":
            result = run_evaluation(args)
        elif args.command == "predict":
            if Path(args.output).exists():
                raise ValueError("Refusing to overwrite predictions")
            predictor = Predictor(args.artifact, device=args.device, dtype=args.dtype)
            samples = list(read_jsonl(args.input, require_labels=False))
            write_jsonl(args.output, predictor.predict(samples, top_k=args.top_k))
            result = {"output": args.output, "samples": len(samples)}
        elif args.command == "export":
            verify_artifact(args.artifact)
            if Path(args.output).exists():
                raise ValueError("Export destination already exists")
            shutil.copytree(args.artifact, args.output)
            verify_artifact(args.output)
            result = {"artifact": str(Path(args.output).resolve()), "operation": "portable_export_copy"}
        elif args.command == "compare":
            result = compare_evaluations(args.baseline, args.candidate, metric=args.metric, max_drop=args.max_drop)
            _json_print(result)
            return 0 if result["passed"] else 2
        elif args.command == "serve":
            _serve(args)
            return 0
        _json_print(result)
        return 0
    except Exception as exc:
        print(f"metis: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
