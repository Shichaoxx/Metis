#!/usr/bin/env python3
"""Compare sealed artifact/pretrained evaluation reports without model imports.

Different compilers and readouts are allowed and disclosed as different model
methods. Dataset, ordered queries/candidates, metric implementation and token
budget must match. Both pilots remain pilots; mixing pilot/full is rejected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from compare_results import METRICS, TOLERANCE, finite_number, paired_bootstrap, sha256

VERSION = "metis-task-method-comparison-v1"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing/invalid SHA-256: {name}")
    return value


def read_evaluation(directory):
    root = Path(directory).expanduser().resolve()
    names = ("evaluation.json", "protocol.json", "metrics.json", "predictions.jsonl")
    evaluation = json.loads((root / "evaluation.json").read_text())
    contract = json.loads((root / "protocol.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    if evaluation.get("status") != "completed":
        raise ValueError("Both evaluations must be completed; partial successes are invalid")
    if evaluation.get("protocol_sha256") != digest_json(contract):
        raise ValueError("Evaluation contract hash mismatch")
    for key in ("manifest_sha256", "split_sha256", "qrels_sha256"):
        require_hash(evaluation.get(key), key)
        if contract.get(key) != evaluation[key]:
            raise ValueError(f"Evaluation and immutable contract disagree: {key}")
    query_ids = evaluation.get("sample_ids")
    if (not isinstance(query_ids, list) or not query_ids
            or any(not isinstance(q, str) or not q for q in query_ids)
            or len(set(query_ids)) != len(query_ids) or contract.get("sample_ids") != query_ids):
        raise ValueError("Need unique ordered query IDs matching the immutable contract")
    if evaluation.get("completed") != len(query_ids) or metrics.get("queries") != len(query_ids):
        raise ValueError("Completion/metric counts differ from requested queries")
    scope = evaluation.get("scope")
    full_count = evaluation.get("full_split_queries")
    if scope not in {"pilot", "full_split"} or type(full_count) is not int or full_count < len(query_ids):
        raise ValueError("Need explicit pilot/full_split scope and complete split size")
    if scope == "full_split" and full_count != len(query_ids):
        raise ValueError("Full-split evaluation must cover every query")
    if metrics.get("scope") != scope or metrics.get("full_split_queries") != full_count:
        raise ValueError("Metric scope/count differs from the evaluation")
    if "full_split_queries" in contract and contract["full_split_queries"] != full_count:
        raise ValueError("Full split size differs from immutable contract")
    limit = contract.get("parameters", {}).get("limit")
    if scope == "pilot" and (type(limit) is not int or limit < 1 or len(query_ids) != min(limit, full_count)):
        raise ValueError("Pilot scope needs matching explicit limit evidence")
    if scope == "full_split" and limit is not None:
        raise ValueError("A limit run cannot be declared full_split")
    if evaluation.get("split") not in {"validation", "dev", "test"}:
        raise ValueError("Only held-out splits are supported")
    if not isinstance(evaluation.get("dataset_id"), str) or not evaluation["dataset_id"]:
        raise ValueError("Missing dataset ID")
    if evaluation.get("k") != 10 or evaluation.get("gain") != "linear_trec_eval":
        raise ValueError("Comparison requires explicit linear/TREC nDCG@10")
    if evaluation.get("candidate_policy") != "frozen_manifest_candidates_no_gold_injection":
        raise ValueError("Held-out candidate set must be frozen and have no gold injection")
    metric_source = contract.get("cometa_source_sha256", {}).get("metrics.py")
    require_hash(metric_source, "cometa metrics.py source")
    definition = evaluation.get("metric_definition")
    if definition != contract.get("metric_definition"):
        raise ValueError("Metric definition differs from immutable contract")
    if definition is not None and (definition.get("k") != 10 or definition.get("gain") != "linear_trec_eval"):
        raise ValueError("Declared metric definition conflicts with K/gain")
    prediction_hash = require_hash(evaluation.get("predictions_sha256"), "predictions_sha256")
    if prediction_hash != sha256(root / "predictions.jsonl"):
        raise ValueError("Completed prediction file hash mismatch")
    records = {}
    for ordinal, line in enumerate((root / "predictions.jsonl").read_text().splitlines()):
        row = json.loads(line)
        if ordinal >= len(query_ids) or row.get("id") != query_ids[ordinal] or row.get("ordinal") != ordinal:
            raise ValueError("Predictions must follow the complete ordered query protocol")
        if row.get("status") != "ok" or row.get("coverage") != 1.0:
            raise ValueError("Prediction status/coverage is not a complete success")
        scores = row.get("scores")
        if not isinstance(scores, list) or not scores:
            raise ValueError("Missing candidate scores")
        ids = [item.get("candidate_id") for item in scores]
        if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique nonempty strings")
        for item in scores:
            finite_number(item.get("score"), "candidate score")
        for metric in METRICS:
            value = finite_number(row.get("metrics", {}).get(metric), f"per-query {metric}")
            if not 0 <= value <= 1 + TOLERANCE:
                raise ValueError(f"Metric out of bounds: {metric}")
        if type(row["metrics"].get("judged_positive_count")) is not int or row["metrics"]["judged_positive_count"] < 0:
            raise ValueError("Missing complete qrels positive denominator")
        records[row["id"]] = row
    if list(records) != query_ids:
        raise ValueError("Predictions do not cover all ordered query IDs")
    for metric in METRICS:
        mean = math.fsum(records[qid]["metrics"][metric] for qid in query_ids) / len(query_ids)
        if not math.isclose(mean, finite_number(metrics.get(metric), metric), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Aggregate {metric} differs from all-query mean")
    if metrics.get("candidates") != sum(len(r["scores"]) for r in records.values()):
        raise ValueError("Aggregate candidate count mismatch")
    input_spec = evaluation.get("input_spec") or evaluation.get("model_spec", {}).get("compiler")
    if not isinstance(input_spec, dict) or type(input_spec.get("max_length")) is not int or input_spec["max_length"] < 1:
        raise ValueError("Both methods must record their compiler and token budget")
    # For artifact evaluations input semantics are sealed before loading. The
    # historic pretrained runner records compiler metadata after model loading.
    if "artifact" in contract and contract["artifact"].get("input_spec") != input_spec:
        raise ValueError("Compiler differs from sealed artifact contract")
    if "artifact" in contract and contract["artifact"].get("model_spec") != evaluation.get("model_spec"):
        raise ValueError("Model specification differs from sealed artifact contract")
    return {"directory": str(root), "protocol": evaluation, "contract": contract, "metrics": metrics,
            "records": records, "metric_source_sha256": metric_source, "input_spec": input_spec,
            "files_sha256": {name: sha256(root / name) for name in names}}


def method(item):
    protocol = item["protocol"]
    return {"model": protocol.get("model"), "directory": item["directory"],
            "model_spec": protocol.get("model_spec"), "input_spec": item["input_spec"],
            "device": protocol.get("device"), "dtype": protocol.get("dtype"),
            "precision": protocol.get("precision", "none"),
            "files_sha256": item["files_sha256"]}


def compare(baseline_directory, candidate_directory):
    left, right = read_evaluation(baseline_directory), read_evaluation(candidate_directory)
    a, b = left["protocol"], right["protocol"]
    keys = ("dataset_id", "manifest_sha256", "split", "split_sha256", "qrels_sha256", "sample_ids",
            "candidate_policy", "gain", "k", "scope", "full_split_queries")
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"Evaluation protocol differs: {key}; never mix pilot/full reports")
    if left["metric_source_sha256"] != right["metric_source_sha256"]:
        raise ValueError("Metric source implementations differ; independently validate before comparing")
    definitions = [p["metric_definition"] for p in (a, b) if p.get("metric_definition") is not None]
    if len(definitions) == 2 and definitions[0] != definitions[1]:
        raise ValueError("Metric definitions differ")
    if left["input_spec"]["max_length"] != right["input_spec"]["max_length"]:
        raise ValueError("Total pair token budgets differ; use the same budget for this experiment")
    per_query = []
    for qid in a["sample_ids"]:
        x, y = left["records"][qid], right["records"][qid]
        if [s["candidate_id"] for s in x["scores"]] != [s["candidate_id"] for s in y["scores"]]:
            raise ValueError(f"Candidate IDs/order differ: {qid}")
        if x["metrics"]["judged_positive_count"] != y["metrics"]["judged_positive_count"]:
            raise ValueError(f"Full-qrels denominator differs: {qid}")
        if not math.isclose(x["metrics"]["candidate_recall"], y["metrics"]["candidate_recall"], rel_tol=0, abs_tol=TOLERANCE):
            raise ValueError(f"Candidate recall differs with identical candidates: {qid}")
        per_query.append({"id": qid, "baseline_ndcg@10": x["metrics"]["ndcg@10"],
                          "candidate_ndcg@10": y["metrics"]["ndcg@10"],
                          "delta": y["metrics"]["ndcg@10"] - x["metrics"]["ndcg@10"]})
    differences = [row["delta"] for row in per_query]
    wins = sum(d > TOLERANCE for d in differences)
    losses = sum(d < -TOLERANCE for d in differences)
    compiler_fields = sorted(set(left["input_spec"]) | set(right["input_spec"]))
    compiler_differences = {key: {"baseline": left["input_spec"].get(key), "candidate": right["input_spec"].get(key)}
                            for key in compiler_fields if left["input_spec"].get(key) != right["input_spec"].get(key)}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "comparison_version": VERSION,
            "scope": a["scope"], "dataset_id": a["dataset_id"], "split": a["split"],
            "queries": len(per_query), "full_split_queries": a["full_split_queries"],
            "selection_allowed": a["scope"] == "full_split" and a["split"] in {"dev", "validation"},
            "protocol_checks": {key: a[key] for key in keys if key != "sample_ids"},
            "metric_source_sha256": left["metric_source_sha256"],
            "metric_definition": definitions[0] if definitions else None,
            "legacy_metric_definition_evidence": "Historical runner records K/gain and source hash; identical metrics.py hash verifies common implementation",
            "baseline": method(left), "candidate": method(right),
            "method_comparison": {"kind": "different_model_methods_not_a_single_factor_ablation",
                "input_compilers_equal": not compiler_differences, "compiler_differences": compiler_differences,
                "same_total_pair_token_budget": left["input_spec"]["max_length"],
                "device_equal": a.get("device") == b.get("device"), "dtype_equal": a.get("dtype") == b.get("dtype"),
                "precision_equal": a.get("precision", "none") == b.get("precision", "none"),
                "attribution": "Differences may include pretraining/post-training, backbone weights, head, template, tokenizer and precision. Cannot attribute the result solely to ScoreHead, Tree Mask, or fine-tuning.",
                "content_budget_note": "Equal total token cap does not imply identical token IDs or retained document text when templates/tokenizers differ"},
            "metrics": {key: {"baseline": left["metrics"][key], "candidate": right["metrics"][key],
                         "delta": right["metrics"][key] - left["metrics"][key]} for key in METRICS},
            "ndcg@10_paired_bootstrap": paired_bootstrap(differences, seed=42, resamples=10_000),
            "query_changes": {"wins": wins, "ties": len(differences) - wins - losses, "losses": losses,
                              "tie_absolute_tolerance": TOLERANCE},
            "latency": {"speed_ratio_computed": False, "baseline_scope": a.get("latency_scope"),
                        "candidate_scope": b.get("latency_scope"),
                        "reason": "This script validates quality comparisons, not controlled same-hardware performance experiments"},
            "interpretation": "One fixed candidate protocol and one benchmark; no SOTA, head-only benefit, mask-only benefit or end-to-end retrieval claim.",
            "per_query": per_query}


def markdown(result):
    title = "PILOT 方法比较：不能作为完整 benchmark" if result["scope"] == "pilot" else "固定候选集上的模型方法比较"
    interval = result["ndcg@10_paired_bootstrap"]
    methods = result["method_comparison"]
    lines = [f"# {title}", "", f"数据集：{result['dataset_id']}；split：{result['split']}；"
             f"query：{result['queries']} / {result['full_split_queries']}。", "",
             f"基线：`{result['baseline']['model']}`；候选：`{result['candidate']['model']}`。", "",
             "已验证相同 manifest、split、完整 qrels、query/candidate ID 与顺序、指标实现及总 token 预算。",
             "两种方法可以采用不同模板、tokenizer、读出和训练历史。不能把总差值单独归因于 head、mask 或微调。", "",
             "| 指标 | 基线 | 候选 | 绝对差值 |", "|---|---:|---:|---:|"]
    for key, values in result["metrics"].items():
        lines.append(f"| {key} | {values['baseline']:.6f} | {values['candidate']:.6f} | {values['delta']:+.6f} |")
    lines += ["", f"nDCG@10 差值 {interval['mean_delta']:+.6f}；paired bootstrap 95% CI "
              f"[{interval['ci_lower']:+.6f}, {interval['ci_upper']:+.6f}]，10,000 resamples，seed 42。",
              "Bootstrap 以 query 为独立抽样单位，不建模共享主题相关性；单 seed 结果也不能证明训练稳定性。", "",
              f"输入协议相同：{methods['input_compilers_equal']}；device 相同：{methods['device_equal']}；"
              f"weight dtype 相同：{methods['dtype_equal']}；autocast precision 相同：{methods['precision_equal']}。",
              "相同 token 上限不保证不同模板保留同样多的正文。具体 compiler/readout 差异见 comparison.json。",
              "不计算速度倍数，不声称 SOTA。test 比较不能用于 checkpoint 或超参数选择。"]
    if result["scope"] == "pilot":
        lines += ["", "本比较只覆盖 pilot query，不能用于完整 dev 门槛或最终 test 结论。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError("Choose a new comparison output directory")
        result = compare(args.baseline, args.candidate)
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (output / "report.md").write_text(markdown(result), encoding="utf-8")
        print(json.dumps({"output": str(output), "scope": result["scope"], "queries": result["queries"],
                          "metrics": result["metrics"], "bootstrap": result["ndcg@10_paired_bootstrap"]},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"comparison: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
