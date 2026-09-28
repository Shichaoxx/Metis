#!/usr/bin/env python3
"""Compare two completed frozen-candidate evaluations without loading models.

The default inference is a paired query bootstrap for mean nDCG@10 difference,
10,000 resamples, seed 42. Old BM25 reports without a scope field are accepted
only when their query list matches an explicitly complete full_split report.
No inference/lookup speed ratio is calculated: the measurement scopes differ.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

METRICS = ("ndcg@10", "map", "mrr", "recall@10", "candidate_recall")
TOLERANCE = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {description}")
    return float(value)


def read_evaluation(directory):
    root = Path(directory).expanduser().resolve()
    names = ("evaluation.json", "metrics.json", "predictions.jsonl")
    protocol = json.loads((root / names[0]).read_text(encoding="utf-8"))
    metrics = json.loads((root / names[1]).read_text(encoding="utf-8"))
    if protocol.get("status") != "completed":
        raise ValueError(f"Evaluation is not completed: {root}")
    query_ids = protocol.get("sample_ids")
    if not isinstance(query_ids, list) or not query_ids or any(not isinstance(q, str) or not q for q in query_ids):
        raise ValueError("Evaluation needs a nonempty explicit query ID list")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate protocol query IDs")
    if protocol.get("completed") != len(query_ids) or metrics.get("queries") != len(query_ids):
        raise ValueError("Reported completion/aggregate counts do not cover all protocol queries")
    for key in ("manifest_sha256", "split_sha256", "qrels_sha256"):
        value = protocol.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing/invalid protocol identity: {key}")
    if protocol.get("k") != 10 or protocol.get("gain") != "linear_trec_eval":
        raise ValueError("This comparison requires K=10 and explicit linear/TREC gain")
    if protocol.get("split") not in {"test", "validation", "dev"}:
        raise ValueError("Only explicit held-out test/validation/dev splits are supported")
    if protocol.get("candidate_policy") != "frozen_manifest_candidates_no_gold_injection":
        raise ValueError("Candidate protocol is not a frozen held-out set without gold injection")
    if protocol.get("scope") not in (None, "full_split"):
        raise ValueError("Pilot/partial scope cannot be reported as a full benchmark comparison")
    prediction_path = root / "predictions.jsonl"
    predicted_hash = sha256(prediction_path)
    if protocol.get("predictions_sha256") and protocol["predictions_sha256"] != predicted_hash:
        raise ValueError("Completed predictions hash mismatch")
    records = {}
    with prediction_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid = row.get("id")
            if qid in records or qid not in query_ids:
                raise ValueError("Unknown or duplicate prediction query ID")
            if row.get("status") != "ok" or row.get("coverage") != 1.0:
                raise ValueError("A prediction is failed or does not cover every candidate")
            scores = row.get("scores")
            if not isinstance(scores, list) or not scores:
                raise ValueError("Missing candidate scores")
            ids = [item.get("candidate_id") for item in scores]
            if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
                raise ValueError("Invalid or duplicate candidate IDs")
            for item in scores:
                finite_number(item.get("score"), "candidate score")
            per_query = row.get("metrics", {})
            for metric in METRICS:
                value = finite_number(per_query.get(metric), f"per-query {metric}")
                if not 0 <= value <= 1 + TOLERANCE:
                    raise ValueError(f"Invalid bounded metric {metric}")
            positives = per_query.get("judged_positive_count")
            if isinstance(positives, bool) or not isinstance(positives, int) or positives < 0:
                raise ValueError("Missing/non-integer full-qrels positive count")
            records[qid] = row
    if set(records) != set(query_ids):
        raise ValueError("Predictions do not cover exactly every requested query")
    for metric in METRICS:
        average = math.fsum(records[qid]["metrics"][metric] for qid in query_ids) / len(query_ids)
        published = finite_number(metrics.get(metric), f"aggregate {metric}")
        if not math.isclose(average, published, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"Aggregate {metric} differs from all-query prediction mean")
    return {"directory": str(root), "protocol": protocol, "metrics": metrics, "records": records,
            "files_sha256": {name: sha256(root / name) for name in names},
            "predictions_sealed_in_source_report": bool(protocol.get("predictions_sha256"))}


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def paired_bootstrap(differences, *, seed=42, resamples=10_000):
    if not differences or not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("Bootstrap needs queries and a positive integer resample count")
    differences = [finite_number(value, "query difference") for value in differences]
    rng = random.Random(seed)
    size = len(differences)
    sampled_means = sorted(math.fsum(rng.choices(differences, k=size)) / size for _ in range(resamples))
    lower, upper = percentile(sampled_means, 0.025), percentile(sampled_means, 0.975)
    return {"method": "paired query bootstrap, percentile interval with linear interpolation",
            "unit": "query", "confidence_level": 0.95, "seed": seed, "resamples": resamples,
            "rng": "Python random.Random(seed).choices with replacement",
            "mean_delta": math.fsum(differences) / size, "ci_lower": lower, "ci_upper": upper,
            "ci_excludes_zero": lower > 0 or upper < 0,
            "ci_entirely_above_zero": lower > 0,
            "assumption": "Queries are treated as independent sampling units; shared source/topic dependence is not modeled",
            "limitations": "Interval for this paired benchmark mean, not global SOTA or out-of-domain generalization evidence"}


def compare(baseline_directory, candidate_directory, *, seed=42, resamples=10_000):
    baseline, candidate = read_evaluation(baseline_directory), read_evaluation(candidate_directory)
    left, right = baseline["protocol"], candidate["protocol"]
    keys = ("dataset_id", "manifest_sha256", "split", "split_sha256", "qrels_sha256", "sample_ids", "candidate_policy", "gain", "k")
    for key in keys:
        if left.get(key) != right.get(key):
            raise ValueError(f"Evaluation protocol differs: {key}")
    query_ids = right["sample_ids"]
    explicit_full = [item for item in (left, right) if item.get("scope") == "full_split"]
    if not explicit_full or any(item.get("full_split_queries") != len(query_ids) for item in explicit_full):
        raise ValueError("Need an explicitly full_split evaluation with a matching full query count")
    scope_evidence = {}
    for name, item in (("baseline", left), ("candidate", right)):
        scope_evidence[name] = ("explicit full_split scope and matching full query count" if item.get("scope") == "full_split"
                               else "legacy report: full scope inferred from identical IDs/hashes/count to explicit full_split counterpart")
    per_query = []
    for qid in query_ids:
        a, b = baseline["records"][qid], candidate["records"][qid]
        aids = [item["candidate_id"] for item in a["scores"]]
        bids = [item["candidate_id"] for item in b["scores"]]
        if aids != bids:
            raise ValueError(f"Frozen candidate IDs/order differ for {qid}")
        if a["metrics"]["judged_positive_count"] != b["metrics"]["judged_positive_count"]:
            raise ValueError(f"Full-qrels positive denominator differs for {qid}")
        if not math.isclose(a["metrics"]["candidate_recall"], b["metrics"]["candidate_recall"], abs_tol=TOLERANCE, rel_tol=0):
            raise ValueError(f"Candidate recall differs despite identical candidate IDs for {qid}")
        av, bv = a["metrics"]["ndcg@10"], b["metrics"]["ndcg@10"]
        per_query.append({"id": qid, "baseline_ndcg@10": av, "candidate_ndcg@10": bv, "delta": bv - av})
    differences = [row["delta"] for row in per_query]
    gains = [value for value in differences if value > TOLERANCE]
    losses = [value for value in differences if value < -TOLERANCE]
    metric_table = {key: {"baseline": baseline["metrics"][key], "candidate": candidate["metrics"][key],
                          "delta": candidate["metrics"][key] - baseline["metrics"][key]} for key in METRICS}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "comparison_version": "cometa-paired-comparison-v1",
        "scope": "full_split", "dataset_id": right["dataset_id"], "split": right["split"], "queries": len(query_ids),
        "protocol_checks": {key: right[key] for key in keys if key != "sample_ids"},
        "scope_evidence": scope_evidence,
        "baseline": {"model": left.get("model"), "directory": baseline["directory"],
                     "files_sha256": baseline["files_sha256"],
                     "predictions_sealed_in_source_report": baseline["predictions_sealed_in_source_report"]},
        "candidate": {"model": right.get("model"), "directory": candidate["directory"],
                      "files_sha256": candidate["files_sha256"],
                      "predictions_sealed_in_source_report": candidate["predictions_sealed_in_source_report"]},
        "metrics": metric_table,
        "ndcg@10_paired_bootstrap": paired_bootstrap(differences, seed=seed, resamples=resamples),
        "query_changes": {"wins": len(gains), "ties": len(differences) - len(gains) - len(losses), "losses": len(losses),
                          "tie_absolute_tolerance": TOLERANCE,
                          "mean_delta_all_queries": math.fsum(differences) / len(differences),
                          "mean_delta_on_wins": math.fsum(gains) / len(gains) if gains else None,
                          "mean_delta_on_losses": math.fsum(losses) / len(losses) if losses else None},
        "latency": {"speed_ratio_computed": False, "comparison_allowed": False,
                    "baseline_scope": left.get("latency_scope"), "candidate_scope": right.get("latency_scope"),
                    "reason": "Frozen BM25 score lookup and neural model inference have different measurement boundaries; no speed ratio is valid"},
        "interpretation": "Paired quality comparison on one frozen candidate protocol. This does not establish SOTA, fine-tuning benefit, or end-to-end retrieval performance.",
        "per_query": per_query,
    }


def markdown(result):
    interval = result["ndcg@10_paired_bootstrap"]
    changes = result["query_changes"]
    lines = ["# 固定候选集上的完整评测比较", "",
             f"数据集：{result['dataset_id']}；split：{result['split']}；全部 {result['queries']} 个 query。",
             f"基线：`{result['baseline']['model']}`；候选模型：`{result['candidate']['model']}`。", "",
             "已校验 manifest、split、qrels、完整 query ID、逐 query 候选 ID 与 gain/K 一致；未丢弃失败或零召回 query。", "",
             "| 指标 | 基线 | 候选模型 | 绝对差值 |", "|---|---:|---:|---:|"]
    for name, values in result["metrics"].items():
        lines.append(f"| {name} | {values['baseline']:.6f} | {values['candidate']:.6f} | {values['delta']:+.6f} |")
    lines += ["", f"nDCG@10 平均差值：**{interval['mean_delta']:+.6f}**；paired bootstrap 95% CI："
              f"**[{interval['ci_lower']:+.6f}, {interval['ci_upper']:+.6f}]**。",
              f"以 query 为配对单位，有放回抽样 {interval['resamples']:,} 次，seed={interval['seed']}；使用 percentile interval。",
              f"改善 / 持平 / 下降：**{changes['wins']} / {changes['ties']} / {changes['losses']}**。", "",
              "各 query 被视为独立抽样单位；此区间不建模共享主题或来源的相关性，也不证明全局 SOTA。",
              "这是固定候选下的质量比较，不是微调收益或全库端到端结果。", "",
              "**不比较速度倍数：BM25 记录的是冻结分数查表，神经模型记录的是实际推理，两者计时范围不同。**", "",
              "完整协议、输入报告 SHA-256、逐 query 差值及 CI 细节见 comparison.json。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, help="New directory for comparison.json/report.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError("Comparison output exists; choose a new directory to preserve prior evidence")
        result = compare(args.baseline, args.candidate, seed=args.seed, resamples=args.resamples)
        output.mkdir(parents=True, exist_ok=False)
        (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "report.md").write_text(markdown(result), encoding="utf-8")
        print(json.dumps({"output": str(output), "queries": result["queries"], "metrics": result["metrics"],
                          "bootstrap": result["ndcg@10_paired_bootstrap"], "query_changes": result["query_changes"]},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"comparison: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
