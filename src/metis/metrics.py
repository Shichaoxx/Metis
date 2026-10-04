"""Deterministic ranking metrics with full-qrels denominators.

nDCG uses linear graded gains (trec_eval/BEIR convention), not 2**grade - 1.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path


def load_qrels(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    result = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0] in {"query-id", "query_id", "qid"}:
                continue
            if len(row) != 3:
                raise ValueError("Expected BEIR qrels: query-id, corpus-id, score")
            qid, did, grade = row
            result.setdefault(qid, {})[did] = float(grade)
    return result


def ranking_metrics(candidate_ids, scores, qrels, *, k=10):
    if len(candidate_ids) != len(scores) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Scores and unique candidate IDs must align")
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Scores must be finite")
    if any(not math.isfinite(float(g)) or float(g) < 0 for g in qrels.values()):
        raise ValueError("Relevance grades must be finite and non-negative")
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: (-float(x[1]), x[0]))
    relevant = {cid for cid, value in qrels.items() if float(value) > 0}
    hits, precision_sum, reciprocal = 0, 0.0, 0.0
    for rank, (cid, _) in enumerate(ranked, 1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / rank
            reciprocal = reciprocal or 1 / rank
    dcg = sum(float(qrels.get(cid, 0)) / math.log2(i + 2) for i, (cid, _) in enumerate(ranked[:k]))
    ideal = sorted((float(v) for v in qrels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    total = len(relevant)
    return {f"ndcg@{k}": dcg / idcg if idcg else 0.0,
            "map": precision_sum / total if total else 0.0,
            "mrr": reciprocal,
            f"recall@{k}": sum(cid in relevant for cid, _ in ranked[:k]) / total if total else 0.0,
            "candidate_recall": hits / total if total else 0.0,
            "judged_positive_count": total}


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot report an empty evaluation as success")
    names = [k for k in rows[0] if k != "judged_positive_count"]
    return {"queries": len(rows), **{name: sum(r[name] for r in rows) / len(rows) for name in names},
            "queries_without_positive_qrels": sum(r["judged_positive_count"] == 0 for r in rows)}

