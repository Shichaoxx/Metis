"""Agent-facing structured predictor. Errors never become fabricated scores."""
from __future__ import annotations
import math
from .schema import validate_sample
from .artifacts import verify_artifact


def decide(sample, scores, *, task_kind="ranking", top_k=None, threshold=0.5,
           min_score=None, model_id=None, score_semantics="raw_yes_minus_no_logit"):
    validate_sample(sample, require_labels=False, task_kind=task_kind)
    ids = [c["id"] for c in sample["candidates"]]
    if len(ids) != len(scores) or any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("Prediction does not cover every candidate with a finite score")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be positive")
    if min_score is not None and not math.isfinite(float(min_score)):
        raise ValueError("min_score must be finite")
    ranked = sorted(zip(ids, map(float, scores)), key=lambda x: (-x[1], x[0]))
    if task_kind == "ranking":
        selected = [cid for cid, value in ranked if min_score is None or value >= min_score]
        selected = selected[:top_k] if top_k else selected
    elif task_kind == "single_choice":
        selected = [ranked[0][0]] if min_score is None or ranked[0][1] >= min_score else []
    elif task_kind == "multi_label":
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie inside (0,1)")
        logit_threshold = math.log(threshold / (1 - threshold))
        selected = [cid for cid, value in ranked if value >= logit_threshold]
    else:
        raise ValueError("Unsupported task kind")
    return {"id": sample["id"], "model_id": model_id, "status": "ok", "coverage": 1.0,
            "score_semantics": score_semantics,
            "scores": [{"candidate_id": cid, "score": float(score)} for cid, score in zip(ids, scores)],
            "ranked_ids": [cid for cid, _ in ranked], "selected_ids": selected,
            "abstained": not selected,
            "policy": {"task_kind": task_kind, "top_k": top_k, "threshold": threshold, "min_score": min_score}}


class Predictor:
    def __init__(self, artifact, *, device="cpu", dtype="float32"):
        manifest = verify_artifact(artifact)
        from .model_registry import load_model
        self.model = load_model(artifact, device=device, dtype=dtype)
        self.task_kind = manifest["task"]["kind"]
        self.artifact = str(artifact)

    def predict(self, samples, **policy):
        for sample in samples:
            validate_sample(sample, require_labels=False, task_kind=self.task_kind)
        scores = self.model.score(samples)
        if len(scores) != len(samples):
            raise ValueError("Model lost samples")
        results = [decide(s, values, task_kind=self.task_kind, model_id=self.artifact,
                          score_semantics=self.model.score_semantics, **policy)
                   for s, values in zip(samples, scores)]
        metadata = getattr(self.model, "last_input_metadata", [])
        if len(metadata) == len(results):
            for result, details in zip(results, metadata):
                result["input_metadata"] = details
        return results
