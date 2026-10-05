"""Agent-facing structured predictor. Errors never become fabricated scores."""
from __future__ import annotations
from .schema import validate_sample
from .artifacts import verify_artifact
from .tasks.decisions import decide


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
