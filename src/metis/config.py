"""Explicit config precedence and validation, JSON by default; optional YAML."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from .schema import TASK_KINDS, load_manifest

DEFAULT = {
    "schema_version": "1.0", "task": {"kind": "ranking"},
    "model": {"adapter": "qwen3_yesno", "name_or_path": "Qwen/Qwen3-Reranker-0.6B", "revision": None, "layout": "pairs",
              "backend": "eager", "device": "cpu", "dtype": "float32", "max_length": 4096,
              "max_tree_tokens": None, "pair_batch_size": 8, "head_hidden_size": None,
              "instruction": None},
    "data": {"manifest": "", "train_split": "train", "validation_split": "validation",
             "train_candidate_sampling": {"enabled": False, "max_candidates": 8,
                                          "max_positives": 2, "seed": 42}},
    "training": {"epochs": 1, "learning_rate": 2e-6, "batch_size": 1, "gradient_accumulation_steps": 1,
                 "seed": 42, "max_steps": -1, "save_steps": 100, "tuning": "full", "resume_from_checkpoint": None,
                 "weight_decay": 0.0, "warmup_ratio": 0.0, "gradient_checkpointing": False,
                 "precision": "float32", "head_learning_rate": None,
                 "selection": {"enabled": False, "metric": "ndcg@10", "evaluate_initial": False},
                 "lora": {"r": 8, "alpha": 16, "dropout": 0.0}},
    "objective": {"bce_weight": 1.0, "pairwise_weight": 0.5},
    "output": {"root": "runs", "project": "experiment"},
}


def _merge(target, source, prefix=""):
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"Unknown config field: {prefix}{key}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {prefix}{key}")
            _merge(target[key], value, prefix + key + ".")
        else:
            target[key] = value


def resolve_config(path, overrides=None):
    path = Path(path).resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML needs PyYAML; JSON configs require no optional dependency") from exc
        raw = yaml.safe_load(path.read_text())
    else:
        raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be an object")
    result = copy.deepcopy(DEFAULT)
    _merge(result, raw)
    _merge(result, overrides or {})
    if result["schema_version"] != "1.0":
        raise ValueError("Unsupported config schema")
    if result["task"]["kind"] not in TASK_KINDS:
        raise ValueError("Unsupported task kind")
    if result["model"]["layout"] not in {"pairs", "tree"}:
        raise ValueError("layout must be pairs or tree")
    if result["model"]["backend"] not in {"eager", "sdpa", "flex"}:
        raise ValueError("Unknown backend")
    if result["model"]["adapter"] not in {"qwen3_yesno", "qwen3_score_head"}:
        raise ValueError("Unknown model adapter")
    head_size = result["model"]["head_hidden_size"]
    if head_size is not None and (type(head_size) is not int or head_size < 1):
        raise ValueError("head_hidden_size must be a positive integer or null")
    if result["model"]["instruction"] is not None and not isinstance(result["model"]["instruction"], str):
        raise ValueError("model.instruction must be a string or null")
    if result["training"]["tuning"] not in {"full", "lora"}:
        raise ValueError("tuning must be full or lora")
    for key in ("epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "save_steps"):
        value = result["training"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be positive numeric")
    for key in ("batch_size", "gradient_accumulation_steps", "save_steps", "seed", "max_steps"):
        if not isinstance(result["training"][key], int) or isinstance(result["training"][key], bool):
            raise ValueError(f"training.{key} must be integer")
    for key in ("bce_weight", "pairwise_weight"):
        if not isinstance(result["objective"][key], (int, float)) or not math.isfinite(result["objective"][key]) or result["objective"][key] < 0:
            raise ValueError("Objective weights must be nonnegative")
    if sum(result["objective"].values()) <= 0:
        raise ValueError("At least one nonzero objective weight required")
    if result["training"]["max_steps"] != -1 and result["training"]["max_steps"] < 1:
        raise ValueError("max_steps must be -1 or positive")
    if type(result["training"]["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    if result["training"]["precision"] not in {"float32", "bf16"}:
        raise ValueError("training.precision must be float32 or bf16; master weights remain float32")
    head_lr = result["training"]["head_learning_rate"]
    if head_lr is not None and (type(head_lr) not in (int, float) or not math.isfinite(head_lr) or head_lr <= 0):
        raise ValueError("head_learning_rate must be positive numeric or null")
    if head_lr is not None and result["model"]["adapter"] != "qwen3_score_head":
        raise ValueError("head_learning_rate requires qwen3_score_head")
    sampling = result["data"]["train_candidate_sampling"]
    if type(sampling["enabled"]) is not bool:
        raise ValueError("train_candidate_sampling.enabled must be boolean")
    for key in ("max_candidates", "max_positives"):
        if type(sampling[key]) is not int or sampling[key] < 1:
            raise ValueError(f"train_candidate_sampling.{key} must be a positive integer")
    if sampling["max_positives"] > sampling["max_candidates"]:
        raise ValueError("max_positives must not exceed max_candidates")
    if type(sampling["seed"]) is not int:
        raise ValueError("train_candidate_sampling.seed must be an integer")
    if sampling["enabled"] and result["task"]["kind"] != "ranking":
        raise ValueError("Candidate sampling currently supports ranking only")
    selection = result["training"]["selection"]
    if type(selection["enabled"]) is not bool or selection["metric"] != "ndcg@10":
        raise ValueError("selection requires boolean enabled and metric=ndcg@10")
    if type(selection["evaluate_initial"]) is not bool or (selection["evaluate_initial"] and not selection["enabled"]):
        raise ValueError("selection.evaluate_initial must be boolean and requires enabled selection")
    if selection["enabled"] and result["task"]["kind"] != "ranking":
        raise ValueError("Quality-based selection currently supports ranking only")
    if type(result["model"]["pair_batch_size"]) is not int or result["model"]["pair_batch_size"] < 1:
        raise ValueError("pair_batch_size must be a positive integer")
    for key in ("weight_decay", "warmup_ratio"):
        value = result["training"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if result["training"]["warmup_ratio"] > 1:
        raise ValueError("warmup_ratio must not exceed 1")
    lora = result["training"]["lora"]
    if type(lora["r"]) is not int or lora["r"] < 1 or type(lora["alpha"]) is not int or lora["alpha"] < 1:
        raise ValueError("LoRA r and alpha must be positive integers")
    if type(lora["dropout"]) not in (int, float) or not 0 <= lora["dropout"] < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    if result["model"]["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Unknown model dtype")
    if not isinstance(result["model"]["max_length"], int) or result["model"]["max_length"] < 1:
        raise ValueError("max_length must be a positive integer")
    tree_limit = result["model"]["max_tree_tokens"]
    if tree_limit is not None and (not isinstance(tree_limit, int) or tree_limit < result["model"]["max_length"]):
        raise ValueError("max_tree_tokens must be at least max_length")
    if result["task"]["kind"] != "ranking" and result["objective"]["pairwise_weight"] != 0:
        raise ValueError("Classification recipes must explicitly set pairwise_weight=0")
    manifest_path = (path.parent / result["data"]["manifest"]).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.get("task_kind", "ranking") != result["task"]["kind"]:
        raise ValueError("Task kind differs from dataset manifest")
    result["data"]["manifest"] = str(manifest_path)
    for split_field in ("train_split", "validation_split"):
        split = result["data"][split_field]
        if not isinstance(split, str) or split.lower() == "test":
            raise ValueError("Test split cannot participate in training or model selection")
    for field, split_field in (("train_file", "train_split"), ("validation_file", "validation_split")):
        split = result["data"][split_field]
        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split: {split}")
        result["data"][field] = manifest["splits"][split]["path"]
    validation_spec = manifest["splits"][result["data"]["validation_split"]]
    result["data"]["validation_qrels_path"] = validation_spec.get("qrels_path")
    if selection["enabled"] and not result["data"]["validation_qrels_path"]:
        raise ValueError("Quality-based selection requires full validation qrels in the manifest")
    test_paths = {spec["path"] for name, spec in manifest["splits"].items() if name.lower() == "test"}
    if test_paths & {result["data"]["train_file"], result["data"]["validation_file"]}:
        raise ValueError("Test split files cannot participate in training or model selection")
    result["output"]["root"] = str((path.parent / result["output"]["root"]).resolve())
    model = Path(result["model"]["name_or_path"])
    if (path.parent / model).exists():
        result["model"]["name_or_path"] = str((path.parent / model).resolve())
    resume = result["training"]["resume_from_checkpoint"]
    if resume:
        result["training"]["resume_from_checkpoint"] = str((path.parent / resume).resolve())
    return result, manifest
