import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from metis.api import decide
from metis.artifacts import create_run, finish_run, seal_artifact, verify_artifact, write_json
from metis.config import resolve_config
from metis.metrics import ranking_metrics
from metis.registry import DatasetRegistry, Registry
from metis.schema import sha256, validate_sample, validate_dataset, write_jsonl


def sample(sid="q1"):
    return {"schema_version": "1.0", "id": sid, "task_id": "rerank",
            "input": {"query": "a query"},
            "candidates": [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}],
            "supervision": {"kind": "candidate_labels", "labels": {"a": 1, "b": 0}, "unjudged_policy": "ignore"}}


def dataset(root):
    splits = {}
    for split in ("train", "validation", "test"):
        path = root / (split + ".jsonl")
        write_jsonl(path, [sample(split)])
        splits[split] = {"path": path.name, "sha256": sha256(path), "count": 1}
    path = root / "manifest.json"
    write_json(path, {"schema_version": "1.0", "dataset_id": "toy", "task_kind": "ranking", "splits": splits})
    return path


class SchemaTests(unittest.TestCase):
    def test_unknown_is_not_negative_and_all_negative_is_valid(self):
        value = sample()
        value["supervision"]["labels"] = {"a": 0}
        validate_sample(value)
        self.assertNotIn("b", value["supervision"]["labels"])
        value["supervision"]["unjudged_policy"] = "error"
        with self.assertRaises(ValueError):
            validate_sample(value)

    def test_duplicate_and_orphan_ids_rejected(self):
        value = sample()
        value["candidates"][1]["id"] = "a"
        with self.assertRaises(ValueError): validate_sample(value)
        value = sample(); value["supervision"]["labels"]["z"] = 1
        with self.assertRaises(ValueError): validate_sample(value)

    def test_task_targets_and_nonfinite(self):
        validate_sample(sample(), task_kind="single_choice")
        value = sample(); value["supervision"]["labels"]["b"] = 1
        with self.assertRaises(ValueError): validate_sample(value, task_kind="single_choice")
        value["supervision"]["labels"]["a"] = float("nan")
        with self.assertRaises(ValueError): validate_sample(value)

    def test_manifest_hash_and_split_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = dataset(root)
            self.assertEqual(validate_dataset(path)["splits"]["train"]["samples"], 1)
            with (root / "test.jsonl").open("a") as f: f.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"): validate_dataset(path)

    def test_registry_version_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = dataset(root)
            registry = DatasetRegistry(root / ".metis/datasets.json")
            registry.register("toy", path)
            self.assertEqual(registry.resolve("toy"), path.resolve())
            with self.assertRaises(ValueError): registry.register("toy", path)
            doc = json.loads(path.read_text()); doc["revision"] = "changed"; write_json(path, doc)
            with self.assertRaisesRegex(ValueError, "changed"): registry.resolve("toy")

    def test_explicit_plugin_registry(self):
        registry = Registry(); registry.register("a", int)
        self.assertIs(registry.get("a"), int)
        with self.assertRaises(ValueError): registry.register("a", float)
        with self.assertRaises(ValueError): registry.get("b")


class MetricTests(unittest.TestCase):
    def test_task_decisions_preserve_candidate_identity_and_policy(self):
        from metis.tasks import decide as task_decide

        self.assertIs(decide, task_decide)
        value = sample()
        value["candidates"].reverse()
        # Equal scores use stable IDs, independently of input ordering.
        ranked = task_decide(value, [0.0, 0.0], top_k=1)
        self.assertEqual(ranked["ranked_ids"], ["a", "b"])
        self.assertEqual(ranked["selected_ids"], ["a"])
        self.assertEqual([row["candidate_id"] for row in ranked["scores"]], ["b", "a"])
        self.assertEqual(task_decide(value, [0.0, 0.0], task_kind="single_choice")["selected_ids"], ["a"])
        self.assertEqual(task_decide(value, [0.0, 0.0], task_kind="multi_label")["selected_ids"], ["a", "b"])
        self.assertTrue(task_decide(value, [0.0, 0.0], task_kind="single_choice", min_score=1.0)["abstained"])

    def test_full_qrels_recall_and_map(self):
        m = ranking_metrics(["a", "b"], [2, 1], {"a": 1, "missing": 1}, k=1)
        self.assertEqual(m["recall@1"], 0.5)
        self.assertEqual(m["candidate_recall"], 0.5)
        self.assertEqual(m["map"], 0.5)
        self.assertEqual(m["ndcg@1"], 1.0)

    def test_linear_ndcg_and_permutation_stability(self):
        left = ranking_metrics(["b", "a"], [1, 1], {"a": 2, "b": 1}, k=2)
        right = ranking_metrics(["a", "b"], [1, 1], {"a": 2, "b": 1}, k=2)
        self.assertEqual(left, right)
        self.assertEqual(left["ndcg@2"], 1.0)

    def test_failures_are_not_probability_half(self):
        with self.assertRaises(ValueError): decide(sample(), [0.5])
        with self.assertRaises(ValueError): decide(sample(), [float("nan"), 1])
        result = decide(sample(), [-2, -1], min_score=0)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["coverage"], 1.0)


class ArtifactTests(unittest.TestCase):
    def test_core_and_task_imports_do_not_require_model_dependencies(self):
        source = Path(__file__).resolve().parents[1] / "src"
        script = '''
import importlib.abc
import sys

class RejectModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "accelerate", "peft"}:
            raise AssertionError(f"Core import loaded optional dependency: {fullname}")

sys.meta_path.insert(0, RejectModelDependencies())
sys.path.insert(0, sys.argv[1])
from metis import Predictor
from metis.api import decide as legacy_decide
from metis.tasks import decide
assert decide is legacy_decide
assert "metis.tasks.objectives" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-I", "-c", script, str(source)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_documented_top_level_predictor_import(self):
        from metis import Predictor
        from metis.api import Predictor as Implementation
        self.assertIs(Predictor, Implementation)

    def test_unique_runs_and_portable_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = create_run(root, "toy", {}); right = create_run(root, "toy", {})
            self.assertNotEqual(left, right)
            from metis.tasks import decisions
            recorded_sources = json.loads((left / "source.sha256.json").read_text())
            self.assertEqual(recorded_sources["tasks/decisions.py"], sha256(decisions.__file__))
            self.assertIn("tasks/objectives.py", recorded_sources)
            self.assertIn("tasks/__init__.py", recorded_sources)
            export = left / "exports/model"; export.mkdir(parents=True)
            (export / "weights.bin").write_bytes(b"toy-weights-not-a-model")
            seal_artifact(export)
            verify_artifact(export)
            (export / "weights.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"): verify_artifact(export)
            finish_run(left, status="failed")
            self.assertEqual(json.loads((left / "run.json").read_text())["status"], "failed")

    def test_symlinks_and_extra_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"w").write_text("w"); seal_artifact(root)
            (root/"unrecorded").write_text("extra")
            with self.assertRaises(ValueError): verify_artifact(root)
            (root/"link").symlink_to(root/"w")
            with self.assertRaises(ValueError): seal_artifact(root)

    def test_config_unknown_keys_fail_and_paths_follow_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset(root)
            cfg = root / "recipe.json"
            write_json(cfg, {"data": {"manifest": "manifest.json"}, "training": {"batch_szie": 1}})
            with self.assertRaisesRegex(ValueError, "Unknown config"): resolve_config(cfg)
            write_json(cfg, {"data": {"manifest": "manifest.json"}})
            value, _ = resolve_config(cfg)
            self.assertEqual(value["data"]["train_file"], str((root/"train.jsonl").resolve()))
            self.assertEqual(value["output"]["root"], str((root/"runs").resolve()))

    def test_score_head_training_config_and_sampling_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset(root)
            cfg = root / "recipe.json"
            raw = {"data": {"manifest": "manifest.json", "train_candidate_sampling": {
                           "enabled": True, "max_candidates": 8, "max_positives": 2, "seed": 17}},
                   "model": {"adapter": "qwen3_score_head", "head_hidden_size": 32},
                   "training": {"precision": "bf16", "head_learning_rate": 0.001}}
            write_json(cfg, raw)
            value, _ = resolve_config(cfg)
            self.assertEqual(value["model"]["adapter"], "qwen3_score_head")
            self.assertIsNone(value["model"]["instruction"])
            self.assertEqual(value["data"]["train_candidate_sampling"]["max_candidates"], 8)
            for overrides in ({"training": {"precision": "float16"}},
                              {"training": {"head_learning_rate": 0}},
                              {"model": {"adapter": "qwen3_yesno"}},
                              {"data": {"train_candidate_sampling": {"max_positives": 9}}}):
                with self.assertRaises(ValueError):
                    resolve_config(cfg, overrides)

    def test_selection_requires_full_dev_qrels_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest_path = dataset(root)
            cfg = root / "recipe.json"
            write_json(cfg, {"data": {"manifest": "manifest.json"},
                             "training": {"selection": {"enabled": True, "evaluate_initial": True}}})
            with self.assertRaisesRegex(ValueError, "full validation qrels"):
                resolve_config(cfg)
            qrels = root / "validation-qrels.json"
            write_json(qrels, {"validation": {"a": 1, "outside": 1}})
            manifest = json.loads(manifest_path.read_text())
            manifest["splits"]["validation"].update(qrels_path=qrels.name, qrels_sha256=sha256(qrels))
            write_json(manifest_path, manifest)
            value, _ = resolve_config(cfg)
            self.assertEqual(value["data"]["validation_qrels_path"], str(qrels.resolve()))
            self.assertTrue(value["training"]["selection"]["evaluate_initial"])
            for field in ("train_split", "validation_split"):
                with self.assertRaisesRegex(ValueError, "Test split"):
                    resolve_config(cfg, {"data": {field: "test"}})


if __name__ == "__main__":
    unittest.main()
