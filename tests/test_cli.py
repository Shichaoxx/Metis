"""CLI vertical tests with local synthetic data and a tiny random Qwen3.

These test software contracts, not pretrained retrieval quality. No network,
credentials, or business data are needed.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cometa.cli import main, parser
from cometa.schema import sha256, write_jsonl


REPOSITORY = Path(__file__).resolve().parents[1]
HAS_MODEL_DEPS = all(importlib.util.find_spec(package) is not None
                     for package in ("torch", "transformers", "accelerate", "tokenizers"))


def load_example(name):
    spec = importlib.util.spec_from_file_location(f"cometa_test_{name}", REPOSITORY / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CLIAssertions(unittest.TestCase):
    def invoke(self, args, expected=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(argument) for argument in args])
        self.assertEqual(status, expected, f"args={args}\nstdout={stdout.getvalue()}\nstderr={stderr.getvalue()}")
        return stdout.getvalue(), stderr.getvalue()


class CLIDataTests(CLIAssertions):
    def make_beir_source(self, root):
        source = root / "source"
        (source / "qrels").mkdir(parents=True)
        write_jsonl(source / "corpus.jsonl", [
            {"_id": "d1", "title": "apple", "text": "fruit"},
            {"_id": "d2", "title": "orange", "text": "fruit"},
            {"_id": "d3", "title": "water", "text": "cold"},
        ])
        write_jsonl(source / "queries.jsonl", [{"_id": f"{split}-q", "text": "apple"}
                                                for split in ("train", "dev", "test")])
        for split in ("train", "dev", "test"):
            (source / "qrels" / f"{split}.tsv").write_text(
                f"query-id\tcorpus-id\tscore\n{split}-q\td1\t1\n{split}-q\td3\t2\n")
        return source

    def test_frozen_retrieval_prepare_evaluate_and_auxiliary_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_beir_source(root)
            retrieval = root / "frozen.jsonl"
            write_jsonl(retrieval, [{"id": f"{split}-q", "scores": {"d1": 0.8, "d2": 0.7}}
                                    for split in ("train", "dev", "test")])
            retrieval.with_name(retrieval.name + ".meta.json").write_text(json.dumps({
                "method": "synthetic-frozen-retriever", "gold_used": False,
                "run_sha256": sha256(retrieval),
                "source_file_sha256": {name: sha256(source / name) for name in ("corpus.jsonl", "queries.jsonl")},
            }))
            output = root / "prepared"
            self.invoke(["prepare-nfcorpus", "--data-dir", source, "--output", output,
                         "--top-k", "2", "--retrieval-run", retrieval])
            manifest = output / "manifest.json"
            self.invoke(["validate", manifest])
            registry = root / "registry.json"
            self.invoke(["dataset", "register", "tiny-retrieval", manifest, "--registry", registry])
            report = root / "retrieval-test"
            self.invoke(["evaluate", "--manifest", "tiny-retrieval", "--registry", registry,
                         "--split", "test", "--baseline", "retrieval", "--output", report])
            metrics = json.loads((report / "metrics.json").read_text())
            self.assertEqual(metrics["queries"], 1)
            self.assertEqual(metrics["candidate_recall"], 0.5)
            self.assertEqual(metrics["recall@10"], 0.5)
            protocol = json.loads((report / "evaluation.json").read_text())
            self.assertEqual(protocol["status"], "completed")
            self.assertEqual(protocol["manifest_sha256"], sha256(manifest))
            self.assertEqual(protocol["retrieval_sha256"], sha256(output / "test.retrieval.jsonl"))
            self.assertEqual(protocol["qrels_sha256"], sha256(output / "qrels" / "test.tsv"))
            prediction = json.loads((report / "predictions.jsonl").read_text())
            self.assertEqual(prediction["score_semantics"], "retriever_score")

            # Dense/frozen runs must not silently masquerade as a BM25 baseline.
            _, error = self.invoke(["evaluate", "--manifest", manifest, "--baseline", "bm25",
                                    "--output", root / "wrong-baseline"], expected=1)
            self.assertIn("No frozen retrieval scores", error)
            # Both auxiliary files are hash-verified before any scoring/output.
            for path in (output / "test.retrieval.jsonl", output / "qrels" / "test.tsv"):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                _, error = self.invoke(["evaluate", "--manifest", manifest, "--baseline", "retrieval",
                                        "--output", root / "corrupted"], expected=1)
                self.assertIn("checksum mismatch", error)
                self.assertFalse((root / "corrupted").exists())
                path.write_bytes(original)

    def test_retrieval_command_dispatch_and_parser(self):
        args = parser().parse_args(["retrieve-nfcorpus", "--data-dir", "raw", "--output", "run.jsonl",
                                   "--model", "local-embedder", "--revision", "pinned", "--top-k", "100",
                                   "--batch-size", "4", "--max-length", "2048"])
        self.assertEqual((args.top_k, args.batch_size, args.max_length), (100, 4, 2048))
        with patch("cometa.benchmarks.retrieval.build_dense_run", return_value=Path("run.jsonl")) as build:
            self.invoke(["retrieve-nfcorpus", "--data-dir", "raw", "--output", "run.jsonl",
                         "--model", "local-embedder", "--revision", "pinned", "--top-k", "100",
                         "--batch-size", "4", "--max-length", "2048"])
        build.assert_called_once_with("raw", "run.jsonl", top_k=100, model_name_or_path="local-embedder",
                                      revision="pinned", device="cpu", dtype="float32", batch_size=4, max_length=2048)


@unittest.skipUnless(HAS_MODEL_DEPS, "optional train dependencies are not installed")
class CLIVerticalTests(CLIAssertions):
    def test_tiny_train_predict_relocate_evaluate_compare_and_tamper(self):
        import torch
        from cometa.artifacts import verify_artifact
        torch.set_num_threads(1)
        smoke = load_example("smoke_train")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = smoke.make_fixture(root / "fixture")
            base = Path(config["model"]["name_or_path"])
            data = root / "data"
            data.mkdir()
            manifest = {"schema_version": "1.0", "dataset_id": "synthetic-cli-smoke",
                        "revision": "fixture-v1", "task_kind": "ranking", "task_id": "rerank", "splits": {}}
            for split in ("train", "validation", "test"):
                sample = copy.deepcopy(smoke.sample_records()[0])
                sample["id"] = f"{split}-q"
                records = [sample]
                qrels = f"query-id\tcorpus-id\tscore\n{split}-q\td1\t1\n{split}-q\td3\t2\n"
                if split == "test":
                    missed = copy.deepcopy(sample)
                    missed["id"] = "test-no-retrieved-positive"
                    missed["supervision"]["labels"] = {}
                    records.append(missed)
                    qrels += "test-no-retrieved-positive\td3\t1\n"
                split_path, qrels_path = data / f"{split}.jsonl", data / f"{split}.tsv"
                write_jsonl(split_path, records)
                qrels_path.write_text(qrels)
                manifest["splits"][split] = {"path": split_path.name, "sha256": sha256(split_path),
                    "count": len(records), "qrels_path": qrels_path.name, "qrels_sha256": sha256(qrels_path)}
            manifest_path = data / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            config["data"] = {"manifest": str(manifest_path), "train_split": "train", "validation_split": "validation"}
            config["output"] = {"root": str(root / "runs"), "project": "vertical"}
            config["model"]["pair_batch_size"] = 1
            config["training"]["gradient_checkpointing"] = True
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))

            self.invoke(["validate", manifest_path])
            self.invoke(["train", config_path])
            runs = list((root / "runs" / "vertical").iterdir())
            self.assertEqual(len(runs), 1)
            run = runs[0]
            run_state = json.loads((run / "run.json").read_text())
            self.assertEqual(run_state["status"], "completed")
            artifact = run / run_state["artifact"]
            seal = verify_artifact(artifact)
            self.assertIn("model_spec.json", seal["files"])
            self.assertEqual(seal["task"]["kind"], "ranking")

            inference = root / "inference.jsonl"
            inference_sample = copy.deepcopy(smoke.sample_records()[0])
            inference_sample["id"] = "agent-request"
            inference_sample.pop("supervision")
            write_jsonl(inference, [inference_sample])
            before = root / "before.jsonl"
            self.invoke(["predict", "--artifact", artifact, "--input", inference, "--output", before, "--top-k", "1"])
            before_prediction = json.loads(before.read_text())
            self.assertEqual(len(before_prediction["selected_ids"]), 1)

            # Exercise direct loading separately; this is the random base, not a
            # claimed strong baseline and not expected to match trained scores.
            direct = root / "direct-eval"
            self.invoke(["evaluate", "--manifest", manifest_path, "--split", "test", "--model", base,
                         "--layout", "tree", "--max-length", "256", "--max-tree-tokens", "512",
                         "--pair-batch-size", "1", "--output", direct])
            direct_protocol = json.loads((direct / "evaluation.json").read_text())
            self.assertEqual(direct_protocol["status"], "completed")
            self.assertIn("model_spec", direct_protocol)

            relocated = root / "independent" / "exported-model"
            self.invoke(["export", "--artifact", artifact, "--output", relocated])
            verify_artifact(relocated)
            # A full export must work without its original artifact or base path.
            shutil.rmtree(artifact)
            shutil.rmtree(base)
            after = root / "after.jsonl"
            self.invoke(["predict", "--artifact", relocated, "--input", inference, "--output", after, "--top-k", "1"])
            after_prediction = json.loads(after.read_text())
            self.assertEqual(before_prediction["scores"], after_prediction["scores"])
            self.assertEqual(before_prediction["selected_ids"], after_prediction["selected_ids"])

            reports = [root / "repeat-one", root / "repeat-two"]
            for report in reports:
                self.invoke(["evaluate", "--manifest", manifest_path, "--split", "test",
                             "--artifact", relocated, "--output", report])
                metrics = json.loads((report / "metrics.json").read_text())
                self.assertEqual(metrics["queries"], 2)
                self.assertEqual(metrics["candidate_recall"], 0.25)
                self.assertEqual(metrics["recall@10"], 0.25)
                self.assertEqual(json.loads((report / "evaluation.json").read_text())["artifact_manifest_sha256"],
                                 sha256(relocated / "artifact.json"))
            text, _ = self.invoke(["compare", "--baseline", reports[0], "--candidate", reports[1], "--max-drop", "0"])
            comparison = json.loads(text)
            self.assertTrue(comparison["passed"])
            self.assertEqual(comparison["delta"], 0)
            self.assertFalse(comparison["selection_allowed"])

            tool_example = load_example("agent_tool")
            tool = tool_example.DecisionTool(relocated)
            response = tool.rank_candidates("apple", inference_sample["candidates"], top_k=1, request_id="tool-q")
            self.assertEqual(response["id"], "tool-q")
            self.assertEqual(response["selected_ids"], after_prediction["selected_ids"])
            self.assertEqual(response["score_semantics"], "raw_yes_minus_no_logit")

            spec = relocated / "model_spec.json"
            spec.write_bytes(spec.read_bytes() + b"\n")
            _, error = self.invoke(["predict", "--artifact", relocated, "--input", inference,
                                    "--output", root / "tampered.jsonl"], expected=1)
            self.assertIn("integrity", error)
            self.assertFalse((root / "tampered.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
