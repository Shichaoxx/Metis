"""Offline protocol tests: retrieval misses must survive into evaluation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metis.benchmarks.nfcorpus import BM25, fetch, prepare, read_qrels


class NFCorpusTests(unittest.TestCase):
    def _source(self, root):
        source = root / "source"
        (source / "qrels").mkdir(parents=True)
        docs = [
            {"_id": "d1", "title": "apple", "text": "apple fruit"},
            {"_id": "d2", "title": "apple", "text": "fruit"},
            {"_id": "d3", "title": "medical", "text": "relevant evidence"},
        ]
        queries = [{"_id": qid, "text": "apple"} for qid in ("train-q", "dev-q", "test-q")]
        for name, rows in (("corpus", docs), ("queries", queries)):
            (source / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for split in ("train", "dev", "test"):
            (source / "qrels" / f"{split}.tsv").write_text(f"query-id\tcorpus-id\tscore\n{split}-q\td3\t2\n")
        return source

    def test_no_test_gold_injection_full_qrels_and_explicit_train_weak_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            path = prepare(source, root / "prepared", top_k=2)
            manifest = json.loads(path.read_text())
            self.assertFalse(manifest["provenance"]["official_release_verified"])
            self.assertEqual(manifest["splits"]["train"]["count"], 1)
            for split in ("train", "validation", "test"):
                detail = manifest["splits"][split]
                data_path = path.parent / detail["path"]
                self.assertEqual(hashlib.sha256(data_path.read_bytes()).hexdigest(), detail["sha256"])
                row = json.loads(data_path.read_text())
                ids = [candidate["id"] for candidate in row["candidates"]]
                qrels = read_qrels(path.parent / detail["qrels_path"])
                self.assertEqual(qrels[row["id"]], {"d3": 2.0})
                self.assertEqual(row["metadata"]["full_positive_count"], 1)
                self.assertEqual(detail["statistics"]["retrieval_mean_recall_at_candidate_k"], 0)
                if split == "train":
                    self.assertEqual(ids, ["d1", "d3"])
                    self.assertEqual(row["supervision"]["labels"], {"d1": 0, "d3": 1})
                    self.assertEqual(row["metadata"]["weak_negative_ids"], ["d1"])
                    self.assertEqual(row["metadata"]["injected_positive_ids"], ["d3"])
                    # BM25 artifact retains the unmodified first-stage result.
                    self.assertNotIn("d3", json.loads((path.parent / detail["bm25_path"]).read_text())["scores"])
                else:
                    self.assertEqual(ids, ["d1", "d2"])
                    self.assertEqual(row["supervision"]["labels"], {})
                    self.assertEqual(row["supervision"]["unjudged_policy"], "ignore")
                    self.assertEqual(row["metadata"]["injected_positive_ids"], [])

    def test_bm25_ties_query_term_frequency_and_document_order(self):
        one = BM25({"z": "apple", "a": "apple", "other": "pear"})
        two = BM25({"other": "pear", "a": "apple", "z": "apple"})
        self.assertEqual(one.retrieve("APPLE apple", 3), one.retrieve("apple", 3))
        self.assertEqual(one.retrieve("apple", 3), two.retrieve("apple", 3))
        self.assertEqual([key for key, _ in one.retrieve("missing", 3)], ["a", "other", "z"])
        self.assertEqual([key for key, _ in one.retrieve("apple", 2)], ["a", "z"])

    def test_deterministic_outputs_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            first = prepare(source, root / "one", top_k=2)
            second = prepare(source, root / "two", top_k=2)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(original, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})

    def test_invalid_source_fails_before_preparing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            with self.assertRaises(ValueError):
                prepare(source, root / "out", top_k=1)
            (source / "qrels" / "dev.tsv").write_text("query-id\tcorpus-id\tscore\ntrain-q\td3\t1\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare(source, root / "out", top_k=2)

    def test_fetch_rejects_bad_archive_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.zip"
            bad.write_bytes(b"not the verified dataset")
            with self.assertRaisesRegex(ValueError, "checksum/size"):
                fetch(root / "data", archive_path=bad)
            self.assertFalse((root / "data" / "nfcorpus").exists())

    def test_frozen_external_run_validates_source_and_never_injects_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            run = root / "dense.jsonl"
            run.write_text("".join(json.dumps({"id": f"{split}-q", "scores": {"d1": 0.9, "d2": 0.8}}) + "\n"
                                   for split in ("train", "dev", "test")))
            metadata = {"method": "test-frozen-dense", "gold_used": False, "run_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
                        "source_file_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                               for name in ("corpus.jsonl", "queries.jsonl")}}
            meta_path = run.with_name(run.name + ".meta.json")
            meta_path.write_text(json.dumps(metadata))
            manifest_path = prepare(source, root / "out", top_k=2, retrieval_run=run)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["provenance"]["retrieval"]["source_run_metadata"]["method"], "test-frozen-dense")
            test = manifest["splits"]["test"]
            self.assertIn("retrieval_path", test)
            self.assertNotIn("bm25_path", test)
            row = json.loads((manifest_path.parent / test["path"]).read_text())
            self.assertEqual([c["id"] for c in row["candidates"]], ["d1", "d2"])
            metadata["source_file_sha256"]["corpus.jsonl"] = "wrong"
            meta_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                prepare(source, root / "bad", top_k=2, retrieval_run=run)

    def test_frozen_external_run_rejects_unknown_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            run = root / "dense.jsonl"
            run.write_text(json.dumps({"id": "train-q", "scores": {"d1": 0.9, "unknown": 0.8}}) + "\n")
            metadata = {"method": "test-frozen-dense", "gold_used": False, "run_sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
                        "source_file_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                               for name in ("corpus.jsonl", "queries.jsonl")}}
            run.with_name(run.name + ".meta.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Unknown document"):
                prepare(source, root / "bad", top_k=2, retrieval_run=run)


if __name__ == "__main__":
    unittest.main()
