"""Tiny offline model tests for the optional dense retrieval recipe."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional torch dependency is not installed")
class DenseRetrievalTests(unittest.TestCase):
    def test_last_token_pool_handles_both_padding_sides(self):
        import torch
        from cometa.benchmarks.retrieval import last_token_pool
        hidden = torch.arange(12).reshape(2, 3, 2).float()
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        pooled = last_token_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.stack([hidden[0, 2], hidden[1, 1]]))
        with self.assertRaisesRegex(ValueError, "empty"):
            last_token_pool(hidden, torch.zeros_like(mask))

    def test_fake_embeddings_produce_frozen_normalized_cosine_run(self):
        import torch
        from cometa.benchmarks.retrieval import write_dense_run

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                # The final token encodes a predictable category. Query prefix is
                # accepted, and unequal padding must not select the pad vector.
                ids = [[0, 1] if "apple" in text else [0, 2] for text in texts]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor([[0, 1]] * len(ids))}

        class Model:
            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask):
                table = torch.tensor([[9., 9.], [2., 0.], [0., 3.]])
                return SimpleNamespace(last_hidden_state=table[input_ids])

        with tempfile.TemporaryDirectory() as temporary:
            path = write_dense_run(
                {"d2": "pear", "d1": "apple"}, {"q1": "apple fruit"},
                Path(temporary) / "frozen.jsonl", tokenizer=Tokenizer(), model=Model(),
                source_hashes={"corpus.jsonl": "fixture", "queries.jsonl": "fixture"},
                model_metadata={"name_or_path": "tiny-fake"}, top_k=2, batch_size=1)
            row = json.loads(path.read_text())
            self.assertEqual(row["scores"], {"d1": 1.0, "d2": 0.0})
            metadata = json.loads(path.with_name(path.name + ".meta.json").read_text())
            self.assertFalse(metadata["gold_used"])
            self.assertEqual(metadata["documents"], 2)
            self.assertEqual(metadata["normalization"], "L2 float32")
            with self.assertRaises(FileExistsError):
                write_dense_run({"d1": "apple"}, {"q1": "apple"}, path, tokenizer=Tokenizer(), model=Model(),
                                source_hashes={}, model_metadata={})


if __name__ == "__main__":
    unittest.main()
