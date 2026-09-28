"""Optional Qwen3 embedding retrieval; no torch or transformers import until use.

Run ``python -m cometa.benchmarks.retrieval --help`` for the GPU recipe. This
module never reads qrels: retrieval candidates cannot be augmented by gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nfcorpus import _read_records, _write_json, document_text, sha256

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def last_token_pool(hidden, attention_mask):
    """Pool the last nonpadding token; supports left/right padding."""
    import torch
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden[B,L,H] and attention_mask[B,L]")
    if (attention_mask.sum(dim=-1) == 0).any():
        raise ValueError("Cannot pool an empty sequence")
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    last = positions.unsqueeze(0).expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def encode_texts(texts, *, tokenizer, model, device, batch_size=8, max_length=8192):
    """Return normalized float32 CPU vectors, following Qwen's pooling recipe."""
    import torch
    import torch.nn.functional as functional
    if batch_size < 1 or max_length < 1 or not texts:
        raise ValueError("Need nonempty texts, positive batch_size and max_length")
    vectors = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            batch = tokenizer(texts[offset:offset + batch_size], padding=True,
                              truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            pooled = last_token_pool(hidden, batch["attention_mask"]).float()
            normalized = functional.normalize(pooled, p=2, dim=1)
            if not torch.isfinite(normalized).all() or (pooled.norm(dim=1) == 0).any():
                raise ValueError("Embedding contains nonfinite or zero vectors")
            vectors.append(normalized.cpu())
    return torch.cat(vectors, dim=0)


def write_dense_run(documents: dict[str, str], queries: dict[str, str], output_path: str | Path,
                    *, tokenizer, model, source_hashes: dict, model_metadata: dict,
                    top_k=100, device="cpu", batch_size=8, max_length=8192,
                    instruction=DEFAULT_INSTRUCTION) -> Path:
    """Exact cosine retrieval over all documents, with a verifiable sidecar.

    This lower-level entry supports tiny local models/tests without downloading
    weights. The public loader below records the requested and resolved model
    revision; freeze both model revision and this run before model comparison.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    output = Path(output_path).expanduser().resolve()
    sidecar = output.with_name(output.name + ".meta.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Frozen retrieval output already exists: {output}")
    docids, qids = sorted(documents), sorted(queries)
    if not docids or not qids:
        raise ValueError("Need documents and queries")
    doc_vectors = encode_texts([documents[key] for key in docids], tokenizer=tokenizer,
                               model=model, device=device, batch_size=batch_size, max_length=max_length)
    query_vectors = encode_texts([f"Instruct: {instruction}\nQuery:{queries[key]}" for key in qids],
                                 tokenizer=tokenizer, model=model, device=device,
                                 batch_size=batch_size, max_length=max_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for offset in range(0, len(qids), batch_size):
            similarities = query_vectors[offset:offset + batch_size] @ doc_vectors.T
            for qid, row in zip(qids[offset:offset + batch_size], similarities.tolist()):
                ranked = sorted(zip(docids, row), key=lambda item: (-item[1], item[0]))[:top_k]
                stream.write(json.dumps({"id": qid, "scores": dict(ranked)}, sort_keys=True, allow_nan=False) + "\n")
    _write_json(sidecar, {
        "schema_version": "1.0", "method": "qwen3-embedding-exact-cosine-v1",
        "run_sha256": sha256(output), "source_file_sha256": source_hashes,
        "model": model_metadata, "query_instruction": instruction,
        "document_template": "title + newline + text, outer whitespace stripped",
        "pooling": "last nonpadding token", "normalization": "L2 float32",
        "search": "exact normalized dot product on CPU in float32, over full corpus",
        "ties": "descending score then ascending document ID",
        "top_k": min(top_k, len(docids)), "max_length": max_length,
        "batch_size": batch_size, "device": str(device), "documents": len(docids),
        "queries": len(qids), "gold_used": False,
        "truncation": "right truncation at max_length; use same frozen run for every comparison",
    })
    return output


def build_dense_run(data_dir: str | Path, output_path: str | Path, top_k: int = 100,
                    model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
                    revision: str | None = None, device: str = "cpu", dtype: str = "float32",
                    batch_size: int = 8, max_length: int = 8192) -> Path:
    """Load the optional public embedder and freeze a native-source retrieval run."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("dtype must be float32, bfloat16 or float16")
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.with_name(output.name + ".meta.json").exists():
        raise FileExistsError(f"Frozen retrieval output already exists: {output}")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or batch_size < 1 or max_length < 1:
        raise ValueError("top_k, batch_size and max_length must be positive")
    source = Path(data_dir).expanduser().resolve()
    if not (source / "corpus.jsonl").is_file():
        source /= "nfcorpus"
    docs = _read_records(source / "corpus.jsonl")
    queries = _read_records(source / "queries.jsonl")
    # Resolve a floating Hub ref once so tokenizer and model use the same commit.
    # Local directories have no Hub commit; their source identifier is retained.
    base_config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
    resolved_revision = getattr(base_config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=resolved_revision, padding_side="left")
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(model_name_or_path, revision=resolved_revision, config=base_config,
                                     torch_dtype=getattr(torch, dtype)).to(device).eval()
    return write_dense_run(
        {key: document_text(row) for key, row in docs.items()},
        {key: row["text"] for key, row in queries.items()}, output_path,
        tokenizer=tokenizer, model=model, device=device, top_k=top_k,
        batch_size=batch_size, max_length=max_length,
        source_hashes={name: sha256(source / name) for name in ("corpus.jsonl", "queries.jsonl")},
        model_metadata={"name_or_path": model_name_or_path, "requested_revision": revision,
                        "resolved_revision": resolved_revision, "dtype": dtype,
                        "transformers_version": __import__("transformers").__version__,
                        "torch_version": torch.__version__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    path = build_dense_run(args.data_dir, args.output, top_k=args.top_k,
                          model_name_or_path=args.model, revision=args.revision,
                          device=args.device, dtype=args.dtype,
                          batch_size=args.batch_size, max_length=args.max_length)
    print(json.dumps({"retrieval_run": str(path), "metadata": str(path) + ".meta.json"}))


if __name__ == "__main__":
    main()
