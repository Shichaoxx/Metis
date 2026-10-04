"""Prepare the native BEIR NFCorpus splits with a pinned, dependency-free BM25.

The dataset is fetched on request, never at import. Its academic-use terms are
separate from this toolkit's code license. Evaluation qrels are never restricted
to retrieved candidates, and validation/test candidates never use judgments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
ARCHIVE_SHA256 = "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
ARCHIVE_BYTES = 2_448_432
SOURCE_HASHES = {
    "corpus.jsonl": "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f",
    "queries.jsonl": "d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a",
    "qrels/train.tsv": "6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a",
    "qrels/dev.tsv": "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8",
    "qrels/test.tsv": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
}
HOMEPAGE = "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/"
BM25_PROTOCOL = {
    "name": "metis-bm25-v1", "k1": 1.2, "b": 0.75,
    "tokenization": "lowercase then regex [a-z0-9]+; no stemming or stopwords",
    "document_text": "title + newline + text, outer whitespace stripped",
    "query_term_frequency": "unique query terms; each term contributes once",
    "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
    "ties": "descending score then ascending document ID (Unicode lexicographic)",
    "zero_score_fill": True,
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path) -> bool:
    return all((path / name).is_file() and sha256(path / name) == expected
               for name, expected in SOURCE_HASHES.items())


def fetch(data_dir: str | Path, archive_path: str | Path | None = None) -> Path:
    """Fetch/verify a 2.4 MB pinned public archive and return its source directory.

    ``archive_path`` permits an already downloaded archive, including an offline
    installation. Archive contents and hashes are checked before extraction;
    existing nonmatching source files are never silently replaced.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    source = data_dir / "nfcorpus"
    if source.exists():
        if _verify_source(source):
            return source
        raise ValueError(f"Existing source does not match the pinned release: {source}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nfcorpus-", dir=data_dir) as temporary:
        temporary = Path(temporary)
        archive = Path(archive_path).expanduser().resolve() if archive_path else temporary / "nfcorpus.zip"
        if archive_path is None:
            request = urllib.request.Request(URL, headers={"User-Agent": "metis-nfcorpus/1"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
                downloaded = 0
                while chunk := response.read(1 << 18):
                    downloaded += len(chunk)
                    if downloaded > ARCHIVE_BYTES:
                        raise ValueError("NFCorpus archive exceeds its pinned size")
                    out.write(chunk)
        if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("NFCorpus archive checksum/size mismatch; source release may have changed")
        with zipfile.ZipFile(archive) as bundle:
            expected = {"nfcorpus/" + name for name in SOURCE_HASHES}
            actual = {item.filename for item in bundle.infolist() if not item.is_dir()}
            if actual != expected or sum(i.file_size for i in bundle.infolist()) > 20_000_000:
                raise ValueError("Unexpected NFCorpus archive members")
            # Copy only the allowlisted names; do not trust archive extraction paths.
            for name in sorted(expected):
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        extracted = temporary / "nfcorpus"
        if not _verify_source(extracted):
            raise ValueError("NFCorpus extracted file checksum mismatch")
        extracted.rename(source)
    return source


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = row.get("_id")
            if not isinstance(key, str) or not key or key in records:
                raise ValueError(f"Invalid/duplicate _id at {path}:{line_number}")
            if not isinstance(row.get("text"), str) or not isinstance(row.get("title", ""), str):
                raise ValueError(f"Text/title must be strings at {path}:{line_number}")
            records[key] = row
    if not records:
        raise ValueError(f"Empty source: {path}")
    return records


def read_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    """Read complete BEIR TSV judgments (including unretrieved documents)."""
    result: dict[str, dict[str, float]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Invalid BEIR qrels header: {path}")
        for row in reader:
            qid, docid, grade = row["query-id"], row["corpus-id"], float(row["score"])
            if not qid or not docid or not math.isfinite(grade) or grade < 0:
                raise ValueError(f"Invalid judgment in {path}")
            if docid in result[qid]:
                raise ValueError(f"Duplicate judgment {qid}/{docid} in {path}")
            result[qid][docid] = grade
    if not result:
        raise ValueError(f"Empty judgments: {path}")
    return dict(result)


def document_text(record: dict) -> str:
    return (record.get("title", "") + "\n" + record["text"]).strip()


class BM25:
    """Small deterministic reference BM25, not an Elasticsearch reproduction."""

    def __init__(self, documents: dict[str, str], k1: float = 1.2, b: float = 0.75):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs documents, k1 > 0 and 0 <= b <= 1")
        self.ids = sorted(documents)
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.lengths = {}
        for docid in self.ids:
            terms = Counter(_TOKEN_PATTERN.findall(documents[docid].lower()))
            self.lengths[docid] = sum(terms.values())
            for term, frequency in terms.items():
                self.postings[term].append((docid, frequency))
        self.average_length = sum(self.lengths.values()) / len(self.ids) or 1.0

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = dict.fromkeys(self.ids, 0.0)
        for term in sorted(set(_TOKEN_PATTERN.findall(query.lower()))):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (len(self.ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for docid, frequency in postings:
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[docid] / self.average_length)
                scores[docid] += idf * frequency * (self.k1 + 1) / denominator
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_retrieval_run(path: Path, source_hashes: dict, documents: dict,
                        queries: set[str], top_k: int) -> tuple[dict, dict]:
    metadata_path = path.with_name(path.name + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_sha256") != sha256(path):
        raise ValueError("Frozen retrieval run hash differs from its metadata")
    for name in ("corpus.jsonl", "queries.jsonl"):
        if metadata.get("source_file_sha256", {}).get(name) != source_hashes[name]:
            raise ValueError(f"Frozen retrieval source mismatch: {name}")
    if not isinstance(metadata.get("method"), str) or not metadata["method"]:
        raise ValueError("Frozen retrieval metadata needs a method name")
    if metadata.get("gold_used") is not False:
        raise ValueError("Frozen retrieval metadata must explicitly declare gold_used=false")
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            qid, scores = row["id"], row["scores"]
            if qid not in queries or qid in result or not isinstance(scores, dict):
                raise ValueError("Unknown/duplicate query or invalid frozen retrieval scores")
            if len(scores) < min(top_k, len(documents)):
                raise ValueError(f"Frozen retrieval has fewer than top_k candidates for {qid}")
            for docid, score in scores.items():
                if docid not in documents or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"Unknown document or nonfinite/non-numeric retrieval score: {qid}/{docid}")
            result[qid] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    if set(result) != queries:
        raise ValueError("Frozen retrieval must cover exactly all native train/dev/test query IDs")
    return result, metadata


def prepare(data_dir: str | Path, output_dir: str | Path, top_k: int = 50,
            retrieval_run: str | Path | None = None) -> Path:
    """Build canonical train/validation/test and full qrels with a fixed protocol.

    Only training may replace the last retrieved hit with one positive when there is
    no retrieved positive. Unjudged training hits are explicit weak negatives;
    evaluation unjudged hits have no supervised labels. No queries are dropped.
    ``retrieval_run`` imports scores plus a .meta.json sidecar; omission uses the
    light BM25 reference. Imported scores must be frozen before reranker tuning.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 2:
        raise ValueError("top_k must be an integer >= 2 for this training recipe")
    source = Path(data_dir).expanduser().resolve()
    if not (source / "corpus.jsonl").is_file():
        source = source / "nfcorpus"
    output = Path(output_dir).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Keep prepared outputs separate from the original source directory")
    documents = _read_records(source / "corpus.jsonl")
    queries = _read_records(source / "queries.jsonl")
    judgments = {split: read_qrels(source / "qrels" / f"{split}.tsv")
                 for split in ("train", "dev", "test")}
    seen = set()
    for split, qrels in judgments.items():
        if seen.intersection(qrels):
            raise ValueError("Native split query IDs overlap")
        seen.update(qrels)
        for qid, labels in qrels.items():
            if qid not in queries or any(docid not in documents for docid in labels):
                raise ValueError(f"Unknown query/document in {split} qrels")
    texts = {docid: document_text(row) for docid, row in documents.items()}
    source_hashes = {name: sha256(source / name) for name in SOURCE_HASHES}
    external, retrieval_metadata = None, None
    if retrieval_run is not None:
        external, retrieval_metadata = _read_retrieval_run(
            Path(retrieval_run), source_hashes, documents, seen, top_k)
        retrieval_protocol = {"name": "frozen_external_run", "top_k": top_k,
                              "source_run_metadata": retrieval_metadata,
                              "ties": BM25_PROTOCOL["ties"]}
        retrieval_name = retrieval_metadata["method"]
    else:
        retriever = BM25(texts)
        retrieval_protocol = dict(BM25_PROTOCOL, top_k=top_k)
        retrieval_name = "metis-bm25-v1"
    source_revision = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "qrels").mkdir(exist_ok=True)
    manifest = {
        "schema_version": "1.0", "dataset_id": "beir/nfcorpus",
        "revision": source_revision, "task_id": "rerank", "task_kind": "ranking",
        "splits": {},
        "provenance": {
            "source_url": URL, "homepage": HOMEPAGE,
            "archive_sha256": ARCHIVE_SHA256 if source_hashes == SOURCE_HASHES else None,
            "expected_official_archive_sha256": ARCHIVE_SHA256,
            "source_file_sha256": source_hashes,
            "official_release_verified": source_hashes == SOURCE_HASHES,
            "license": {
                "name": "NFCorpus academic-use terms; other uses require checking original owners' terms",
                "url": HOMEPAGE + "#terms-of-use",
                "statement": "Free to use for academic purposes. For other uses of NutritionFacts.org data, consult its Terms of Service and contact Dr. Michael Greger.",
            },
            "adapter": "metis.nfcorpus.v1", "corpus_count": len(documents),
            "queries_count": len(queries), "query_split": "native BEIR train/dev/test; dev renamed validation",
            "retrieval": retrieval_protocol,
            "supervision": {
                "candidate_labels": "binary: qrel grade > 0 is 1; original graded qrels retained separately",
                "train_unjudged": "weak negative label 0, not a human-judged negative",
                "train_positive_injection": "if frozen retrieval has no positive, replace last hit with one positive, ordered by descending grade then document ID",
                "validation_test_unjudged": "label omitted, unjudged_policy=ignore",
                "validation_test_positive_injection": "never",
                "evaluation": "full-qrels denominator; retrieval misses stay missed; unjudged hits contribute zero gain under standard IR evaluation",
            },
        },
    }
    for native, split in (("train", "train"), ("dev", "validation"), ("test", "test")):
        qrels = judgments[native]
        data_path = output / f"{split}.jsonl"
        run_path = output / f"{split}.retrieval.jsonl"
        qrels_path = output / "qrels" / f"{split}.tsv"
        shutil.copyfile(source / "qrels" / f"{native}.tsv", qrels_path)
        injected, weak_negatives, no_retrieved_positive, recall_sum, full_positive_count = 0, 0, 0, 0.0, 0
        with data_path.open("w", encoding="utf-8") as data_out, run_path.open("w", encoding="utf-8") as run_out:
            for qid in sorted(qrels):
                full_labels = qrels[qid]
                positives = {docid for docid, grade in full_labels.items() if grade > 0}
                retrieved = external[qid] if external is not None else retriever.retrieve(queries[qid]["text"], top_k)
                candidate_ids = [docid for docid, _ in retrieved]
                hits = positives.intersection(candidate_ids)
                full_positive_count += len(positives)
                recall_sum += len(hits) / len(positives) if positives else 0.0
                no_retrieved_positive += int(not hits)
                injected_ids = []
                if native == "train" and positives and not hits:
                    selected = min(positives, key=lambda docid: (-full_labels[docid], docid))
                    candidate_ids[-1] = selected
                    injected_ids = [selected]
                    injected += 1
                labels = {docid: int(full_labels[docid] > 0) for docid in candidate_ids if docid in full_labels}
                weak_ids = []
                if native == "train":
                    weak_ids = [docid for docid in candidate_ids if docid not in full_labels]
                    labels.update({docid: 0 for docid in weak_ids})
                    weak_negatives += len(weak_ids)
                row = {
                    "schema_version": "1.0", "id": qid, "task_id": "rerank",
                    "input": {"query": queries[qid]["text"], "context": ""},
                    "candidates": [{"id": docid, "text": texts[docid]} for docid in candidate_ids],
                    "supervision": {"kind": "candidate_labels", "labels": labels, "unjudged_policy": "ignore"},
                    "metadata": {
                        "native_split": native, "full_positive_count": len(positives),
                        "retrieval": retrieval_name, "injected_positive_ids": injected_ids,
                        "weak_negative_ids": weak_ids,
                    },
                }
                data_out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                run_out.write(json.dumps({"id": qid, "scores": dict(retrieved)}, sort_keys=True) + "\n")
        manifest["splits"][split] = {
            "path": data_path.name, "sha256": sha256(data_path), "count": len(qrels),
            "qrels_path": str(qrels_path.relative_to(output)), "qrels_sha256": sha256(qrels_path),
            "retrieval_path": run_path.name, "retrieval_sha256": sha256(run_path),
            "native_split": native,
            "statistics": {"injected_queries": injected, "weak_negative_labels": weak_negatives,
                           "retrieval_queries_with_no_positive": no_retrieved_positive,
                           "retrieval_mean_recall_at_candidate_k": recall_sum / len(qrels),
                           "full_positive_judgments": full_positive_count},
        }
        if external is None:
            manifest["splits"][split].update(bm25_path=run_path.name, bm25_sha256=sha256(run_path))
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
