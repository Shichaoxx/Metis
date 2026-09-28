"""Expose a local decision artifact as a structured tool for an agent harness.

Run with a sealed Cometa artifact, without agent API credentials or a server:
    python examples/agent_tool.py --artifact /path/to/export --demo columns
Print only the tool contract (no model needed):
    python examples/agent_tool.py --show-schema

All demonstration text is invented. A ranking-trained document model does not
automatically become a validated column selector; the column demo illustrates
the interface only, and needs a column-task artifact for meaningful decisions.
"""

from __future__ import annotations

import argparse
import json
import uuid

TOOL_SCHEMA = {
    "name": "rank_candidates",
    "description": "Score bounded candidates against a query and return stable IDs; scores are raw model logits, not calibrated confidence.",
    "parameters": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
            "context": {"type": "string"},
            "candidates": {"type": "array", "minItems": 1, "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"]}},
            "top_k": {"type": "integer", "minimum": 1},
            "request_id": {"type": "string"},
        },
        "required": ["query", "candidates"],
    },
}


class DecisionTool:
    """Load once at harness startup, then call with query + bounded candidates."""

    def __init__(self, artifact, *, device="cpu", dtype="float32"):
        from cometa.api import Predictor
        self.predictor = Predictor(artifact, device=device, dtype=dtype)

    def rank_candidates(self, query, candidates, *, context="", top_k=3, request_id=None):
        sample = {"schema_version": "1.0", "id": request_id or uuid.uuid4().hex,
                  "task_id": "rerank", "input": {"query": query, "context": context},
                  "candidates": candidates}
        # The harness owns tool dispatch, context, tracing and retries. This
        # module owns scoring; it returns no generated prose or invented scores.
        return self.predictor.predict([sample], top_k=top_k)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--demo", choices=["documents", "columns"], default="documents")
    parser.add_argument("--show-schema", action="store_true")
    args = parser.parse_args()
    if args.show_schema:
        print(json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2))
        return
    if not args.artifact:
        parser.error("--artifact is required unless --show-schema is selected")
    tool = DecisionTool(args.artifact, device=args.device, dtype=args.dtype)
    if args.demo == "documents":
        query = "What color is a ripe apple?"
        context = "Synthetic document retrieval demonstration."
        candidates = [{"id": "doc.apple", "text": "A ripe apple may have red or green skin."},
                      {"id": "doc.sky", "text": "The clear daytime sky appears blue."}]
    else:
        query = "Total paid order amount for each customer"
        context = "Synthetic orders table, one row per order."
        candidates = [{"id": "orders.customer_id", "text": "customer_id: identifier of the customer"},
                      {"id": "orders.paid_amount", "text": "paid_amount: amount paid for the order"},
                      {"id": "orders.created_at", "text": "created_at: timestamp when the order was created"}]
    print(json.dumps(tool.rank_candidates(query, candidates, context=context, top_k=2),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
