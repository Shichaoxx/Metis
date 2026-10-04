"""One tokenizer/compiler for training, offline predictions, and serving.

The template follows the Qwen3-Reranker model card. Complete candidate bodies
are tokenized before splitting, so merges at the query/document boundary do
not change the pair input when sharing a prefix. No label or candidate ID is
inserted into the model text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYSTEM_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
                 'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                 '<|im_end|>\n<|im_start|>user\n')
ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
COMPILER_VERSION = 'qwen-reranker-v1'
SCORE_HEAD_COMPILER_VERSION = 'qwen-score-head-v1'
SCORE_HEAD_INSTRUCTION = 'Score how relevant the candidate is to the query and its context'
SCORE_HEAD_PREFIX = '<Task>: Estimate candidate relevance.\n'
SCORE_HEAD_SUFFIX = '\n<Relevance>:'


@dataclass(frozen=True)
class CompiledSample:
    sample_id: str
    candidate_ids: tuple[str, ...]
    prefix_ids: tuple[int, ...]
    branch_ids: tuple[tuple[int, ...], ...]
    truncated: tuple[bool, ...]

    @property
    def pairs(self) -> list[list[int]]:
        return [list(self.prefix_ids + branch) for branch in self.branch_ids]


class QwenCompiler:
    version = COMPILER_VERSION
    system_prefix = SYSTEM_PREFIX
    assistant_suffix = ASSISTANT_SUFFIX
    default_instruction = DEFAULT_INSTRUCTION
    readout = 'last_assistant_suffix_token_yes_minus_no'

    def __init__(self, tokenizer: Any, max_length: int = 4096,
                 instruction: str | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = self.default_instruction if instruction is None else instruction
        self.system_ids = tuple(tokenizer.encode(self.system_prefix, add_special_tokens=False))
        self.suffix_ids = tuple(tokenizer.encode(self.assistant_suffix, add_special_tokens=False))
        if not self.system_ids or not self.suffix_ids:
            raise ValueError('The tokenizer must encode the Qwen system and assistant template')

    def spec(self) -> dict:
        return {'version': self.version, 'system_prefix': self.system_prefix,
                'assistant_suffix': self.assistant_suffix, 'instruction': self.instruction,
                'max_length': self.max_length, 'truncation': 'document_tail',
                'position_policy': 'prefix_then_independent_branch',
                'readout': self.readout}

    def compile(self, sample: dict) -> CompiledSample:
        candidates = sample.get('candidates', [])
        ids = tuple(c['id'] for c in candidates)
        if len(set(ids)) != len(ids):
            raise ValueError('Candidate IDs must be unique within a sample')
        if not candidates:
            return CompiledSample(str(sample.get('id', '')), (), self.system_ids, (), ())
        query = sample['input']['query']
        context = sample['input'].get('context', '')
        if not isinstance(context, str):
            raise ValueError('input.context must be a string')
        instruction = self.instruction
        if context:
            instruction += '\nContext: ' + context
        body_prefix = f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: '
        prefix_tokens = self.tokenizer.encode(body_prefix, add_special_tokens=False)
        bodies = [self.tokenizer.encode(body_prefix + c['text'], add_special_tokens=False)
                  for c in candidates]
        # Restrict sharing to the request prefix. Do not accidentally make the
        # shared prefix depend on common document text or the answer suffix.
        shared = len(prefix_tokens)
        for tokens in bodies:
            shared = min(shared, len(tokens))
            for index in range(shared):
                if tokens[index] != prefix_tokens[index]:
                    shared = index
                    break
        prefix = self.system_ids + tuple(prefix_tokens[:shared])
        available = self.max_length - len(prefix) - len(self.suffix_ids)
        # An extremely long query must be reported, not silently removed.
        request_remainder = len(prefix_tokens) - shared
        if available < max(1, request_remainder):
            raise ValueError('Query/context/template leaves no document budget; increase max_length')
        branches, truncated = [], []
        for body in bodies:
            candidate_tokens = body[shared:]
            truncated.append(len(candidate_tokens) > available)
            branches.append(tuple(candidate_tokens[:available]) + self.suffix_ids)
        return CompiledSample(str(sample.get('id', '')), ids, prefix, tuple(branches), tuple(truncated))


class ScoreHeadCompiler(QwenCompiler):
    """Plain relevance input; the final suffix state feeds a learned scalar head.

    No label words, natural-language answer or chat generation are required.
    Reuses the same token-boundary-safe split and truncation as the yes/no
    compiler, so pairs and tree receive identical per-candidate token inputs.
    """
    version = SCORE_HEAD_COMPILER_VERSION
    system_prefix = SCORE_HEAD_PREFIX
    assistant_suffix = SCORE_HEAD_SUFFIX
    default_instruction = SCORE_HEAD_INSTRUCTION
    readout = 'last_relevance_suffix_token_scalar_logit'
