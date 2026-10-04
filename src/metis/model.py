"""Qwen3 candidate scorers with shared pairs/dense-tree execution.

Backbone/readout, input semantics and attention execution are separate choices.
QwenReranker retains the original yes/no path; QwenScoreHead uses AutoModel
and a newly trainable MLP, with no language-model output head.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch
from torch import nn

from .attention import tree_attention_mask, tree_position_ids, tree_readout_positions
from .compiler import QwenCompiler, ScoreHeadCompiler


def resolve_dtype(name: str) -> torch.dtype:
    values = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    if name not in values:
        raise ValueError(f'Unsupported dtype {name!r}; expected {tuple(values)}')
    return values[name]


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _local_base_files(source):
    base = Path(source)
    if not base.is_dir():
        return {}
    files = [path for path in base.iterdir() if path.is_file() and
             (path.name == 'config.json' or path.suffix in ('.bin', '.safetensors')
              or path.name.endswith('.index.json'))]
    return {path.name: _file_digest(path) for path in sorted(files)}


class QwenCandidateScorer(nn.Module):
    """Shared execution and serialization; subclasses define the readout.

    Tree uses dense 4D masking. It avoids repeated prefix hidden states but
    does not promise a faster kernel. Large candidate sets are split into
    trees bounded by max_tree_tokens; independence makes chunking exact.
    Pairs use bounded forward batches. During training, the query-level loss
    retains every batch's autograd graph until backward; batching alone does
    not bound total saved activations to pair_batch_size candidates.
    """
    family = None
    compiler_class = QwenCompiler
    architecture_version = '1'

    def __init__(self, model_name_or_path, layout='pairs', backend='eager', device='cpu',
                 dtype='float32', revision=None, max_length=4096, *,
                 max_tree_tokens=None, instruction=None, pair_batch_size=8):
        super().__init__()
        from transformers import AutoConfig, AutoTokenizer
        self._set_options(model_name_or_path, layout, backend, revision, max_length,
                          max_tree_tokens, instruction, pair_batch_size)
        base_config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
        # A floating Hub ref is resolved once, then used for tokenizer, model,
        # and future LoRA reloads. Local base dependencies get content hashes.
        self.revision = getattr(base_config, '_commit_hash', None) or revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=self.revision)
        self.backbone = self._auto_model_class().from_pretrained(
            model_name_or_path, revision=self.revision, config=base_config, torch_dtype=resolve_dtype(dtype),
            attn_implementation=backend)
        self._finish_setup(device)

    def _set_options(self, source, layout, backend, revision, max_length, max_tree_tokens, instruction,
                     pair_batch_size):
        if layout not in ('pairs', 'tree'):
            raise ValueError('layout must be pairs or tree')
        if backend not in ('eager', 'sdpa'):
            raise NotImplementedError('v0 implements eager and SDPA only; flex is not implemented')
        if max_length < 1:
            raise ValueError('max_length must be positive')
        if isinstance(pair_batch_size, bool) or not isinstance(pair_batch_size, int) or pair_batch_size < 1:
            raise ValueError('pair_batch_size must be a positive integer')
        self.source = str(source)
        self.revision = revision
        self.layout, self.backend, self.max_length = layout, backend, max_length
        self.max_tree_tokens = max_tree_tokens or max_length
        if self.max_tree_tokens < max_length:
            raise ValueError('max_tree_tokens must cover at least one max_length pair')
        self.instruction = self.compiler_class.default_instruction if instruction is None else instruction
        self.pair_batch_size = pair_batch_size

    def _finish_setup(self, device):
        if self.backbone.config.model_type != 'qwen3':
            raise ValueError('v0 supports the Qwen3 model family only')
        self.backbone.config.use_cache = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer requires a pad_token or eos_token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.compiler = self.compiler_class(self.tokenizer, self.max_length, self.instruction)
        self.last_input_metadata = []
        self._setup_readout()
        self.to(device)

    @classmethod
    def from_components(cls, backbone, tokenizer, *, layout='pairs', backend='eager',
                        device='cpu', max_length=4096, max_tree_tokens=None,
                        instruction=None, source='tiny-random-qwen3', pair_batch_size=8,
                        **readout_options):
        """Offline injection path for tests; not a pretrained quality baseline."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._set_readout_options(**readout_options)
        obj._set_options(source, layout, backend, None, max_length, max_tree_tokens, instruction,
                         pair_batch_size)
        backbone.config._attn_implementation = backend
        obj.backbone, obj.tokenizer = backbone, tokenizer
        obj._finish_setup(device)
        return obj

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def _set_readout_options(self):
        """No additional readout configuration for the legacy LM-head path."""

    def _save_readout(self, destination):
        """The LM head is already included in the backbone/adapter export."""

    @classmethod
    def _readout_options_from_spec(cls, spec):
        return {}

    def _load_readout(self, source, spec):
        pass

    def _pairs(self, compiled):
        pairs = compiled.pairs
        results = []
        for start in range(0, len(pairs), self.pair_batch_size):
            chunk = pairs[start:start + self.pair_batch_size]
            max_len = max(map(len, chunk))
            ids = torch.full((len(chunk), max_len), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for i, tokens in enumerate(chunk):
                ids[i, :len(tokens)] = torch.tensor(tokens, device=self.device)
                mask[i, :len(tokens)] = 1
            pos = torch.arange(max_len, device=self.device).expand(len(chunk), -1)
            rows = torch.arange(len(chunk), device=self.device)
            ends = torch.tensor([len(x) - 1 for x in chunk], device=self.device)
            results.append(self._read_scores(ids, mask, pos, (rows, ends)))
        self._last_pair_chunk_count = len(results)
        return torch.cat(results)

    def _tree(self, compiled):
        prefix = compiled.prefix_ids
        chunks, current, size = [], [], len(prefix)
        for branch in compiled.branch_ids:
            if current and size + len(branch) > self.max_tree_tokens:
                chunks.append(current)
                current, size = [], len(prefix)
            current.append(branch)
            size += len(branch)
        if current:
            chunks.append(current)
        self._last_tree_chunk_count = len(chunks)
        results = []
        for branches in chunks:
            lengths = [len(x) for x in branches]
            ids = torch.tensor([list(prefix) + [t for b in branches for t in b]],
                               dtype=torch.long, device=self.device)
            dtype = next(self.backbone.parameters()).dtype
            mask = tree_attention_mask(len(prefix), lengths, dtype=dtype, device=self.device)
            positions = tree_position_ids(len(prefix), lengths, device=self.device)
            ends = torch.tensor(tree_readout_positions(len(prefix), lengths), device=self.device)
            rows = torch.zeros(len(ends), dtype=torch.long, device=self.device)
            results.append(self._read_scores(ids, mask, positions, (rows, ends)))
        return torch.cat(results)

    def score_tensors(self, samples: list[dict]) -> list[torch.Tensor]:
        """Differentiable raw scores, retaining caller candidate order."""
        scores = []
        self.last_input_metadata = []
        for sample in samples:
            compiled = self.compiler.compile(sample)
            if not compiled.candidate_ids:
                scores.append(torch.empty(0, device=self.device))
            else:
                scores.append(self._pairs(compiled) if self.layout == 'pairs' else self._tree(compiled))
            self.last_input_metadata.append({
                'sample_id': compiled.sample_id,
                'candidate_truncated': dict(zip(compiled.candidate_ids, compiled.truncated)),
                'input_truncated': any(compiled.truncated),
                'layout': self.layout, 'backend': self.backend,
                'physical_chunks': (self._last_tree_chunk_count if self.layout == 'tree'
                                    else self._last_pair_chunk_count)
                                   if compiled.candidate_ids else 0,
                'shared_prefix_tokens': len(compiled.prefix_ids),
                'candidate_tokens': {cid: len(branch) for cid, branch in
                                     zip(compiled.candidate_ids, compiled.branch_ids)},
            })
        return scores

    def score(self, samples: list[dict]) -> list[list[float]]:
        previous = self.training
        self.eval()
        try:
            with torch.no_grad():
                return [s.cpu().tolist() for s in self.score_tensors(samples)]
        finally:
            self.train(previous)

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        is_lora = hasattr(self.backbone, 'peft_config')
        self.backbone.save_pretrained(destination / 'model', safe_serialization=True)
        self.tokenizer.save_pretrained(destination / 'tokenizer')
        self._save_readout(destination)
        spec = {'format_version': '1.0', 'family': self.family,
                'architecture_version': self.architecture_version,
                'score_semantics': self.score_semantics,
                'tuning': 'lora' if is_lora else 'full',
                'source': self.source, 'revision': self.revision,
                'layout': self.layout, 'backend': self.backend, 'max_length': self.max_length,
                'pair_batch_size': self.pair_batch_size,
                'max_tree_tokens': self.max_tree_tokens, 'instruction': self.instruction,
                'compiler_version': self.compiler.version,
                'base_dependency_sha256': _local_base_files(self.source) if is_lora else {}}
        spec.update(self._readout_spec())
        (destination / 'model_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
        (destination / 'input_spec.json').write_text(json.dumps(self.compiler.spec(), indent=2) + '\n')
        return destination

    @classmethod
    def load(cls, source, device='cpu', dtype='float32'):
        from transformers import AutoTokenizer
        source = Path(source)
        spec = json.loads((source / 'model_spec.json').read_text())
        if (spec.get('format_version') != '1.0' or spec.get('compiler_version') != cls.compiler_class.version
                or spec.get('family') != cls.family
                or spec.get('architecture_version', '1') != cls.architecture_version):
            raise ValueError('Unsupported model or compiler format')
        if spec.get('score_semantics', cls.score_semantics) != cls.score_semantics:
            raise ValueError('Score semantics differ from the model adapter')
        readout_options = cls._readout_options_from_spec(spec)
        tokenizer = AutoTokenizer.from_pretrained(source / 'tokenizer')
        if spec['tuning'] == 'full':
            backbone = cls._auto_model_class().from_pretrained(source / 'model', torch_dtype=resolve_dtype(dtype),
                                                           attn_implementation=spec['backend'])
        elif spec['tuning'] == 'lora':
            from peft import PeftModel
            expected_base = spec.get('base_dependency_sha256', {})
            if expected_base and _local_base_files(spec['source']) != expected_base:
                raise ValueError('Local LoRA base model files changed or are missing')
            backbone = cls._auto_model_class().from_pretrained(spec['source'], revision=spec['revision'],
                torch_dtype=resolve_dtype(dtype), attn_implementation=spec['backend'])
            backbone = PeftModel.from_pretrained(backbone, source / 'model')
        else:
            raise ValueError('Unknown tuning mode in model artifact')
        obj = cls.from_components(backbone, tokenizer, layout=spec['layout'], backend=spec['backend'],
            device=device, max_length=spec['max_length'], max_tree_tokens=spec['max_tree_tokens'],
            instruction=spec['instruction'], source=spec['source'], pair_batch_size=spec.get('pair_batch_size', 8),
            **readout_options)
        obj.revision = spec['revision']
        obj._load_readout(source, spec)
        if obj.compiler.spec() != json.loads((source / 'input_spec.json').read_text()):
            raise ValueError('Input compiler differs from exported input spec')
        return obj


class QwenReranker(QwenCandidateScorer):
    """Existing Qwen3 LM-head readout: logit(yes) minus logit(no)."""
    family = 'qwen3_yesno'
    score_semantics = 'raw_yes_minus_no_logit'
    peft_task_type = 'CAUSAL_LM'

    @staticmethod
    def _auto_model_class():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def _setup_readout(self):
        yes = self.tokenizer.encode('yes', add_special_tokens=False)
        no = self.tokenizer.encode('no', add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1 or yes == no:
            raise ValueError('yes and no must each be distinct single tokenizer tokens')
        self.yes_id, self.no_id = yes[0], no[0]
        if self.tokenizer.unk_token_id in (self.yes_id, self.no_id):
            raise ValueError('yes/no readout cannot use the unknown token')

    def _read_scores(self, input_ids, attention_mask, position_ids, readouts):
        # Preserve the pretrained LM head without allocating full-vocabulary logits.
        base = self.backbone.get_base_model() if hasattr(self.backbone, 'peft_config') else self.backbone
        outputs = base.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, use_cache=False, return_dict=True)
        rows, positions = readouts
        selected = outputs.last_hidden_state[rows, positions]
        lm_head = base.get_output_embeddings()
        logits = torch.nn.functional.linear(selected, lm_head.weight[[self.no_id, self.yes_id]])
        if getattr(lm_head, 'bias', None) is not None:
            logits = logits + lm_head.bias[[self.no_id, self.yes_id]]
        return (logits[:, 1] - logits[:, 0]).float()

    def _readout_spec(self):
        return {'yes_token_id': self.yes_id, 'no_token_id': self.no_id,
                'readout': {'kind': 'existing_lm_head_two_rows'}}

    def _load_readout(self, source, spec):
        if (self.yes_id, self.no_id) != (spec['yes_token_id'], spec['no_token_id']):
            raise ValueError('Tokenizer yes/no IDs do not match exported model spec')


class QwenScoreHead(QwenCandidateScorer):
    """Qwen3 AutoModel plus a trainable relevance MLP, no LM head.

    Each raw scalar is an unconstrained relevance logit. Its interpretation is
    established by the task loss; sigmoid is not a claim of calibrated probability.
    The head remains outside PEFT, so LoRA wrapping of the backbone does not
    freeze or omit the head from our wrapper checkpoint and inference export.
    """
    family = 'qwen3_score_head'
    score_semantics = 'raw_relevance_logit'
    peft_task_type = 'FEATURE_EXTRACTION'
    compiler_class = ScoreHeadCompiler

    def __init__(self, *args, head_hidden_size=None, **kwargs):
        self._set_readout_options(head_hidden_size=head_hidden_size)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _auto_model_class():
        from transformers import AutoModel
        return AutoModel

    def _set_readout_options(self, head_hidden_size=None):
        if head_hidden_size is not None and (type(head_hidden_size) is not int or head_hidden_size < 1):
            raise ValueError('head_hidden_size must be a positive integer or None')
        self.head_hidden_size = head_hidden_size

    def _setup_readout(self):
        base = self.backbone.get_base_model() if hasattr(self.backbone, 'peft_config') else self.backbone
        if base.get_output_embeddings() is not None:
            raise ValueError('QwenScoreHead requires a Qwen3 AutoModel without an LM head')
        hidden_size = self.backbone.config.hidden_size
        if self.head_hidden_size is None:
            self.head_hidden_size = max(1, hidden_size // 4)
        self.score_head = nn.Sequential(nn.Linear(hidden_size, self.head_hidden_size), nn.GELU(),
                                        nn.Linear(self.head_hidden_size, 1))
        self.score_head.to(dtype=next(self.backbone.parameters()).dtype)

    def _read_scores(self, input_ids, attention_mask, position_ids, readouts):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, use_cache=False, return_dict=True)
        rows, positions = readouts
        return self.score_head(outputs.last_hidden_state[rows, positions]).squeeze(-1).float()

    def _readout_spec(self):
        return {'readout': {'kind': 'mlp_score_head', 'input_size': self.backbone.config.hidden_size,
                            'hidden_size': self.head_hidden_size, 'output_size': 1,
                            'activation': 'gelu', 'bias': True,
                            'weights': 'score_head.safetensors'}}

    def _save_readout(self, destination):
        from safetensors.torch import save_file
        save_file({name: tensor.detach().cpu().contiguous()
                   for name, tensor in self.score_head.state_dict().items()},
                  str(destination / 'score_head.safetensors'))

    @classmethod
    def _readout_options_from_spec(cls, spec):
        head = spec.get('readout', {})
        if (head.get('kind') != 'mlp_score_head' or head.get('activation') != 'gelu'
                or head.get('output_size') != 1 or head.get('bias') is not True
                or head.get('weights') != 'score_head.safetensors'
                or type(head.get('hidden_size')) is not int or head['hidden_size'] < 1):
            raise ValueError('Unsupported ScoreHead structure in model spec')
        return {'head_hidden_size': head['hidden_size']}

    def _load_readout(self, source, spec):
        from safetensors.torch import load_file
        if spec['readout'] != self._readout_spec()['readout']:
            raise ValueError('ScoreHead structure differs from the loaded backbone')
        # strict=True rejects missing/extra tensors; no random fallback head.
        self.score_head.load_state_dict(load_file(str(source / 'score_head.safetensors')), strict=True)
