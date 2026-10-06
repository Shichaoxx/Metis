"""Batched BoolQ readout variants for independent architecture experiments.

The module reuses a QwenScoreHead backbone and input compiler without changing
the production model or its serialization. All variants use an MLP with width
256. They differ only in the readout rule and, for decision_marker, the final
text marker. That marker uses existing tokenizer vocabulary; it is not a newly
added or independently trainable embedding.

Candidate attention pools tokens containing candidate-body characters. A token
that merges the separator with the first candidate characters is indivisible:
it is included and reported as a boundary merge. Request-only tokens, suffix
tokens, padding, and other examples are always excluded from the pooling mask.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from metis.compiler import CompiledSample, ScoreHeadCompiler
from metis.model import QwenScoreHead


READOUT_NAMES = ('last_relevance', 'decision_marker', 'candidate_attention')
DECISION_SUFFIX = '\n<Decision>:'


class CommonBudgetCompiler(ScoreHeadCompiler):
    """Compile the same candidate body before choosing a final marker.

    Reserving the longer marker for every variant prevents a marker-length
    difference from changing the retained candidate text. The underlying v1
    compiler still handles token-boundary-safe prefix splitting and truncation.
    """

    version = 'metis-research-boolq-readout-v1'

    def __init__(self, tokenizer, max_length: int, instruction: str, readout: str):
        if readout not in READOUT_NAMES:
            raise ValueError(f'Unknown readout: {readout}')
        super().__init__(tokenizer, max_length=max_length, instruction=instruction)
        self.original_suffix_ids = self.suffix_ids
        self.decision_suffix_ids = tuple(tokenizer.encode(DECISION_SUFFIX, add_special_tokens=False))
        if not self.decision_suffix_ids:
            raise ValueError('The tokenizer must encode the decision marker')
        self.reserved_suffix_tokens = max(len(self.original_suffix_ids), len(self.decision_suffix_ids))
        effective_length = max_length - self.reserved_suffix_tokens + len(self.original_suffix_ids)
        self.common_compiler = ScoreHeadCompiler(tokenizer, max_length=effective_length,
                                                instruction=instruction)
        self.variant = readout
        if readout == 'decision_marker':
            self.assistant_suffix = DECISION_SUFFIX
            self.suffix_ids = self.decision_suffix_ids
            self.readout = 'last_existing_vocabulary_decision_marker_token_scalar_logit'
        elif readout == 'candidate_attention':
            self.readout = 'learned_query_attention_over_candidate_body_tokens'

    def compile(self, sample: dict) -> CompiledSample:
        common = self.common_compiler.compile(sample)
        branches = tuple(branch[:-len(self.original_suffix_ids)] + self.suffix_ids
                         for branch in common.branch_ids)
        return CompiledSample(common.sample_id, common.candidate_ids, common.prefix_ids,
                              branches, common.truncated)

    def spec(self) -> dict:
        return {**super().spec(), 'variant': self.variant,
                'reserved_suffix_tokens': self.reserved_suffix_tokens,
                'body_budget_policy': 'common body budget reserves the longer Relevance/Decision marker',
                'relevance_suffix_ids': list(self.original_suffix_ids),
                'decision_suffix_ids': list(self.decision_suffix_ids),
                'common_body_compiler': self.common_compiler.spec()}


def candidate_body_mask(compiler: CommonBudgetCompiler, sample: dict,
                        compiled: CompiledSample) -> tuple[list[bool], int]:
    """Return a physical-token mask and the number of retained boundary merges.

    Offset mappings, rather than separately tokenizing the candidate, preserve
    the compiler's full-text tokenization at the request/candidate boundary.
    Any retained token containing candidate characters belongs to the mask.
    """
    if len(sample.get('candidates', [])) != 1:
        raise ValueError('BoolQ readout experiments require exactly one candidate per sample')
    instruction = compiler.instruction
    context = sample['input'].get('context', '')
    if context:
        instruction += '\nContext: ' + context
    body_prefix = (f"<Instruct>: {instruction}\n<Query>: {sample['input']['query']}\n<Document>: ")
    text = body_prefix + sample['candidates'][0]['text']
    try:
        encoded = compiler.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
    except (NotImplementedError, KeyError, TypeError) as error:
        raise ValueError('Candidate pooling requires a tokenizer with reliable offset mappings') from error
    pair = compiled.pairs[0]
    system_size = len(compiler.system_ids)
    suffix_size = len(compiler.suffix_ids)
    body_size = len(pair) - system_size - suffix_size
    if pair[system_size:system_size + body_size] != encoded['input_ids'][:body_size]:
        raise ValueError('Candidate offset mapping differs from the compiler tokenization')
    if len(offsets) != len(encoded['input_ids']):
        raise ValueError('Tokenizer returned invalid offset mappings')
    boundary = len(body_prefix)
    selected, boundary_merges = [], 0
    for start, end in offsets[:body_size]:
        includes_candidate = end > boundary and start < len(text) and end > start
        selected.append(includes_candidate)
        boundary_merges += int(includes_candidate and start < boundary)
    return [False] * system_size + selected + [False] * suffix_size, boundary_merges


class BoolQReadoutModel(nn.Module):
    """One backbone forward for a batch of independent, single-candidate samples.

    Load the source QwenScoreHead in FP32. A training driver may use BF16
    autocast while retaining FP32 parameter masters. Scores are differentiable
    FP32 logits with shape [batch], not probabilities or calibrated confidence.
    """

    def __init__(self, scorer: QwenScoreHead, readout: str = 'last_relevance',
                 mlp_width: int = 256):
        super().__init__()
        if readout not in READOUT_NAMES:
            raise ValueError(f'Unknown readout: {readout}')
        if mlp_width != 256:
            raise ValueError('This controlled readout experiment fixes MLP width at 256')
        if any(parameter.is_floating_point() and parameter.dtype != torch.float32
               for parameter in scorer.backbone.parameters()):
            raise ValueError('Load the backbone in FP32 to retain FP32 parameter masters')
        self.backbone = scorer.backbone
        self.tokenizer = scorer.tokenizer
        self.source, self.revision = scorer.source, scorer.revision
        self.readout, self.mlp_width = readout, mlp_width
        self.compiler = CommonBudgetCompiler(self.tokenizer, scorer.max_length,
                                             scorer.instruction, readout)
        dimension = self.backbone.config.hidden_size
        self.score_head = nn.Sequential(nn.Linear(dimension, mlp_width), nn.GELU(),
                                        nn.Linear(mlp_width, 1)).to(self.device, dtype=torch.float32)
        # Uniform candidate attention at initialization; no random-number use.
        self.register_parameter('pool_query', nn.Parameter(torch.zeros(dimension, device=self.device))
                                if readout == 'candidate_attention' else None)
        self.last_input_metadata = []

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def initialize_readout(self, positive_probability: float, seed: int | None = None) -> None:
        """Initialize a common MLP, then set its initial logit to the train prior.

        Passing the same seed gives identical MLP weights across variants without
        changing the caller's random-number state. Prior estimation belongs to
        the driver and must use training labels only. Endpoint priors are clipped
        to [1e-7, 1-1e-7] to keep the initial logit finite.
        """
        if not math.isfinite(positive_probability) or not 0 <= positive_probability <= 1:
            raise ValueError('Training prior must be a finite probability')
        if seed is not None:
            # Initialize on CPU so the same seed also produces identical heads
            # across different CUDA devices, without changing their RNG states.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                common_head = nn.Sequential(
                    nn.Linear(self.backbone.config.hidden_size, self.mlp_width, device='cpu'),
                    nn.GELU(), nn.Linear(self.mlp_width, 1, device='cpu'))
            self.score_head.load_state_dict(common_head.state_dict(), strict=True)
        probability = min(max(positive_probability, 1e-7), 1 - 1e-7)
        with torch.no_grad():
            self.score_head[-1].weight.zero_()
            self.score_head[-1].bias.fill_(math.log(probability / (1 - probability)))
            if self.pool_query is not None:
                self.pool_query.zero_()

    def prepare_batch(self, samples: list[dict]) -> dict:
        """Build right-padded independent rows and explicit candidate pooling masks."""
        if not samples:
            raise ValueError('prepare_batch requires a nonempty batch')
        compiled, pairs, masks, merges = [], [], [], []
        for sample in samples:
            if len(sample.get('candidates', [])) != 1:
                raise ValueError('BoolQ readout experiments require exactly one candidate per sample')
            item = self.compiler.compile(sample)
            mask, boundary_merges = candidate_body_mask(self.compiler, sample, item)
            if self.readout == 'candidate_attention' and not any(mask):
                raise ValueError('Candidate attention requires at least one retained candidate-body token')
            compiled.append(item)
            pairs.append(item.pairs[0])
            masks.append(mask)
            merges.append(boundary_merges)
        lengths = [len(pair) for pair in pairs]
        width = max(lengths)
        input_ids = torch.full((len(samples), width), self.tokenizer.pad_token_id,
                               dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        candidate_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, (pair, mask) in enumerate(zip(pairs, masks)):
            input_ids[row, :len(pair)] = torch.tensor(pair, device=self.device)
            attention_mask[row, :len(pair)] = 1
            candidate_mask[row, :len(mask)] = torch.tensor(mask, device=self.device)
        position_ids = torch.arange(width, device=self.device).expand(len(samples), -1)
        positions = torch.tensor(lengths, dtype=torch.long, device=self.device) - 1
        self.last_input_metadata = [
            {'sample_id': item.sample_id, 'input_tokens': length, 'candidate_truncated': item.truncated[0],
             'candidate_pooling_tokens': sum(mask), 'candidate_boundary_merge_tokens': merge}
            for item, length, mask, merge in zip(compiled, lengths, masks, merges)]
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'position_ids': position_ids, 'candidate_mask': candidate_mask,
                'readout_positions': positions, 'lengths': lengths,
                'truncated': [item.truncated[0] for item in compiled], 'boundary_merge_tokens': merges}

    def forward(self, samples: list[dict]) -> torch.Tensor:
        if not samples:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        batch = self.prepare_batch(samples)
        hidden = self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               position_ids=batch['position_ids'], use_cache=False,
                               return_dict=True).last_hidden_state
        if self.readout == 'candidate_attention':
            values = hidden.float()
            energies = (values * self.pool_query[None, None, :]).sum(dim=-1) / math.sqrt(values.shape[-1])
            energies = energies.masked_fill(~batch['candidate_mask'], -torch.inf)
            weights = torch.softmax(energies, dim=-1)
            representation = (weights.unsqueeze(-1) * values).sum(dim=1)
        else:
            rows = torch.arange(len(samples), device=self.device)
            representation = hidden[rows, batch['readout_positions']].float()
        return self.score_head(representation).squeeze(-1).float()

    def readout_spec(self) -> dict:
        """Describe the research contract separately from production v1 artifacts."""
        return {'format': 'metis-research-boolq-readout-v1', 'readout': self.readout,
                'compiler': self.compiler.spec(), 'mlp_width': self.mlp_width,
                'head_parameters': sum(parameter.numel() for parameter in self.score_head.parameters()),
                'pooling_parameters': self.pool_query.numel() if self.pool_query is not None else 0,
                'pooling_scope': 'candidate-body tokens only; indivisible prefix/candidate merges included and counted',
                'pool_query_initialization': 'zeros (uniform attention)' if self.pool_query is not None else None,
                'decision_marker': 'existing vocabulary text marker; no added embedding',
                'master_dtype': 'float32', 'layout': 'independent causal rows in one batch',
                'score_semantics': 'raw affirmative-answer logit',
                'artifact_scope': 'research model; not a production v1 Predictor artifact'}


def configure_lora(model: BoolQReadoutModel, *, rank: int = 8, alpha: int = 16,
                   gradient_checkpointing: bool = True) -> BoolQReadoutModel:
    """Freeze base weights and train q/k/v/o LoRA plus the independent readout."""
    from peft import LoraConfig, get_peft_model
    if rank < 1 or alpha < 1:
        raise ValueError('LoRA rank and alpha must be positive')
    if hasattr(model.backbone, 'peft_config'):
        raise ValueError('Configure LoRA once on an unwrapped backbone')
    model.backbone.requires_grad_(False)
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type='FEATURE_EXTRACTION', r=rank, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model.backbone.enable_input_require_grads()
    model.score_head.requires_grad_(True)
    if model.pool_query is not None:
        model.pool_query.requires_grad_(True)
    return model
