"""Thin Hugging Face Trainer integration, including real resumable checkpoints."""
from __future__ import annotations

import json
import math
import hashlib
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .model_registry import build_model
from .tasks.objectives import supervised_loss


class SupervisedReranker(nn.Module):
    def __init__(self, reranker, task_kind='ranking', bce_weight=1.0, pairwise_weight=0.5):
        super().__init__()
        self.reranker = reranker
        self.task_kind = task_kind
        self.bce_weight, self.pairwise_weight = bce_weight, pairwise_weight

    def forward(self, samples, labels=None):
        values = self.reranker.score_tensors(samples)
        losses = [supervised_loss(s, x, task_kind=self.task_kind, bce_weight=self.bce_weight,
                                  pairwise_weight=self.pairwise_weight)
                  for s, x in zip(values, samples)]
        loss = torch.stack(losses).mean()
        logits = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
        return {'loss': loss, 'logits': logits}


def _resume_metadata_paths(checkpoint):
    """Prefer current metadata; otherwise accept one unambiguous prior prefix.

    The selection file must share the resume file's prefix. Contract contents
    are still checked by the caller before any checkpoint state is restored.
    """
    checkpoint = Path(checkpoint)
    resume = checkpoint / 'metis_resume.json'
    if not resume.is_file():
        candidates = sorted(path for path in checkpoint.glob('*_resume.json') if path.is_file())
        if len(candidates) > 1:
            raise ValueError('Checkpoint has ambiguous resume metadata; specify a single metadata prefix')
        if candidates:
            resume = candidates[0]
    prefix = resume.name.removesuffix('_resume.json')
    return resume, checkpoint / f'{prefix}_selection.json'


def _collate(samples):
    # Labels signal to Trainer that eval loss is defined. Actual labels remain
    # ID-addressed in supervision, and never enter the input compiler.
    return {'samples': samples, 'labels': torch.zeros(len(samples), dtype=torch.long)}


def _read_samples(path, task_kind, *, allow_unjudged=False, stats=None):
    from .schema import read_jsonl
    records = []
    counts = {'total': 0, 'eligible': 0, 'skipped_unjudged': 0}
    for sample in read_jsonl(path, require_labels=not allow_unjudged, task_kind=task_kind):
        counts['total'] += 1
        judgments = sample.get('supervision', {}).get('labels', {})
        if allow_unjudged and not judgments:
            counts['skipped_unjudged'] += 1
            continue
        # Validate objective eligibility before constructing/loading a model.
        supervised_loss(torch.zeros(len(sample['candidates'])), sample, task_kind=task_kind)
        records.append(sample)
        counts['eligible'] += 1
    if stats is not None:
        stats.update(counts)
    if not records and not allow_unjudged:
        raise ValueError(f'No training records in {path}')
    return records


def _sample_training_candidates(samples, sampling):
    """Select judged train candidates once per run, independent of iteration order.

    Missing judgments never become negatives. Fewer positives allow additional
    explicit zero-grade negatives up to max_candidates. Source order and grades
    are retained after stable per-ID hash selection.
    """
    enabled = sampling.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('train_candidate_sampling.enabled must be boolean')
    policy = {'enabled': enabled, 'max_candidates': sampling.get('max_candidates', 8),
              'max_positives': sampling.get('max_positives', 2), 'seed': sampling.get('seed', 42)}
    for key in ('max_candidates', 'max_positives'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f'train_candidate_sampling.{key} must be positive integer')
    if policy['max_positives'] > policy['max_candidates'] or type(policy['seed']) is not int:
        raise ValueError('Invalid training candidate sampling limits or seed')
    result, lineage = [], []
    for sample in samples:
        labels = sample['supervision']['labels']
        candidates = sample['candidates']
        positives = [x['id'] for x in candidates if labels.get(x['id'], 0) > 0]
        negatives = [x['id'] for x in candidates if x['id'] in labels and labels[x['id']] == 0]
        selected_ids = {x['id'] for x in candidates}
        if enabled:
            def priority(cid):
                return hashlib.sha256(json.dumps([policy['seed'], sample['id'], cid]).encode()).digest(), cid
            chosen_positive = sorted(positives, key=priority)[:policy['max_positives']]
            slots = policy['max_candidates'] - len(chosen_positive)
            selected_ids = set(chosen_positive + sorted(negatives, key=priority)[:slots])
            if not selected_ids:
                raise ValueError(f"No judged candidates left after sampling: {sample['id']}")
            selected = [x for x in candidates if x['id'] in selected_ids]
            sampled = sample | {'candidates': selected, 'supervision': sample['supervision'] | {
                'labels': {cid: grade for cid, grade in labels.items() if cid in selected_ids}}}
        else:
            sampled = sample
        result.append(sampled)
        lineage.append({'query_id': sample['id'], 'source_candidates': len(candidates),
                        'selected_ids': [x['id'] for x in sampled['candidates']],
                        'selected_labels': sampled['supervision']['labels'],
                        'selected_positives': len(set(positives) & selected_ids),
                        'selected_negatives': len(set(negatives) & selected_ids),
                        'source_unjudged': len(candidates) - len(labels)})
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
    summary = {'policy': policy, 'mode': 'fixed_per_run' if enabled else 'disabled',
               'scope': 'train_only', 'queries': len(result),
               'source_candidates': sum(len(x['candidates']) for x in samples),
               'selected_candidates': sum(len(x['candidates']) for x in result),
               'sampled_training_sha256': digest}
    return result, summary, lineage


def _selection_data(data_cfg, selection, task_kind, train_samples):
    """Load the complete validation population independently of loss eligibility."""
    from .schema import read_jsonl, load_manifest, sha256
    from .metrics import load_qrels
    enabled = selection.get('enabled', False)
    if type(enabled) is not bool or selection.get('metric', 'ndcg@10') != 'ndcg@10':
        raise ValueError('selection requires boolean enabled and metric=ndcg@10')
    evaluate_initial = selection.get('evaluate_initial', False)
    if type(evaluate_initial) is not bool or (evaluate_initial and not enabled):
        raise ValueError('selection.evaluate_initial must be boolean and requires enabled selection')
    for field in ('train_split', 'validation_split'):
        if str(data_cfg.get(field, '')).lower() == 'test':
            raise ValueError('Test split cannot participate in training or model selection')
    if not enabled:
        return None, None, {'enabled': False}
    if task_kind != 'ranking':
        raise ValueError('Quality-based selection requires ranking')
    path = data_cfg.get('validation_file')
    qrels_path = data_cfg.get('validation_qrels_path')
    split = data_cfg.get('validation_split', 'validation')
    if not path or not qrels_path:
        raise ValueError('Quality-based selection requires complete validation data and full qrels')
    test_ids = set()
    if data_cfg.get('manifest'):
        manifest = load_manifest(data_cfg['manifest'])
        spec = manifest['splits'].get(split)
        if not spec or Path(spec['path']).resolve() != Path(path).resolve():
            raise ValueError('Selection data differs from its manifest split')
        if not spec.get('qrels_path') or Path(spec['qrels_path']).resolve() != Path(qrels_path).resolve():
            raise ValueError('Selection qrels differ from the validation manifest')
        for name, test_spec in manifest['splits'].items():
            if name.lower() == 'test':
                if Path(test_spec['path']).resolve() in {Path(path).resolve(), Path(data_cfg['train_file']).resolve()}:
                    raise ValueError('Test data cannot participate in training or selection')
                if test_spec.get('qrels_path') and Path(test_spec['qrels_path']).resolve() == Path(qrels_path).resolve():
                    raise ValueError('Test qrels cannot participate in selection')
                test_ids.update(x['id'] for x in read_jsonl(test_spec['path'], require_labels=False))
    samples = list(read_jsonl(path, require_labels=False, task_kind=task_kind))
    if not samples:
        raise ValueError('Validation population is empty')
    ids = {x['id'] for x in samples}
    train_ids = {x['id'] for x in train_samples}
    if ids & train_ids or test_ids & (ids | train_ids):
        raise ValueError('Training, validation and test query IDs must be disjoint for selection')
    qrels = load_qrels(qrels_path)
    if set(qrels) != ids:
        raise ValueError('Full validation qrels and validation query IDs must match exactly')
    # Validate every full-qrels grade before training or model construction.
    if any(not math.isfinite(float(g)) or float(g) < 0 for row in qrels.values() for g in row.values()):
        raise ValueError('Validation qrels grades must be finite and nonnegative')
    return samples, qrels, {'enabled': True, 'metric': 'ndcg@10', 'split': split,
                            'scope': 'complete_validation_full_qrels', 'queries': len(samples),
                            'validation_sha256': sha256(path), 'qrels_sha256': sha256(qrels_path),
                            'validation_candidate_sampling': False, 'tie_break': 'earliest_step',
                            'evaluate_initial': evaluate_initial}


def _complete_validation_metrics(reranker, samples, qrels, *, precision='float32'):
    from .metrics import ranking_metrics, aggregate
    rows = []
    device_type = next(reranker.parameters()).device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext():
        for sample in samples:
            scores = reranker.score([sample])[0]
            metrics = ranking_metrics([x['id'] for x in sample['candidates']], scores, qrels[sample['id']], k=10)
            rows.append({'query_id': sample['id'], 'candidate_ids': [x['id'] for x in sample['candidates']],
                         'scores': scores, 'metrics': metrics})
    return aggregate([x['metrics'] for x in rows]), rows


def _parameter_snapshot(model):
    """Hash trainable tensors only; check frozen tensors' PyTorch version counters.

    This avoids an extra full backbone copy. Frozen version checks supplement
    requires_grad/optimizer exclusion and are not presented as content hashes.
    """
    snapshot = {}
    for name, parameter in model.named_parameters():
        item = {'numel': parameter.numel(), 'dtype': str(parameter.dtype),
                'trainable': parameter.requires_grad, 'version': parameter._version,
                'role': 'score_head' if name.startswith('reranker.score_head.') else 'backbone'}
        if parameter.requires_grad:
            item['sha256'] = hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        snapshot[name] = item
    return snapshot


def train(config: dict, run_dir: Path) -> Path:
    """Train FP32 master parameters; optionally autocast compute to BF16.

    Trainer restores optimizer, scheduler, RNG and consumed-step state. Selection
    evaluates every validation query with full qrels each epoch, independently of
    judged-candidate loss eligibility. Both final and best weights are exported;
    the return value identifies the selected artifact (best when enabled).
    """
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    from .artifacts import event, write_json
    from .schema import sha256, write_jsonl
    from .model import _local_base_files
    run_dir = Path(run_dir)
    model_cfg, data_cfg = config['model'], config['data']
    train_cfg, objective = config.get('training', {}), config.get('objective', {})
    task_kind = config.get('task', {}).get('kind', 'ranking')
    if task_kind not in ('ranking', 'multi_label', 'single_choice'):
        raise ValueError('Unsupported task kind')
    bce_weight = float(objective.get('bce_weight', 1.0))
    pairwise_weight = float(objective.get('pairwise_weight', 0.5 if task_kind == 'ranking' else 0.0))
    if bce_weight < 0 or pairwise_weight < 0 or not math.isfinite(bce_weight + pairwise_weight):
        raise ValueError('Objective weights must be finite and nonnegative')
    if task_kind == 'ranking' and bce_weight + pairwise_weight == 0:
        raise ValueError('At least one ranking objective weight must be positive')
    if task_kind == 'multi_label' and (bce_weight == 0 or pairwise_weight != 0):
        raise ValueError('multi_label requires positive bce_weight and pairwise_weight=0')
    if task_kind == 'single_choice' and pairwise_weight != 0:
        raise ValueError('single_choice uses cross entropy; set pairwise_weight=0')
    precision = train_cfg.get('precision', 'float32')
    if precision not in ('float32', 'bf16'):
        raise ValueError('training.precision must be float32 or bf16; FP16 training is unsupported')
    head_lr = train_cfg.get('head_learning_rate')
    if head_lr is not None and (type(head_lr) not in (int, float) or not math.isfinite(head_lr) or head_lr <= 0):
        raise ValueError('head_learning_rate must be positive numeric or null')
    source_samples = _read_samples(data_cfg['train_file'], task_kind)
    sampling = data_cfg.get('train_candidate_sampling', {})
    if sampling.get('enabled') and task_kind != 'ranking':
        raise ValueError('Candidate sampling currently supports ranking only')
    train_samples, sampling_summary, sampling_lineage = _sample_training_candidates(source_samples, sampling)
    write_json(run_dir / 'sampling.json', sampling_summary)
    write_jsonl(run_dir / 'sampling.jsonl', sampling_lineage)
    event(run_dir, 'training_candidate_sampling', **sampling_summary)
    selection_samples, selection_qrels, selection_contract = _selection_data(
        data_cfg, train_cfg.get('selection', {}), task_kind, source_samples)
    selection_enabled = selection_contract['enabled']
    validation_stats = {'total': 0, 'eligible': 0, 'skipped_unjudged': 0,
                        'scope': 'judged-candidate loss subset; not a full-qrels benchmark',
                        'used_for_selection': False, 'loss_evaluation_enabled': not selection_enabled}
    validation = (_read_samples(data_cfg['validation_file'], task_kind,
                    allow_unjudged=True, stats=validation_stats)
                  if data_cfg.get('validation_file') else None)
    validation = validation or None
    event(run_dir, 'validation_loss_eligibility', **validation_stats)
    seed = int(train_cfg.get('seed', 42))
    set_seed(seed)
    # model.dtype remains an inference preference; low precision loading would
    # quantize master weights and Adam updates. Mixed compute is Trainer autocast.
    reranker = build_model(model_cfg | {'dtype': 'float32'})
    if head_lr is not None and not hasattr(reranker, 'score_head'):
        raise ValueError('head_learning_rate requires a score-head model adapter')
    tuning = train_cfg.get('tuning', 'full')
    if tuning == 'lora':
        from peft import LoraConfig, get_peft_model
        lora = train_cfg.get('lora', {})
        reranker.backbone = get_peft_model(reranker.backbone, LoraConfig(
            task_type=reranker.peft_task_type, r=int(lora.get('r', 8)),
            lora_alpha=int(lora.get('alpha', 16)), lora_dropout=float(lora.get('dropout', 0.0)),
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
        if hasattr(reranker, 'score_head') and not all(p.requires_grad for p in reranker.score_head.parameters()):
            raise ValueError('ScoreHead must remain trainable when the backbone uses LoRA')
    elif tuning != 'full':
        raise ValueError('tuning must be full or lora')
    gradient_checkpointing = train_cfg.get('gradient_checkpointing', False)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError('training.gradient_checkpointing must be a boolean')
    if gradient_checkpointing:
        # Enable on the actual HF backbone, since Trainer sees our task wrapper.
        # Non-reentrant checkpointing supports the custom 4D-mask forward and
        # preserves RNG for dropout during recomputation.
        reranker.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if tuning == 'lora':
            reranker.backbone.enable_input_require_grads()
    model = SupervisedReranker(reranker, task_kind, bce_weight, pairwise_weight)
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('Trainable master parameters must all remain float32')
    # Resume may change the output directory or total budget, but cannot silently
    # change input semantics, train data or objective under the same state.
    contract = {'model': model_cfg, 'task_kind': task_kind, 'tuning': tuning,
                'resolved_base_revision': reranker.revision,
                'local_base_dependency_sha256': _local_base_files(reranker.source),
                'input_spec': reranker.compiler.spec(),
                'local_tokenizer_sha256': {name: sha256(Path(reranker.source) / name) for name in
                    ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                     'vocab.json', 'merges.txt', 'tokenizer.model', 'vocab.txt')
                    if (Path(reranker.source) / name).is_file()},
                'backbone_config': reranker.backbone.config.to_dict(),
                'objective': {'bce_weight': bce_weight, 'pairwise_weight': pairwise_weight},
                'lora': train_cfg.get('lora', {}), 'seed': seed,
                'gradient_checkpointing': gradient_checkpointing,
                'precision': precision, 'master_dtype': 'float32',
                'learning_rate': train_cfg.get('learning_rate', 2e-6), 'head_learning_rate': head_lr,
                'weight_decay': train_cfg.get('weight_decay', 0.0),
                'warmup_ratio': train_cfg.get('warmup_ratio', 0.0),
                'batch_size': train_cfg.get('batch_size', 1),
                'gradient_accumulation_steps': train_cfg.get('gradient_accumulation_steps', 1),
                'train_sha256': sha256(data_cfg['train_file']),
                'sampling': sampling_summary, 'selection': selection_contract}
    # HF config can contain integer-key maps (id2label); compare the persisted
    # JSON representation rather than Python-only key types.
    contract = json.loads(json.dumps(contract))
    write_json(run_dir / 'training_contract.json', contract)
    resume = train_cfg.get('resume_from_checkpoint')
    best = None
    if resume:
        saved_contract, state_path = _resume_metadata_paths(resume)
        if not saved_contract.is_file() or json.loads(saved_contract.read_text()) != contract:
            raise ValueError('Resume contract differs or is missing; use a fresh run to initialize weights')
        if selection_enabled:
            if not state_path.is_file():
                raise ValueError('Checkpoint is missing selection state')
            best = json.loads(state_path.read_text()).get('best')
            if best:
                source = Path(best['artifact'])
                actual = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*') if p.is_file()}
                if not actual or actual != best['artifact_files']:
                    raise ValueError('Resume best artifact is missing or changed; retain its immutable selection directory')

    objective_metadata = contract['objective'] | {
        'task_kind': task_kind, 'bce_label_mapping': 'relevance > 0',
        'pairwise_comparisons': 'all strictly greater relevance grades',
        'reduction': 'per-query mean, then batch mean'}

    def export_model(destination, selection_metadata):
        destination = Path(reranker.save(destination))
        write_json(destination / 'training_objective.json', objective_metadata)
        write_json(destination / 'selection.json', selection_metadata)
        return destination

    parameter_before = {}
    validation_history = []
    initial_validation = None

    class SaveContract(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal initial_validation
            # Trainer has already restored model/optimizer state at this point.
            parameter_before.update(_parameter_snapshot(model))
            write_json(run_dir / 'parameters_initial.json', parameter_before)
            event(run_dir, 'training_parameters', master_dtype='float32', compute_precision=precision,
                  trainable_parameters=sum(x['numel'] for x in parameter_before.values() if x['trainable']),
                  frozen_parameters=sum(x['numel'] for x in parameter_before.values() if not x['trainable']),
                  score_head_trainable_parameters=sum(x['numel'] for x in parameter_before.values()
                                                     if x['trainable'] and x['role'] == 'score_head'))
            if selection_contract.get('evaluate_initial') and not resume and state.is_world_process_zero:
                metrics, rows = _complete_validation_metrics(reranker, selection_samples, selection_qrels,
                                                             precision=precision)
                initial_validation = {'global_step': 0, 'epoch': 0, 'metrics': metrics,
                                      'scope': 'complete_validation_full_qrels', 'precision': precision,
                                      'role': 'initial_model_reference', 'eligible_for_best': False}
                write_json(run_dir / 'validation' / 'initial.json', initial_validation)
                write_jsonl(run_dir / 'validation' / 'initial.jsonl', rows)
                event(run_dir, 'validation_initial_reference', **initial_validation)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                event(run_dir, 'trainer_log', step=state.global_step, epoch=state.epoch,
                      metrics=logs)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            nonlocal best
            if selection_enabled and state.is_world_process_zero:
                metrics, rows = _complete_validation_metrics(reranker, selection_samples, selection_qrels,
                                                             precision=precision)
                value = metrics['ndcg@10']
                record = {'global_step': state.global_step, 'epoch': state.epoch, 'metrics': metrics,
                          'scope': 'complete_validation_full_qrels', 'precision': precision}
                validation_history.append(record)
                write_json(run_dir / 'validation' / f'step-{state.global_step}.json', record)
                write_jsonl(run_dir / 'validation' / f'step-{state.global_step}.jsonl', rows)
                improved = best is None or value > best['metric_value']
                event(run_dir, 'validation_full_qrels', **record, improved=improved)
                if improved:
                    metadata = selection_contract | {
                        'kind': 'best_validation_metric', 'metric_value': value,
                        'global_step': state.global_step, 'epoch': state.epoch,
                        'compute_precision': precision}
                    artifact = export_model(run_dir / 'selection' / f'best-step-{state.global_step}', metadata)
                    best = {'metric_value': value, 'global_step': state.global_step, 'epoch': state.epoch,
                            'artifact': str(artifact.resolve()),
                            'artifact_files': {str(p.relative_to(artifact)): sha256(p)
                                               for p in artifact.rglob('*') if p.is_file()}}
                # Persist selection state after the epoch metric, including when
                # a step-based checkpoint was saved earlier at the same step.
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                destination = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                write_json(destination / 'metis_resume.json', contract)
                write_json(destination / 'metis_selection.json', {'best': best, 'selection': selection_contract})
                write_json(Path(args.output_dir) / 'index.json', {
                    'last': destination.name, 'best': best, 'global_step': state.global_step,
                    'selection': selection_contract})
                event(run_dir, 'checkpoint_saved', step=state.global_step, checkpoint=str(destination))
            return control

    class TaskTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                head_ids = {id(p) for p in reranker.score_head.parameters()} if hasattr(reranker, 'score_head') else set()
                decay_names = set(self.get_decay_parameter_names(self.model))
                groups, descriptions = [], []
                for is_head in (False, True):
                    for decay in (False, True):
                        named = [(name, p) for name, p in self.model.named_parameters()
                                 if p.requires_grad and (id(p) in head_ids) == is_head
                                 and (name in decay_names) == decay]
                        if not named:
                            continue
                        lr = head_lr if is_head and head_lr is not None else self.args.learning_rate
                        weight_decay = self.args.weight_decay if decay else 0.0
                        groups.append({'params': [p for _, p in named], 'lr': lr, 'weight_decay': weight_decay})
                        descriptions.append({'role': 'score_head' if is_head else 'backbone',
                                             'learning_rate': lr, 'weight_decay': weight_decay,
                                             'parameters': sum(p.numel() for _, p in named),
                                             'names': [name for name, _ in named]})
                self.optimizer = torch.optim.AdamW(groups, lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
                write_json(run_dir / 'optimizer_groups.json', descriptions)
            return self.optimizer

    checkpoint_dir = run_dir / 'checkpoints'
    save_steps = int(train_cfg.get('save_steps', 100))
    args = TrainingArguments(
        output_dir=str(checkpoint_dir), num_train_epochs=float(train_cfg.get('epochs', 1)),
        max_steps=int(train_cfg.get('max_steps', -1)),
        learning_rate=float(train_cfg.get('learning_rate', 2e-6)),
        per_device_train_batch_size=int(train_cfg.get('batch_size', 1)),
        per_device_eval_batch_size=int(train_cfg.get('batch_size', 1)),
        gradient_accumulation_steps=int(train_cfg.get('gradient_accumulation_steps', 1)),
        save_strategy='steps', save_steps=save_steps,
        eval_strategy='steps' if validation and not selection_enabled else 'no', eval_steps=save_steps,
        logging_steps=1, report_to=[], disable_tqdm=True,
        remove_unused_columns=False, label_names=['labels'],
        prediction_loss_only=True, save_safetensors=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        use_cpu=model_cfg.get('device', 'cpu') == 'cpu', seed=seed, data_seed=seed,
        weight_decay=float(train_cfg.get('weight_decay', 0.0)),
        warmup_ratio=float(train_cfg.get('warmup_ratio', 0.0)),
        bf16=precision == 'bf16', fp16=False, optim='adamw_torch',
    )
    if args.world_size != 1:
        raise ValueError('This trainer currently supports one process; distributed selection is not implemented')
    trainer = TaskTrainer(model=model, args=args, train_dataset=train_samples, eval_dataset=validation,
                      data_collator=_collate, callbacks=[SaveContract()])
    result = trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()
    if not trainer.is_world_process_zero():
        return run_dir / 'exports' / 'final'
    parameter_after = _parameter_snapshot(model)
    updates = {}
    for name, before in parameter_before.items():
        after = parameter_after[name]
        updates[name] = {'numel': before['numel'], 'role': before['role'], 'trainable': before['trainable'],
                         'dtype': after['dtype']}
        if before['trainable']:
            updates[name].update({'initial_sha256': before['sha256'], 'final_sha256': after['sha256'],
                                  'changed': before['sha256'] != after['sha256']})
        else:
            updates[name].update({'initial_version': before['version'], 'final_version': after['version'],
                                  'version_changed': before['version'] != after['version']})
    update_summary = {'trainable_tensors': sum(x['trainable'] for x in updates.values()),
                      'changed_trainable_tensors': sum(x.get('changed', False) for x in updates.values()),
                      'changed_head_tensors': sum(x.get('changed', False) for x in updates.values()
                                                 if x['role'] == 'score_head'),
                      'changed_backbone_tensors': sum(x.get('changed', False) for x in updates.values()
                                                     if x['role'] == 'backbone'),
                      'frozen_tensors_with_version_changes': sum(x.get('version_changed', False) for x in updates.values()),
                      'frozen_check': 'requires_grad=false, excluded from optimizer; PyTorch version counters, not full value hashes'}
    write_json(run_dir / 'parameter_updates.json', {'summary': update_summary, 'parameters': updates})
    destination = export_model(run_dir / 'exports' / 'final', {
        'kind': 'final_training_state', 'global_step': trainer.state.global_step,
        'epoch': trainer.state.epoch, 'selected_for_use': not selection_enabled,
        'compute_precision': precision})
    selected = destination
    if selection_enabled:
        if best is None:
            raise ValueError('No validation selection was completed; final weights were exported for recovery')
        selected = run_dir / 'exports' / 'best'
        shutil.copytree(best['artifact'], selected, dirs_exist_ok=True)
        event(run_dir, 'model_exported', step=best['global_step'], export=str(selected),
              selection='best_validation_metric', metric_value=best['metric_value'])
    write_json(run_dir / 'training_metrics.json', result.metrics | {
        'validation_loss_subset': validation_stats, 'validation_history': validation_history,
        'initial_validation': initial_validation,
        'best': best, 'sampling': sampling_summary, 'parameter_updates': update_summary,
        'compute_precision': precision, 'master_dtype': 'float32', 'selected_artifact': str(selected)})
    event(run_dir, 'model_exported', step=trainer.state.global_step, export=str(destination),
          selection='final_training_state')
    return selected
