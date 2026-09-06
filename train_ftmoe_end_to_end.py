"""Serial, independent full-model training with paired last/best evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '3'
import numpy as np
import psutil
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from recovery.PreGANSrc.src.ftmoe_end_to_end import AblationConfig, FTMoEEndToEnd
from recovery.PreGANSrc.src.ftmoe_context import enable_moe_context
from recovery.PreGANSrc.src.ftmoe_expert_regularization import enable_expert_dropout
from recovery.PreGANSrc.src.ftmoe_fusion_controls import GatedCrossFusion
from recovery.PreGANSrc.src.ftmoe_source_fusion import SourceAttentionFusion, SourceGatedFusion
from train_ftmoe_ablation_existing import loss_fn

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts/ftmoe_end_to_end'
METRICS = ('f1', 'diagnosis_hr_at_100pct', 'diagnosis_ndcg_at_100pct')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    temp.replace(path)


def save_pt(path, state):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def resources():
    return {'available_ram_gib': psutil.virtual_memory().available / 2**30,
            'free_disk_gib': shutil.disk_usage(ROOT).free / 2**30,
            'rss_gib': psutil.Process().memory_info().rss / 2**30}


def guard():
    while True:
        state = resources()
        if state['free_disk_gib'] < 20:
            raise RuntimeError('Disk guard: free disk below 20 GiB')
        if state['available_ram_gib'] >= 4.5:
            return state
        print('RAM_GUARD_WAIT ' + json.dumps(state), flush=True)
        time.sleep(15)


def load_data(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    t = np.load(directory / 'time_series.npy')
    g = np.load(directory / 'container_demand_series.npy')
    schedules = np.load(directory / 'schedule_series.npy').astype('float32')
    y = np.load(directory / 'labels.npy').astype('int64')
    train = manifest['train_blocks']
    time_scale = np.maximum(t[train].reshape(-1, 112).max(axis=0), 1e-8)
    graph_scale = np.maximum(g[train].reshape(-1, 16, 7).max(axis=(0, 1)), 1e-8)
    t = (t / time_scale).reshape(*t.shape[:2], 16, 7).astype('float32')
    g = (g.reshape(*g.shape[:2], 16, 7) / graph_scale).astype('float32')
    length = t.shape[1]
    indices = np.maximum(np.arange(length)[:, None] - 11 + np.arange(12)[None], 0)
    blocks = []
    for block in range(len(t)):
        x = torch.from_numpy(t[block, indices].transpose(0, 2, 1, 3).copy())
        graph_x = torch.from_numpy(g[block, indices].transpose(0, 2, 1, 3).copy())
        schedule = torch.from_numpy(schedules[block, indices].copy())
        labels = torch.from_numpy(y[block].copy())
        blocks.append((x, graph_x, schedule, labels))
    training = tuple(torch.cat([blocks[b][column] for b in train]) for column in range(4))
    validation = {str(manifest['seeds'][b]): blocks[b] for b in manifest['validation_blocks']}
    normalization = {'time_scale': time_scale.tolist(), 'graph_scale': graph_scale.tolist()}
    if 'host_capacities' in manifest:
        normalization['graph_host_capacity'] = (np.asarray(manifest['host_capacities']) /
                                                graph_scale[[0, 1, 4]]).tolist()
    return training, validation, normalization, manifest


def metric_arrays(probability, class_probability, labels):
    anomaly = labels > 0
    pred = probability >= .5
    predicted_class = class_probability.argmax(axis=1) + 1
    tp = int((pred & anomaly).sum()); fp = int((pred & ~anomaly).sum())
    fn = int((~pred & anomaly).sum()); tn = int((~pred & ~anomaly).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    score = probability * class_probability.max(axis=1)
    relevance = (anomaly & (predicted_class == labels)).astype(float)
    order = np.argsort(-score)[:100]
    discount = 1 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((relevance[order] * discount).sum())
    historical_ideal = float((np.sort(relevance)[::-1][:100] * discount).sum())
    # Standard diagnostic ranking over all (host-step, resource-class) pairs.
    pair_scores = (probability[:, None] * class_probability).ravel()
    pair_relevance = np.zeros_like(class_probability)
    positive = np.flatnonzero(anomaly)
    pair_relevance[positive, labels[positive] - 1] = 1
    pair_relevance = pair_relevance.ravel()
    pair_order = np.argsort(-pair_scores)[:100]
    pair_discount = 1 / np.log2(np.arange(2, len(pair_order) + 2))
    ideal_count = min(len(positive), 100)
    pair_dcg = float((pair_relevance[pair_order] * pair_discount).sum())
    # Each anomalous host-step has exactly one dominant-resource label.
    # TranAD @100% therefore means top-1 resource diagnosis, not 100 events.
    # A stable class-index tie break is used for both metrics and disclosed.
    diagnosis_hit = float((predicted_class[anomaly] == labels[anomaly]).mean()) if anomaly.any() else 0.
    return {'f1': f1, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'diagnosis_hr_at_100pct': diagnosis_hit,
            'diagnosis_ndcg_at_100pct': diagnosis_hit,
            'hr_at_100': float(relevance[order].sum() / max(ideal_count, 1)),
            'ndcg_at_100': dcg / max(historical_ideal, 1e-12),
            'diagnostic_pair_ndcg_at_100': pair_dcg / max(float(pair_discount[:ideal_count].sum()), 1e-12),
            'pr_auc': float(average_precision_score(anomaly, probability)) if anomaly.any() else 0.,
            'positive_class_macro_f1': float(f1_score(labels[anomaly], predicted_class[anomaly],
                 labels=[1, 2, 3], average='macro', zero_division=0)) if anomaly.any() else 0.,
            'class_confusion': confusion_matrix(labels[anomaly], predicted_class[anomaly], labels=[1, 2, 3]).tolist(),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


@torch.no_grad()
def evaluate(model, blocks):
    model.eval()
    per_block = {}
    for key, (x, graph_x, schedule, y) in blocks.items():
        probabilities, classes = [], []
        for start in range(0, len(x), 64):
            output = model(x[start:start + 64], schedule[start:start + 64], graph_x[start:start + 64])
            probabilities.append(output['detection_logits'].softmax(-1)[..., 1].flatten())
            classes.append(output['class_logits'].softmax(-1).reshape(-1, 3))
        per_block[key] = metric_arrays(torch.cat(probabilities).numpy(), torch.cat(classes).numpy(), y.numpy().ravel())
    numeric = [key for key, value in next(iter(per_block.values())).items() if isinstance(value, (float, int))]
    mean = {key: float(np.mean([value[key] for value in per_block.values()])) for key in numeric}
    mean['score'] = float(np.mean([mean[key] for key in METRICS]))
    return {'mean': mean, 'per_replay': per_block}


def create_model(variant, seed, normalization=None, moe_context='none', expert_dropout=0., fusion='attention'):
    if fusion not in ('attention','gated','source_attention','source_gated'):
        raise ValueError(fusion)
    if moe_context not in ('none', 'local', 'global'):
        raise ValueError(moe_context)
    if not 0 <= expert_dropout < 1:
        raise ValueError(expert_dropout)
    torch.manual_seed(seed)
    model = FTMoEEndToEnd(variant, AblationConfig(experts=4, moe_residual_initial=0.,
        eagate_residual_initial=.5, graph_residual_initial=0., cmha_residual_initial=0.)).float()
    if model.cmha is not None and fusion=='gated':
        model.cmha = GatedCrossFusion(model.cmha)
    if model.cmha is not None and fusion=='source_attention':
        model.cmha = SourceAttentionFusion(model.cmha)
    if model.cmha is not None and fusion=='source_gated':
        model.cmha = SourceGatedFusion(model.cmha)
    if model.moe is not None and moe_context != 'none':
        enable_moe_context(model.moe, moe_context)
    if expert_dropout:
        for name in ('moe', 'eagate'):
            if getattr(model, name) is not None:
                enable_expert_dropout(getattr(model, name), expert_dropout)
    if model.graph_encoder is not None and normalization and 'graph_host_capacity' in normalization:
        model.graph_encoder.host_capacity.copy_(torch.tensor(normalization['graph_host_capacity'], dtype=torch.float32))
    return model


def train_one(args, variant, seed, training, validation, normalization, run_root):
    out = run_root / f'{variant}_seed{seed}'
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    out.mkdir(exist_ok=True)
    guard()
    random.seed(seed); np.random.seed(seed)
    model = create_model(variant, seed, normalization, args.moe_context, args.expert_dropout, args.fusion)
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    assert {id(p) for p in model.parameters()} == {id(p) for group in optimizer.param_groups for p in group['params']}
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=5),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - 5, 1), eta_min=1e-5)], milestones=[5])
    generator = torch.Generator().manual_seed(seed)
    best_score = -float('inf'); best_epoch = 0; first_epoch = 1
    history = []; elapsed_before = 0.
    initial = evaluate(model, validation)
    torch.manual_seed(seed + 10000)
    if (out / 'resume.pt').exists():
        resume = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler']); generator.set_state(resume['generator'])
        torch.set_rng_state(resume['torch_rng']); np.random.set_state(resume['numpy_rng']); random.setstate(resume['random_rng'])
        first_epoch = resume['epoch'] + 1; best_epoch = resume['best_epoch']; best_score = resume['best_score']
        history = resume['history']; elapsed_before = resume['elapsed_seconds']
    start_time = time.perf_counter()
    x, graph_x, schedule, labels = training
    actual_epochs = args.stop_after or args.epochs
    if not 1 <= actual_epochs <= args.epochs:
        raise ValueError('stop-after must be within the registered scheduler horizon')
    for epoch in range(first_epoch, actual_epochs + 1):
        sample = guard()
        model.train()
        model.set_eagate_temperature(1. - .8 * (epoch - 1) / max(args.epochs - 1, 1))
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.; routing_balance = {}
        for batch in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            output = model(x[batch], schedule[batch], graph_x[batch])
            loss = loss_fn(model, output, labels[batch], .7, .3, 0., .01, 2., .5)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())
            for name, prob in model.routing_outputs.items():
                routing_balance[name] = float(((prob.detach().mean((0, 1)) - .25) ** 2).mean())
        scheduler.step()
        validation_metrics = evaluate(model, validation)
        score = validation_metrics['mean']['score']
        elapsed = elapsed_before + time.perf_counter() - start_time
        row = {'epoch': epoch, 'loss': total_loss / len(order.split(32)), 'seconds': elapsed,
               'lr': optimizer.param_groups[0]['lr'], **sample, **validation_metrics['mean']}
        row.update({f'balance_{key}': value for key, value in routing_balance.items()})
        history.append(row)
        checkpoint = {'variant': variant, 'seed': seed, 'epoch': epoch, 'model': model.state_dict(),
                      'validation': validation_metrics, 'normalization': normalization,
                      'model_options': {'moe_context': args.moe_context, 'expert_dropout': args.expert_dropout,
                                        'fusion': args.fusion}}
        if score > best_score:
            best_score = score; best_epoch = epoch
            save_pt(out / 'best.pt', checkpoint)
        save_pt(out / 'last.pt', checkpoint)
        if args.save_all_epochs:
            epoch_dir = out / 'checkpoints_by_epoch'
            epoch_dir.mkdir(exist_ok=True)
            save_pt(epoch_dir / f'epoch{epoch:03d}.pt', checkpoint)
        save_pt(out / 'resume.pt', dict(checkpoint, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            generator=generator.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            random_rng=random.getstate(), best_epoch=best_epoch, best_score=best_score, history=history, elapsed_seconds=elapsed))
        write_json(out / 'progress.json', row)
        with (out / 'epochs.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        if epoch == 1 or epoch % 10 == 0 or epoch == actual_epochs:
            print(f'{variant} s{seed} ep{epoch}/{actual_epochs} F1={row["f1"]:.4f} HR100pct={row["diagnosis_hr_at_100pct"]:.4f} '
                  f'NDCG100pct={row["diagnosis_ndcg_at_100pct"]:.4f} best_ep={best_epoch} seconds={elapsed:.1f}', flush=True)
    last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    baseline_score = max((row['score'] for row in history[:-10]), default=history[0]['score'])
    recent_gain = max(row['score'] for row in history[-10:]) - baseline_score
    summary = {'variant': variant, 'seed': seed, 'configuration': vars(args), 'best_epoch': best_epoch,
               'parameters': sum(p.numel() for p in model.parameters()), 'all_parameters_trainable': True,
               'initial': initial, 'A_last': last['validation'], 'B_best': best['validation'],
               'last10_best_score_gain': recent_gain, 'elapsed_seconds': history[-1]['seconds'],
               'peak_rss_gib': max(row['rss_gib'] for row in history)}
    write_json(out / 'summary.json', summary)
    print('COMPLETE ' + json.dumps({'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
                                  'A': last['validation']['mean'], 'B': best['validation']['mean']}), flush=True)
    return summary


def aggregate(summaries):
    result = {}
    for protocol in ('A_last', 'B_best'):
        variants = {}
        for variant in sorted({row['variant'] for row in summaries}):
            selected = [row for row in summaries if row['variant'] == variant]
            variants[variant] = {metric: {'mean': float(np.mean([row[protocol]['mean'][metric] for row in selected])),
                'std': float(np.std([row[protocol]['mean'][metric] for row in selected], ddof=1)) if len(selected) > 1 else 0.}
                for metric in METRICS}
        chain = {metric: len(variants) == 5 and all(variants[f'v{i+1}'][metric]['mean'] > variants[f'v{i}'][metric]['mean'] for i in range(4)) for metric in METRICS}
        result[protocol] = {'variants': variants, 'chain': chain}
    return result


def preflight(training, validation, normalization=None, moe_context='none', expert_dropout=0., fusion='attention'):
    reports = []
    shared = {}
    for variant in ('v0', 'v1', 'v2', 'v3', 'v4'):
        guard()
        model = create_model(variant, 1, normalization, moe_context, expert_dropout, fusion)
        for name, parameter in model.named_parameters():
            if name in shared:
                assert torch.equal(parameter, shared[name]), name
            else:
                shared[name] = parameter.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        x, graph_x, schedule, labels = (value[20:28] for value in training)
        for _ in range(4):
            model.train(); optimizer.zero_grad()
            output = model(x, schedule, graph_x)
            loss = loss_fn(model, output, labels, .7, .3, 0., .01, 2., .5)
            loss.backward(); optimizer.step()
        changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]
        if model.moe is not None and moe_context != 'none':
            assert 'moe.context_projection.weight' in changed, variant
        for prefix in ('encoder', 'moe', 'eagate', 'graph_encoder', 'cmha'):
            if getattr(model, prefix, None) is not None:
                assert any(name.startswith(prefix + '.') for name in changed), (variant, prefix)
        reports.append({'variant': variant, 'changed_parameter_tensors': len(changed), 'total_parameter_tensors': len(before),
                        'finite_loss': bool(torch.isfinite(loss)), 'routers': list(model.routing_outputs),
                        'all_parameters_trainable': all(p.requires_grad for p in model.parameters())})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', default='artifacts/ftmoe_end_to_end/data/protocol_001')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--stop-after', type=int, default=None,
                        help='Shared stopping epoch while retaining the registered learning-rate horizon')
    parser.add_argument('--save-all-epochs', action='store_true',
                        help='Retain exact candidate checkpoints for shared-budget selection')
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--moe-context', choices=['none', 'local', 'global'], default='none')
    parser.add_argument('--expert-dropout', type=float, default=0.)
    parser.add_argument('--fusion', choices=['attention','gated','source_attention','source_gated'], default='attention')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 6])
    parser.add_argument('--variants', nargs='+', default=['v0', 'v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    guard()
    directory = ROOT / args.data
    training, validation, normalization, manifest = load_data(directory)
    run_root = ART / 'runs' / args.run
    run_root.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__).resolve(), ROOT / 'recovery/PreGANSrc/src/ftmoe_end_to_end.py',
             ROOT / 'recovery/PreGANSrc/src/ftmoe_context.py',
             ROOT / 'recovery/PreGANSrc/src/ftmoe_expert_regularization.py',
             ROOT / 'recovery/PreGANSrc/src/ftmoe_fusion_controls.py',
             ROOT / 'recovery/PreGANSrc/src/ftmoe_source_fusion.py',
             ROOT / 'recovery/PreGANSrc/src/ftmoe_ablation.py', ROOT / 'train_ftmoe_ablation_existing.py']
    configuration = {'arguments': vars(args), 'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
                     'manifest_sha256': sha(directory / 'manifest.json')}
    if (run_root / 'configuration.json').exists():
        existing = json.loads((run_root / 'configuration.json').read_text())
        assert existing == configuration, 'Run configuration changed; use a new run name'
    else:
        write_json(run_root / 'configuration.json', configuration)
        snapshot = run_root / 'source_snapshot'; snapshot.mkdir()
        for path in files:
            shutil.copy2(path, snapshot / path.name)
    if args.preflight_only:
        report = preflight(training, validation, normalization, args.moe_context, args.expert_dropout, args.fusion)
        write_json(run_root / 'preflight.json', report)
        print(json.dumps(report), flush=True)
        return
    summaries = []
    for variant in args.variants:
        for seed in args.seeds:
            summaries.append(train_one(args, variant, seed, training, validation, normalization, run_root))
            write_json(run_root / 'aggregate.json', aggregate(summaries))
    print(json.dumps(aggregate(summaries), indent=2), flush=True)


if __name__ == '__main__':
    main()
