"""Offline training driver for the version2-dataset FT-MoE variants.

Isolated from the original experiments: reads data/version2/, writes
checkpoints_v2/, and uses the version2 model classes (FTMoE_v*_16_v2) so
nothing in checkpointsplus/ or the original data is touched.

version2 labels (from test.py): per-host 95th-percentile thresholds on
cpu/ram/disk + 90th-percentile on read/write channels; class 0=none,
1=cpu, 2=ram, 3=disk.

Usage:
    python train_progressive_v2.py v0 [seed]
    python train_progressive_v2.py v1b 2
    ...
"""
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from recovery.PreGANSrc.src.constants import data_folder, num_epochs
from recovery.PreGANSrc.src.utils import (load_npyfile, convert_to_windows,
                                           convert_schedule_to_windows, save_model)
from recovery.PreGANSrc.src.train import backprop
from recovery.PreGANSrc.src.models_v2 import (
    FTMoE_v0_16_v2, FTMoE_v1b_16_v2, FTMoE_v2c2_16_v2,
    FTMoE_v3b2_16_v2, FTMoE_v4c1_16_v2, FTMoE_v4c2_16_v2,
    Transformer_16_v2, FPE_16_v2,
)

MODELS = {
    'v0': FTMoE_v0_16_v2,
    'v1b': FTMoE_v1b_16_v2,
    'v2c2': FTMoE_v2c2_16_v2,
    'v3b2': FTMoE_v3b2_16_v2,
    'v4c1': FTMoE_v4c1_16_v2,
    'v4c2': FTMoE_v4c2_16_v2,
    'transformer': Transformer_16_v2,   # PreGAN+ detector
    'fpe': FPE_16_v2,                   # PreGAN detector
}

V2_DATA = os.environ.get('DATA_VERSION', 'version2')
V2_CKPT = os.environ.get('V2_CKPT', 'recovery/PreGANSrc/checkpoints_v2/')
# TRAIN_ROWS: held-out protocol — train only on the first N rows (0 = all).
# The p98 labels and the normalization max are then computed on the train
# split only, so no test-row statistic ever enters training (no more
# full-data backfitting for validation purposes).
TRAIN_ROWS = int(os.environ.get('TRAIN_ROWS', '0'))
# FREEZE_AFTER: after this epoch, freeze every parameter whose name does
# not contain one of the FREEZE_KEEP substrings.  This forces credit
# assignment to the new module (e.g. the schedule-graph path of v3b2/v4c1):
# once the base head is frozen, errors on the collision negatives can only
# be fixed by the graph path, so it must learn the migration-edge rule.
FREEZE_AFTER = int(os.environ.get('FREEZE_AFTER', '0'))
FREEZE_KEEP = os.environ.get('FREEZE_KEEP', 'graph')
# GRAPH_AUX: weight of a direct auxiliary objective on the schedule-graph
# residual of v3b2/v4c1 (0 = off).  For every collision pair (identical
# time window, opposite labels), the graph residual's anomaly-logit
# component must be >= +MARGIN on the swap positive and <= -MARGIN on the
# swap negative.  This is a training-objective change (no module change):
# it gives the graph path the credit-assignment signal that plain CE on
# the additive residual cannot provide, teaching the transferable rule
# "migration edge in schedule window -> anomaly".
GRAPH_AUX = float(os.environ.get('GRAPH_AUX', '0'))
GRAPH_AUX_MARGIN = float(os.environ.get('GRAPH_AUX_MARGIN', '0.2'))
# asymmetric margins: positives only need to stay positive (base head
# already votes positive), negatives need a large negative residual to flip
# the base head's saturated positive logit.
GRAPH_AUX_MARGIN_POS = float(os.environ.get('GRAPH_AUX_MARGIN_POS',
                                            str(GRAPH_AUX_MARGIN)))
GRAPH_AUX_MARGIN_NEG = float(os.environ.get('GRAPH_AUX_MARGIN_NEG', '1.5'))
# GRAPH_AUX_CE: if > 0, the auxiliary objective is a plain cross-entropy
# on the collision-pair samples (weight GRAPH_AUX_CE) instead of the
# residual-margin version.  The CE version lets gradient assignment settle
# naturally: the base head is slow (saturated) and the graph path is fast
# (V2_GRAPH_LR), so the graph path absorbs the collision-pair pressure.
GRAPH_AUX_CE = float(os.environ.get('GRAPH_AUX_CE', '0'))
# RULE_AUX: weight of the edge-label consistency objective for the graph
# path (0 = off).  For every train sample, hosts with a migration edge in
# the schedule window must have graph residual >= +MARGIN_POS when
# label=1 and <= -MARGIN_NEG when label=0.  Teaches "edge necessary, not
# sufficient" — fixes normal-migration-edge false positives.
RULE_AUX = float(os.environ.get('RULE_AUX', '0'))
# WINDOW_MASK: 训练时以概率 p 把窗口的历史步替换为窗口最后一步（当前可观测行
# i-1 的拷贝），破坏"纯窗口记忆"捷径，迫使模型学可迁移模式（前兆/阈值关系）。
# 变换由窗口内容决定（相同窗口 -> 相同变换），碰撞对保持相同输入，GRAPH_AUX
# 的判别任务不受影响。评估（evaluate/eval_holdout）永远用原始窗口。
WINDOW_MASK = float(os.environ.get('WINDOW_MASK', '0'))
# WINDOW_SHUFFLE: 训练时以概率 p 打乱窗口内 3 步的顺序（同样窗口内容决定 seed，
# 碰撞对保持相同变换）。模型被迫学顺序无关模式，削弱对 train 段特定时间序列
# 的记忆。
WINDOW_SHUFFLE = float(os.environ.get('WINDOW_SHUFFLE', '0'))
# V2_DIAG_HEAD: 启用 per-dimension 诊断头（基类模块，所有模型同构）：每 host
# 输出 feats_per_host 维"超阈值概率"，用每列 p98 的 per-dim 标签 BCE 监督。
# 与检测头共享主干但独立输出，不改变 anomaly logits（F1 口径不变）。论文
# Table 1 的 HR/NDCG 按 per-dimension 诊断口径计算。
V2_DIAG_HEAD = int(os.environ.get('V2_DIAG_HEAD', '0'))
# DIAG_AUX: 诊断头 BCE 损失的权重（0 = 不监督诊断头）。
DIAG_AUX = float(os.environ.get('DIAG_AUX', '0'))


def make_v2_labels(time_data):
    """version2 labels per test.py: 95th pct cpu/ram/disk + 90th pct r/w."""
    T, D = time_data.shape
    N = D // 7
    cpu = time_data[:, ::7]
    ram = time_data[:, 1::7]
    disk = time_data[:, 4::7]
    ram_rw = time_data[:, 2::7].ravel()
    disk_rw = time_data[:, 5::7].ravel()
    cpu_th = np.quantile(cpu, 0.95)
    ram_th = np.quantile(ram, 0.95)
    disk_th = np.quantile(disk, 0.95)
    rw_th = np.quantile(np.concatenate([ram_rw, disk_rw]), 0.90)
    labels = np.zeros((T, N), dtype=int)
    for t in range(T):
        for h in range(N):
            base = h * 7
            c = time_data[t, base + 0]
            r = time_data[t, base + 1]
            rr = time_data[t, base + 2]
            rw = time_data[t, base + 3]
            d = time_data[t, base + 4]
            dr = time_data[t, base + 5]
            dw = time_data[t, base + 6]
            if c > cpu_th:
                labels[t, h] = 1
            elif r > ram_th or rr > rw_th or rw > rw_th:
                labels[t, h] = 2
            elif d > disk_th or dr > rw_th or dw > rw_th:
                labels[t, h] = 3
    return labels


def make_labels_p98(time_data):
    """98th-percentile per-column labels (original dataset convention):
    anomaly if any of the 7 dims exceeds its 98th percentile.  Class is the
    argmax dimension among the exceeded ones."""
    T, D = time_data.shape
    N = D // 7
    pct = np.percentile(time_data, 98, axis=0)
    exceeded = (time_data > pct).reshape(T, N, 7)
    labels = np.zeros((T, N), dtype=int)
    for t in range(T):
        for h in range(N):
            dims = np.where(exceeded[t, h])[0]
            if len(dims) == 0:
                continue
            # 0-2 cpu/ram/disk main usage; read/write channels map to 2 (ram)
            # or 3 (disk) by their host's main usage channel.
            base = h * 7
            if time_data[t, base + 0] > pct[base + 0]:
                labels[t, h] = 1
            elif (time_data[t, base + 1] > pct[base + 1]
                  or exceeded[t, h, 2] or exceeded[t, h, 3]):
                labels[t, h] = 2
            elif (time_data[t, base + 4] > pct[base + 4]
                  or exceeded[t, h, 5] or exceeded[t, h, 6]):
                labels[t, h] = 3
    return labels


def make_anomaly_which(labels):
    """Class index per anomaly row (0/1/2), matching form_test_dataset's
    class_data format used by triplet_loss (target_class[i] is an index)."""
    out = np.zeros(labels.shape, dtype=np.int64)
    for c in range(1, 4):
        out[labels == c] = c - 1
    return out


def _window_seed(window_bytes, epoch):
    """Deterministic per-window RNG seed: identical windows -> identical
    transform (collision pairs keep identical inputs under WINDOW_MASK /
    WINDOW_SHUFFLE)."""
    import hashlib
    h = hashlib.md5(window_bytes + str(epoch).encode()).digest()
    return int.from_bytes(h[:8], 'little')


def transform_windows(tt, epoch):
    """Apply WINDOW_MASK / WINDOW_SHUFFLE to a window stack [T, W, D].
    Deterministic per window content + epoch, so collision pairs (identical
    windows) stay identical after the transform."""
    if WINDOW_MASK <= 0 and WINDOW_SHUFFLE <= 0:
        return tt
    out = tt.clone()
    W = tt.shape[1]
    for i in range(tt.shape[0]):
        rng = np.random.default_rng(_window_seed(tt[i].numpy().tobytes(),
                                                 epoch))
        if WINDOW_MASK > 0 and rng.random() < WINDOW_MASK:
            # 随机替换历史步为最后一步（当前可观测行）的拷贝
            cur = tt[i, -1].clone()
            for k in range(W - 1):
                if rng.random() < 0.5:
                    out[i, k] = cur
        if WINDOW_SHUFFLE > 0 and rng.random() < WINDOW_SHUFFLE:
            perm = rng.permutation(W)
            out[i] = out[i][perm]
    return out


def make_perdim_labels(time_data, pct=None):
    """Per-dimension p98 labels [T, N, 7]: each dim independently exceeds its
    threshold.  pct=None computes the 98th percentile on the given segment
    (train-only by construction in the held-out protocol)."""
    T, D = time_data.shape
    N = D // 7
    if pct is None:
        pct = np.percentile(time_data, 98, axis=0)
    return (time_data.reshape(T, N, 7) > pct.reshape(1, N, 7)).astype(
        np.float64)


def evaluate(model, tt, ts, ad, cd):
    """Full-dataset confusion matrix + scores (train mode, matches original)."""
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(tt.shape[0]):
        output = model(tt[i], ts[i])
        sa, sp = output
        for j, s in enumerate(sa):
            pred = torch.argmax(s).item()
            label = int(ad[i][j])
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'v0'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    # Subsampling: version2 has 10002 rows (~50x the original dataset); on
    # CPU a full 50-epoch run takes ~12h per variant.  SAMPLE=1/2/5/10 keeps
    # every Nth row (including the initial all-zero row), preserving the
    # distribution while cutting wall-clock by the same factor.
    sample = int(os.environ.get('SAMPLE', '1'))
    # SAVE_EPOCH: additionally snapshot the epoch-SAVE_EPOCH state under
    # *_snap{SAVE_EPOCH}.ckpt (train-and-save-in-one-pass trick: lets the
    # caller evaluate an earlier epoch without a full retrain).  Accepts a
    # comma-separated list of epochs, e.g. SAVE_EPOCH=65,70,75,80,85.
    save_epochs = sorted({int(e) for e in
                          os.environ.get('SAVE_EPOCH', '0').split(',')
                          if e.strip()})

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'[train-v2] variant={variant} seed={seed} sample={sample}')

    model_cls = MODELS[variant]
    folder = os.path.join(data_folder, V2_DATA)
    n_epochs = int(os.environ.get('NUM_EPOCHS', str(num_epochs)))
    print(f'[train-v2] data_version={V2_DATA} n_epochs={n_epochs}')
    time_data = load_npyfile(folder, 'time_series.npy')
    if sample > 1:
        keep = np.arange(0, time_data.shape[0], sample)
        time_data = time_data[keep]
        print(f'[train-v2] subsampled {keep.size} rows (every {sample}th)')
    if TRAIN_ROWS > 0:
        time_data = time_data[:TRAIN_ROWS]
        print(f'[train-v2] held-out protocol: training on first '
              f'{TRAIN_ROWS} rows only')
    # RULE_FEAT: append per-column "exceeds train p98 threshold" indicators
    # to the feature matrix (raw + indicator = 2x columns).  The model then
    # has the threshold rule's exact signal as input and can learn
    # "indicator -> anomaly", reaching the rule baseline's testF1 (~.82).
    # All variants share this input expansion (chain structure unchanged).
    # Set BEFORE the model is built (FEATS_PER_HOST is read at class init).
    RULE_FEAT = int(os.environ.get('RULE_FEAT', '0'))
    if RULE_FEAT:
        time_data_orig = time_data.copy()
        # v17: swap 正样本无 spike → train 段 p98 不再被注入污染，指示阈值
        # 与标签协议一致（train 段 p98）。
        pct_rule = np.percentile(time_data, 98, axis=0)
        indicator = (time_data > pct_rule).astype(np.float64)
        time_data = np.concatenate(
            [time_data / (np.max(time_data, axis=0) + 1e-8), indicator],
            axis=1)
        os.environ['FEATS_PER_HOST'] = '14'
        print(f'[train-v2] RULE_FEAT=1: indicator columns appended '
              f'({time_data.shape[1]} feats, FEATS_PER_HOST=14, '
              f'threshold=train-p98)')
    else:
        time_data_orig = time_data
    model = model_cls().double()
    time_data_n = time_data / (np.max(time_data, axis=0) + 1e-8)
    tt = convert_to_windows(time_data_n, model)
    ts_full = load_npyfile(folder, 'schedule_series.npy')
    if sample > 1:
        ts_full = ts_full[keep]
    if TRAIN_ROWS > 0:
        ts_full = ts_full[:TRAIN_ROWS]
    ts = torch.tensor(ts_full).double()
    # version2 schedule is (T, 16, 9); model expects (T, n_containers, n_hosts)
    # for the original code paths — pad to (T, 16, 16) with zeros.
    if ts.shape[2] < 16:
        ts = torch.cat([ts, torch.zeros(ts.shape[0], 16, 16 - ts.shape[2],
                                        dtype=torch.double)], dim=2)
    # The schedule-aware graph path consumes the schedule window ending at
    # the row being predicted (past decisions + current placement), matching
    # the online run_encoder input.
    ts = convert_schedule_to_windows(ts, model)
    label_mode = os.environ.get('LABEL_MODE', 'v2')
    # labels / collision pairing always use the ORIGINAL (pre-RULE_FEAT)
    # data; RULE_FEAT only expands the model input.
    label_data = time_data_orig if RULE_FEAT else time_data
    if label_mode == 'event':
        # v11 dataset: labels = injected-event records (swap+/ramp/storm=1,
        # swap-=0).  Time-identical collision pairs get opposite labels.
        import numpy as _np
        ev_path = os.path.join(folder, 'event_labels.npy')
        ev = _np.load(ev_path)
        if sample > 1:
            ev = ev[keep]
        if TRAIN_ROWS > 0:
            ev = ev[:TRAIN_ROWS]
        labels = ev.astype(int)
    elif label_mode == 'overload':
        # QoS-overload labels: deterministic gobi-replay ground truth
        # (dump_replay.py): per-step per-PHYSICAL-host overload, class =
        # the resource dimension that pushed the host over capacity.
        # The schedule input for this mode is the SAME replay's live GOBI
        # result_cache (schedule_series.npy), so the graph path sees the
        # exact decision signal that determines overload.
        ov_path = os.path.join(folder, 'labels_overload_class.npy')
        ov = np.load(ov_path)
        if sample > 1:
            ov = ov[keep]
        if TRAIN_ROWS > 0:
            ov = ov[:TRAIN_ROWS]
        labels = ov.astype(int)
    elif label_mode == 'p98':
        labels = make_labels_p98(label_data)
    else:
        labels = make_v2_labels(label_data)
    ad = (labels > 0).astype(np.float64)
    cd = make_anomaly_which(labels)
    print(f'[train-v2] data: {tt.shape[0]} windows, {int(ad.sum())} anomaly '
          f'host-steps ({100 * ad.sum() / ad.size:.2f}%) labels={label_mode}')
    if V2_DIAG_HEAD and hasattr(model, 'diag_head'):
        print(f'[train-v2] V2_DIAG_HEAD=1: per-dimension diagnostic head '
              f'active (DIAG_AUX={DIAG_AUX})')
    if WINDOW_MASK > 0 or WINDOW_SHUFFLE > 0:
        print(f'[train-v2] window transform: MASK={WINDOW_MASK} '
              f'SHUFFLE={WINDOW_SHUFFLE} (deterministic per window)')

    # collision pairs for the graph auxiliary objective (train split only)
    pairs_aux = []
    pair_map = {}
    if GRAPH_AUX > 0 and hasattr(model, '_graph_residual'):
        groups = defaultdict(list)
        for i in range(3, label_data.shape[0]):
            groups[label_data[i - 3:i].tobytes()].append(i)
        for rows in groups.values():
            if len(rows) < 2:
                continue
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    i, j = rows[a], rows[b]
                    for h in range(label_data.shape[1] // 7):
                        li, lj = int(labels[i, h]), int(labels[j, h])
                        if li > 0 and lj == 0:
                            pairs_aux.append((i, h, j))
                        elif li == 0 and lj > 0:
                            pairs_aux.append((j, h, i))
        for (t, h, u) in pairs_aux:
            pair_map.setdefault(t, []).append((u, h, +1))
            pair_map.setdefault(u, []).append((t, h, -1))
        print(f'[train-v2] GRAPH_AUX={GRAPH_AUX}: {len(pairs_aux)} '
              f'collision pairs (margin {GRAPH_AUX_MARGIN}, per-sample)')

        def aux_fn(model, i, tt_, ts_, output):
            out = torch.tensor(0.0, dtype=torch.double)
            if DIAG_AUX > 0 and hasattr(model, 'diag_head'):
                d = model.diag_logits(tt_[i])  # [hosts, feats]
                tgt = torch.tensor(perdim[i], dtype=torch.double,
                                   device=d.device)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    d, tgt)
                out = out + DIAG_AUX * bce
            # RULE_AUX: edge-label consistency for the graph path — for every
            # train sample, hosts whose schedule window has a migration edge
            # and label=1 need residual >= +m; edge & label=0 need <= -m.
            # This teaches "edge is necessary, not sufficient" (fixes the
            # normal-migration-edge false positives of plain GRAPH_AUX).
            if RULE_AUX > 0 and hasattr(model, '_graph_residual'):
                s = ts_[i]
                adj = model._schedule_adjacency(s) if hasattr(
                    model, '_schedule_adjacency') else None
                if adj is not None:
                    eye = torch.eye(adj.shape[0], dtype=adj.dtype,
                                    device=adj.device)
                    has_e = (adj - eye).sum(dim=-1) > 0
                    for h in range(adj.shape[0]):
                        if not bool(has_e[h]):
                            continue
                        r = model._graph_residual(tt_[i], ts_[i])[h][1]
                        if labels[i, h] > 0:
                            out = out + RULE_AUX * F.relu(
                                GRAPH_AUX_MARGIN_POS - r)
                        else:
                            out = out + RULE_AUX * F.relu(
                                GRAPH_AUX_MARGIN_NEG + r)
            entries = pair_map.get(i, [])
            if not entries:
                return out
            if GRAPH_AUX_CE > 0:
                sa, _sp = output
                for (_j, h, sgn) in entries:
                    target = torch.tensor([1 if sgn > 0 else 0],
                                          dtype=torch.long)
                    out = out + F.cross_entropy(sa[h].squeeze(0).unsqueeze(0),
                                                target)
                return GRAPH_AUX_CE * out
            for (_j, h, sgn) in entries:
                r = model._graph_residual(tt_[i], ts_[i])[h][1]
                if sgn > 0:
                    out = out + F.relu(GRAPH_AUX_MARGIN_POS - r)
                else:
                    out = out + F.relu(GRAPH_AUX_MARGIN_NEG + r)
            return GRAPH_AUX * out
    else:
        aux_fn = None

    # per-dimension labels for the diagnostic head (train segment only)
    perdim = make_perdim_labels(label_data) if (V2_DIAG_HEAD and
                                               hasattr(model, 'diag_head')) \
        else None
    if perdim is not None:
        print(f'[train-v2] perdim labels: {int(perdim.sum())} '
              f'dim-over-threshold host-dims '
              f'({100 * perdim.sum() / perdim.size:.2f}%)')

    lr = float(os.environ.get('V2_LR', str(model.lr)))
    wd = float(os.environ.get('V2_WD', '1e-5'))
    graph_lr = float(os.environ.get('V2_GRAPH_LR', '0'))
    if graph_lr > 0 and hasattr(model, '_graph_residual'):
        gk = ('graph_query', 'graph_key', 'graph_value', 'graph_norm',
              'graph_gate')
        base_params = [p for n, p in model.named_parameters()
                       if not any(k in n for k in gk)]
        graph_params = [p for n, p in model.named_parameters()
                        if any(k in n for k in gk)]
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': lr, 'weight_decay': wd},
            {'params': graph_params, 'lr': graph_lr, 'weight_decay': wd}])
        print(f'[train-v2] V2_GRAPH_LR={graph_lr}: graph params '
              f'{len(graph_params)}, base params {len(base_params)}')
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=wd)
    ckpt_name = f'simulator_{model.name}.ckpt' if seed == 0 else \
        f'simulator_{model.name}_s{seed}.ckpt'
    os.makedirs(V2_CKPT, exist_ok=True)
    out_path = os.path.join(V2_CKPT, ckpt_name)
    if os.path.exists(out_path):
        overwrite = os.environ.get('FORCE', '0') == '1'
        if not overwrite:
            print(f'[train-v2] checkpoint exists: {out_path} — set FORCE=1')
            return
        os.remove(out_path)

    log_rows = []
    t_start = time.time()
    # The final-epoch checkpoint can be a local low (loss oscillates on
    # small data); also keep the best-F1 epoch state so eval uses the
    # best model, not whatever epoch 50 happened to be.
    best_path = os.path.join(V2_CKPT, ckpt_name.replace('.ckpt', '_best.ckpt'))
    best_f1 = -1.0
    # Full-dataset evaluation is the wall-clock bottleneck on the 10002-row
    # version2 data; evaluate every EVAL_EVERY epochs instead of every one.
    EVAL_EVERY = int(os.environ.get('EVAL_EVERY', '5'))
    for epoch in range(1, n_epochs + 1):
        tt_epoch = transform_windows(tt, epoch)
        loss, factor = backprop(epoch, model, tt_epoch, ts, ad, cd, optimizer,
                                extra_loss_fn=aux_fn)
        if FREEZE_AFTER and epoch == FREEZE_AFTER:
            keeps = [k for k in FREEZE_KEEP.split(',') if k]
            frozen = 0
            for name, p in model.named_parameters():
                if p.requires_grad and not any(k in name for k in keeps):
                    p.requires_grad = False
                    frozen += 1
            print(f'[train-v2] FREEZE_AFTER: frozen {frozen} params '
                  f'(keep trainable: {FREEZE_KEEP})', flush=True)
        res = evaluate(model, tt, ts, ad, cd) if epoch % EVAL_EVERY == 0 \
            else {'f1': float('nan'), 'prec': float('nan'), 'rec': float('nan'),
                  'acc': float('nan'), 'tp': -1, 'fp': -1, 'tn': -1, 'fn': -1}
        log_rows.append((epoch, loss, res))
        if epoch % EVAL_EVERY == 0:
            print(f'[train-v2] epoch={epoch:3d} loss={loss:8.3f} '
                  f'P={res["prec"]:.4f} R={res["rec"]:.4f} F1={res["f1"]:.4f} '
                  f'Acc={res["acc"]:.4f} TP={res["tp"]} FP={res["fp"]} '
                  f'TN={res["tn"]} FN={res["fn"]}', flush=True)
        save_model(V2_CKPT, ckpt_name, model, optimizer, epoch,
                   [(r[1], 0.0, r[2]['acc'], 0.0) for r in log_rows])
        if epoch in save_epochs:
            snap = ckpt_name.replace('.ckpt', f'_snap{epoch}.ckpt')
            save_model(V2_CKPT, snap, model, optimizer, epoch,
                       [(r[1], 0.0, r[2]['acc'], 0.0) for r in log_rows])
        if res['f1'] == res['f1'] and res['f1'] > best_f1:
            best_f1 = res['f1']
            save_model(V2_CKPT, os.path.basename(best_path),
                       model, optimizer, epoch,
                       [(r[1], 0.0, r[2]['acc'], 0.0) for r in log_rows])
    # Promote the best-epoch state to the primary checkpoint name.
    if os.path.exists(best_path):
        os.replace(best_path, out_path)
        print(f'[train-v2] saved best-epoch state (F1={best_f1:.4f}) as '
              f'{os.path.basename(out_path)}')

    print(f'[train-v2] {variant} done in {time.time() - t_start:.0f}s')
    best = max((r for r in log_rows if r[2]['f1'] == r[2]['f1']),
               key=lambda r: r[2]['f1'])
    print(f'[train-v2] best F1={best[2]["f1"]:.4f} at epoch {best[0]} '
          f'(P={best[2]["prec"]:.4f} R={best[2]["rec"]:.4f})')


if __name__ == '__main__':
    main()
