"""Held-out evaluation (no more backfitting): train on cycle 1 rows
[0, TRAIN_ROWS), evaluate on the never-seen cycle 2 rows [TRAIN_ROWS, T).

Protocol details (strict):
- normalization max: computed on the TRAIN split only (fixes the old
  full-data max leak; matches the online normalizer).
- labels: truth A (deployment style) = test split's OWN p98 labels;
  truth B (strict) = train-split p98 thresholds applied to the test split
  (only train data used).  Both reported; B is the strictest "rule
  transfer" view.
- windows: test rows use past-window rows, which may include train rows —
  that is the legitimate deployment situation (past history is known).
  Collision-pair windows never cross the split (verified: block layout
  keeps t/u and their windows inside one cycle).
- collision-pair discrimination on the test split: the swap design gives
  v3/v4 an exclusive, transferable signal (migration edge); discrim>0 on
  unseen data is direct evidence of rule learning, not memorization.
- rule baseline: the train-threshold rule itself, evaluated on the test
  split (zero-cost reference; a model that cannot beat it on unseen data
  has learned nothing beyond the trivial threshold rule).

Usage:
  DATA_VERSION=expand_final_swap_v4 CHAIN_CKPT_DIR=<dir> TRAIN_ROWS=202 \
      python eval_holdout.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

os.environ['V2_MOE_EXPERTS'] = os.environ.get('V2_MOE_EXPERTS', '12')
os.environ['V2_DROPOUT'] = os.environ.get('V2_DROPOUT', '0')
os.environ['V2_GRAPH_DROPOUT'] = os.environ.get('V2_GRAPH_DROPOUT', '0')
os.environ['V2_MOE_HEAD'] = os.environ.get('V2_MOE_HEAD', '1')
os.environ['V2_PROTO_DIM'] = os.environ.get('V2_PROTO_DIM', '8')
# diagnostic head must match training (structure-affecting)
os.environ['V2_DIAG_HEAD'] = os.environ.get('V2_DIAG_HEAD', '0')

sys.path.append('recovery/PreGANSrc/')
from recovery.PreGANSrc.src.constants import data_folder
from recovery.PreGANSrc.src.utils import (load_model, load_npyfile,
                                           convert_to_windows,
                                           convert_schedule_to_windows)
from recovery.PreGANSrc.src.models_v2 import (
    FTMoE_v0_16_v2, FTMoE_v1b_16_v2, FTMoE_v2c2_16_v2,
    FTMoE_v3b2_16_v2, FTMoE_v4c1_16_v2,
)
from train_progressive_v2 import make_labels_p98

DATA_VERSION = os.environ.get('DATA_VERSION', 'expand_final_swap_v4')
CKPT_DIR = os.environ.get('CHAIN_CKPT_DIR',
                          'recovery/PreGANSrc/checkpoints_final_chain')
TRAIN_ROWS = int(os.environ.get('TRAIN_ROWS', '202'))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '1,2,6').split(',')]
VARIANTS = {
    'v0': FTMoE_v0_16_v2,
    'v1b': FTMoE_v1b_16_v2,
    'v2c2': FTMoE_v2c2_16_v2,
    'v3b2': FTMoE_v3b2_16_v2,
    'v4c1': FTMoE_v4c1_16_v2,
}
# FILTER_VARIANTS: comma list to evaluate a subset (e.g. graph-path
# variants need V2_GRAPH_EXPLICIT/SCALE envs matching their training).
_filter = os.environ.get('FILTER_VARIANTS', '')
if _filter:
    keep = set(_filter.split(','))
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in keep}


def labels_with_pct(Xi, pct):
    """make_labels_p98 with externally supplied per-column thresholds."""
    Td, Dd = Xi.shape
    Nd = Dd // 7
    exceeded = (Xi > pct).reshape(Td, Nd, 7)
    labels = np.zeros((Td, Nd), dtype=int)
    for t in range(Td):
        for h in range(Nd):
            base = h * 7
            if Xi[t, base + 0] > pct[base + 0]:
                labels[t, h] = 1
            elif (Xi[t, base + 1] > pct[base + 1]
                  or exceeded[t, h, 2] or exceeded[t, h, 3]):
                labels[t, h] = 2
            elif (Xi[t, base + 4] > pct[base + 4]
                  or exceeded[t, h, 5] or exceeded[t, h, 6]):
                labels[t, h] = 3
    return labels


def f1_counts(pred, truth):
    tp = int(((pred > 0) & (truth > 0)).sum())
    fp = int(((pred > 0) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth > 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8), tp, fp, fn


def hr_ndcg_at100(emb_list, true_cls_list, prototypes):
    """Paper metrics HitRate@100% / NDCG@100% (TranAD definition).

    P% of the number of ground-truth dimensions decides how many top
    predicted candidates are considered.  With our mutually-exclusive
    class labels (make_labels_p98 elif-priority -> one class per anomalous
    host-step), |gt dimensions| = 1, so @100% considers the top-1
    candidate: HR@100% == NDCG@100% == top-1 prototype hit rate.
    (The paper's Table 1 numbers come from per-dimension diagnosis with
    multi-dimensional ground truth, which our chain models do not emit.)
    """
    hr_c = 0
    ndcg_s = []
    n = 0
    for emb, tcls in zip(emb_list, true_cls_list):
        if tcls <= 0:
            continue
        n += 1
        distances = np.array([np.mean((emb - p) ** 2) for p in prototypes])
        ranks = np.argsort(distances)
        k = 1  # @100% of |gt dimensions| = 1 (single-class labels)
        hit = int(tcls - 1 in ranks[:k])   # class labels 1/2/3 -> idx 0/1/2
        hr_c += hit
        rel = np.zeros(len(ranks))
        rel[np.where(ranks == tcls - 1)[0][0]] = 1.0
        dcg = sum(rel[i] / np.log2(i + 2) for i in range(min(k, len(rel))))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
        ndcg_s.append(dcg / idcg if idcg > 0 else 0.0)
    return (hr_c / max(n, 1), np.mean(ndcg_s) if ndcg_s else 0.0, n)


def hr_ndcg_perdim(diag_probs, gt_perdim):
    """Paper-format HitRate@100% / NDCG@100% from the per-dimension
    diagnostic head (multi-dimensional ground truth, TranAD definition).

    diag_probs: [n, 7] per-dim anomaly probabilities (per host-step).
    gt_perdim:  [n, 7] 0/1 per-dim ground truth (dim exceeds train p98).
    @100%: k = |gt dimensions|; top-k predicted dims; HR = |hit|/k.
    """
    hr_s, ndcg_s = [], []
    n = 0
    for p, g in zip(diag_probs, gt_perdim):
        k = int(g.sum())
        if k == 0:
            continue
        n += 1
        gt_dims = set(np.where(g)[0])
        ranks = np.argsort(-p)
        topk = ranks[:k]
        hit = len(set(topk) & gt_dims)
        hr_s.append(hit / k)
        rel = np.zeros(7)
        rel[list(gt_dims)] = 1.0
        dcg = sum(rel[r] / np.log2(i + 2) for i, r in enumerate(topk))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, 7)))
        ndcg_s.append(dcg / idcg if idcg > 0 else 0.0)
    return ((np.mean(hr_s) if hr_s else 0.0),
            (np.mean(ndcg_s) if ndcg_s else 0.0), n)


def perdim_with_pct(Xi, pct):
    """Per-dimension labels with externally supplied per-column thresholds."""
    Td, Dd = Xi.shape
    Nd = Dd // 7
    return (Xi.reshape(Td, Nd, 7) > pct.reshape(1, Nd, 7)).astype(np.float64)


def collision_pairs_from_labels(lab, T, N, X):
    """(t_pos, h, u_neg) pairs with identical windows + opposite labels."""
    groups = defaultdict(list)
    for i in range(3, T):
        groups[X[i - 3:i].tobytes()].append(i)
    pairs = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                for h in range(N):
                    li, lj = int(lab[i, h]), int(lab[j, h])
                    if li > 0 and lj == 0:
                        pairs.append((i, h, j))
                    elif li == 0 and lj > 0:
                        pairs.append((j, h, i))
    return pairs


def main():
    X = np.load(os.path.join(data_folder, DATA_VERSION, 'time_series.npy'))
    S = np.load(os.path.join(data_folder, DATA_VERSION,
                             'schedule_series.npy'))
    T, D = X.shape
    N = D // 7
    assert TRAIN_ROWS < T
    split = TRAIN_ROWS

    # --- statistics from the TRAIN split only ---
    mx = np.max(X[:split], axis=0) + 1e-8
    pct_train = np.percentile(X[:split], 98, axis=0)
    # RULE_FEAT: expand input with threshold indicators (must happen
    # BEFORE model construction — FEATS_PER_HOST is read at class init).
    # v17: indicator threshold = TRAIN-split p98 (consistent with the label
    # protocol; swap positives carry no spike so train p98 is unpolluted).
    if int(os.environ.get('RULE_FEAT', '0')):
        indicator = (X > pct_train).astype(np.float64)
        X_feat = np.concatenate([X / mx, indicator], axis=1)
        os.environ['FEATS_PER_HOST'] = '14'
    else:
        X_feat = X
    time_n = X_feat / (np.max(X_feat, axis=0) + 1e-8)
    pct_test = np.percentile(X[split:], 98, axis=0)
    # LABEL_MODE=event: v11 injected-event labels (swap+/ramp/storm=1, swap-=0)
    if os.environ.get('LABEL_MODE', '') == 'event':
        ev = np.load(os.path.join(data_folder, DATA_VERSION,
                                  'event_labels.npy'))
        lab_train = ev[:split].astype(int)
        lab_test_own = ev[split:].astype(int)         # truth A
        lab_test_by_train = ev[split:].astype(int)    # truth B (same source)
        pct_train_rule = pct_train
    elif os.environ.get('LABEL_MODE', '') == 'overload':
        # QoS-overload labels (dump_replay.py): same source for train/test.
        ov = np.load(os.path.join(data_folder, DATA_VERSION,
                                  'labels_overload_class.npy'))
        lab_train = ov[:split].astype(int)
        lab_test_own = ov[split:].astype(int)         # truth A
        lab_test_by_train = ov[split:].astype(int)    # truth B (same source)
        pct_train_rule = pct_train
    else:
        lab_train = make_labels_p98(X[:split])
        lab_test_own = make_labels_p98(X[split:])          # truth A
        lab_test_by_train = labels_with_pct(X[split:],
                                            pct_train)     # truth B
        pct_train_rule = pct_train
    print(f'=== held-out eval: {DATA_VERSION} '
          f'train=[0,{split}) test=[{split},{T}) ===')
    print(f'train anomalies={int((lab_train > 0).sum())} '
          f'test anomalies own-thresh={int((lab_test_own > 0).sum())} '
          f'train-thresh={int((lab_test_by_train > 0).sum())}')

    # --- rule baselines (zero-cost references) ---
    f1A, tpA, fpA, fnA = f1_counts(lab_test_by_train > 0, lab_test_own > 0)
    print(f'[rule baseline] train-threshold rule on test split: '
          f'F1={f1A:.4f} (truth A: test-own labels) TP={tpA} FP={fpA} '
          f'FN={fnA}')
    agree = (lab_test_by_train > 0) == (lab_test_own > 0)
    print(f'[rule baseline] threshold agreement train-vs-test: '
          f'{agree.mean():.4f}')

    # --- collision pairs on the test split (strict truth B labels) ---
    # full-length labels under the TRAIN thresholds = lab_train ++
    # lab_test_by_train (same threshold source, single rule view).
    if os.environ.get('LABEL_MODE', '') == 'event':
        # v11: time-identical pairs (含当前步 [i-3, i+1]) with opposite labels
        ev_all = np.load(os.path.join(data_folder, DATA_VERSION,
                                      'event_labels.npy'))
        groups = defaultdict(list)
        for i in range(3, T):
            groups[X[i - 3:i + 1].tobytes()].append(i)
        pairs_test = []
        for rows in groups.values():
            if len(rows) < 2:
                continue
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    i, j = rows[a], rows[b]
                    for h in range(N):
                        li, lj = int(ev_all[i, h]), int(ev_all[j, h])
                        if li > 0 and lj == 0:
                            pairs_test.append((i, h, j))
                        elif li == 0 and lj > 0:
                            pairs_test.append((j, h, i))
        pairs_test = [(t, h, u) for (t, h, u) in pairs_test
                      if t >= split and u >= split]
        full_lab_rule = ev_all
    elif os.environ.get('LABEL_MODE', '') == 'overload':
        # QoS-overload labels: identical time windows with opposite overload
        # (the schedule-dependent "collision" pairs, natural not injected).
        ov_all = np.load(os.path.join(data_folder, DATA_VERSION,
                                      'labels_overload_class.npy'))
        full_lab_rule = ov_all
        pairs_test = [(t, h, u) for (t, h, u) in
                      collision_pairs_from_labels(full_lab_rule, T, N, X)
                      if t >= split and u >= split]
    else:
        full_lab_rule = labels_with_pct(X, pct_train)
        pairs_test = [(t, h, u) for (t, h, u) in
                      collision_pairs_from_labels(full_lab_rule, T, N, X)
                      if t >= split and u >= split]
    pos_test = set((t, h) for (t, h, _u) in pairs_test)
    # NOTE: never write `set((u, h) for (_t, _h, u) in pairs_test)` — the
    # unpacked _h shadows h inside the genexp (Python scoping pitfall).
    neg_test = set((p[2], p[1]) for p in pairs_test)
    print(f'[collision] test-split pairs={len(pairs_test)} '
          f'pos={len(pos_test)} neg={len(neg_test)}')

    # --- model evaluation (test rows only) ---
    ts_full = torch.tensor(S).double()
    print('\n' + '-' * 130)
    print(f'{"variant":5s} {"seed":>4s} {"testF1(A)":>9s} '
          f'{"testF1(B)":>9s} {"trainF1":>8s} {"swap+":>6s} '
          f'{"swap-":>6s} {"discrim":>7s} {"HR@100":>7s} '
          f'{"NDCG@100":>8s} {"clsN":>5s}'
          + (f' {"dHR@100":>8s} {"dNDCG":>7s} {"dN":>4s}' if
             int(os.environ.get('V2_DIAG_HEAD', '0')) else ''))
    print('-' * 130)
    for v, cls in VARIANTS.items():
        for seed in SEEDS:
            base = f'simulator_{cls().name}_s{seed}.ckpt'
            snap_ep = os.environ.get('SNAP_EPOCH', '85')
            path = os.path.join(CKPT_DIR, base.replace('.ckpt',
                                                       f'_snap{snap_ep}.ckpt'))
            if not os.path.exists(path):
                print(f'{v:5s} {seed:4d}  (no ckpt: {os.path.basename(path)})')
                continue
            model, _, _, _ = load_model(CKPT_DIR, os.path.basename(path),
                                        cls.__name__)
            tt = convert_to_windows(time_n, model)
            ts = convert_schedule_to_windows(ts_full, model)
            model.eval()
            preds = np.zeros((T, N), dtype=int)
            emb_list, cls_list = [], []
            diag_list, diag_gt = [], []
            # DIAG_PRED: 用诊断头聚合做检测（max over per-dim probs > 0.5）
            diag_pred_mode = int(os.environ.get('DIAG_PRED', '0'))
            with torch.no_grad():
                for i in range(T):
                    sa, sp = model(tt[i], ts[i])
                    if diag_pred_mode and hasattr(model, 'diag_head'):
                        d = model.diag_logits(tt[i])
                        dp = torch.sigmoid(d)
                        for j in range(N):
                            preds[i, j] = int(dp[j].max().item() > 0.5)
                    else:
                        for j, s in enumerate(sa):
                            preds[i, j] = torch.argmax(s).item()
                    if i >= split:
                        if hasattr(model, 'diag_head'):
                            d = model.diag_logits(tt[i])
                            diag_list.append(
                                torch.sigmoid(d).detach().cpu().numpy())
                        for j, p_emb in enumerate(sp):
                            if lab_test_by_train[i - split, j] > 0:
                                emb_list.append(
                                    p_emb.detach().cpu().numpy())
                                cls_list.append(
                                    int(lab_test_by_train[i - split, j]))
            # test-split metrics
            fA, _, _, _ = f1_counts(preds[split:] > 0,
                                    lab_test_own > 0)
            fB, tpB, fpB, fnB = f1_counts(preds[split:] > 0,
                                          lab_test_by_train > 0)
            # train-split fit reference (backfit indicator)
            fTr, _, _, _ = f1_counts(preds[:split] > 0, lab_train > 0)
            # collision discrimination on unseen rows
            pos_hit = np.mean([preds[t, h] == 1 for (t, h) in pos_test]) \
                if pos_test else float('nan')
            neg_hit = np.mean([preds[u, h] == 0 for (u, h) in neg_test]) \
                if neg_test else float('nan')
            disc = np.mean([(preds[t, h] == 1) and (preds[u, h] == 0)
                            for (t, h, u) in pairs_test]) \
                if pairs_test else float('nan')
            # classification metrics (paper @100% definition, train
            # threshold labels, frozen prototypes)
            protos = [p.detach().cpu().numpy() for p in model.prototype]
            hr, ndcg, cn = hr_ndcg_at100(emb_list, cls_list, protos)
            extra = ''
            if hasattr(model, 'diag_head') and diag_list:
                dprobs = np.concatenate(diag_list, axis=0)  # [202, 16, 7]
                # per-dim GT for test anomalies (train thresholds, truth B)
                pd_gt = perdim_with_pct(X[split:], pct_train)
                dhr, dndcg, dn = hr_ndcg_perdim(
                    dprobs.reshape(-1, 7),
                    pd_gt.reshape(-1, 7))
                extra = f' {dhr:8.4f} {dndcg:7.4f} {dn:4d}'
            print(f'{v:5s} {seed:4d} {fA:9.4f} {fB:9.4f} {fTr:8.4f} '
                  f'{pos_hit:6.3f} {neg_hit:6.3f} {disc:7.3f} '
                  f'{hr:7.4f} {ndcg:8.4f} {cn:5d}' + extra, flush=True)
    print('-' * 130)
    print('testF1(A)=truth test-own p98 labels; testF1(B)=truth train-'
          'thresh labels (strict); trainF1=backfit reference; '
          'swap+=P(pred=1|pos), swap-=P(pred=0|neg), '
          'discrim=P(pred(t)=1 and pred(u)=0); '
          'HR@100/NDCG@100=paper HitRate@100%/NDCG@100% on test anomalies '
          '(single-class labels -> top-1 prototype hit; clsN=#samples)')


if __name__ == '__main__':
    main()
