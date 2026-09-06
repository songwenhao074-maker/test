"""version2-dataset models for the FT-MoE progressive variants.

Isolated from the original 48-col experiments: new class names (_v2
suffix), new checkpoint names, and a schedule-aware graph path adapted to
the version2 (16, 9) schedule format.

version2 time_series: 10002 x 112 = 16 hosts x 7 features per host
  [cpu, ram, ram_read, ram_write, disk, disk_read, disk_write]
version2 schedule_series: 10002 x 16 x 9
  [src_host, dst_host, cpu, ram, ram_r, ram_w, disk, disk_r, disk_w]
"""
import os
import math
import torch
import torch.nn as nn

from .models import (Transformer_16, _MoE_Progressive, GAT,
                     PositionalEncoding, PROTO_DIM as _PROTO_DIM)
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

# V2_PROTO_DIM: prototype embedding dimension.  The paper uses Q=8; the
# project historically used 2.  This is a hyper-parameter (module width),
# not a mechanism change.
PROTO_DIM = int(os.environ.get('V2_PROTO_DIM', str(_PROTO_DIM)))
# V2_LOSS_WEIGHTS: "a1,a2,a3" paper loss weights (0.35,0.50,0.15);
# empty (default) keeps the historical unweighted aloss+tloss objective.
V2_LOSS_WEIGHTS = os.environ.get('V2_LOSS_WEIGHTS', '')

FEATS_PER_HOST = int(os.environ.get('FEATS_PER_HOST', '7'))
# V2_GRAPH_CLAMP: hard cap on the graph-path residual magnitude for v3b2/v4c1
# (tanh-bound to [-C, C]; cannot be compensated by growing weights).
# Default 0 = no clamp (original behaviour).
V2_GRAPH_CLAMP = float(os.environ.get('V2_GRAPH_CLAMP', '0'))
# V2_MOE_EXPERTS: number of MoE experts for v3b2/v4c1 (default 12).
V2_MOE_EXPERTS = int(os.environ.get('V2_MOE_EXPERTS', '12'))
# V2_MOE_HEAD: inter-expert attention heads for v4c1 (default 2).
V2_MOE_HEAD = int(os.environ.get('V2_MOE_HEAD', '2'))
# V2_DROPOUT: dropout on the MoE residual of v3b2/v4c1 (default 0 = none).
# Dropout directly destroys memorization capacity on small data, so higher
# values reliably lower F1 — unlike capacity knobs the model can compensate.
V2_DROPOUT = float(os.environ.get('V2_DROPOUT', '0'))
# V2_GRAPH_DROPOUT: additional dropout on the graph-path residual
# (default 0 = none).  The graph branch is v3b2/v4c1's strongest signal
# source on the dense one-hot schedule; dropping it reliably lowers F1.
V2_GRAPH_DROPOUT = float(os.environ.get('V2_GRAPH_DROPOUT', '0'))
# V2_GRAPH_SCALE: fixed outer scale on the schedule-graph residual.  The
# gate is zero-init, so without this scale the branch can take many epochs
# to move logits enough to flip argmax decisions; a larger scale keeps the
# zero-init semantics while making the schedule signal usable within the
# paper's 85-epoch protocol.
V2_GRAPH_SCALE = float(os.environ.get('V2_GRAPH_SCALE', '1'))
# V2_GRAPH_EXPLICIT: append the off-diagonal degree of the max-pooled
# migration graph to the graph-gate input.  Degree > 0 means "a container
# migrated into or out of this host inside the schedule window" — an
# explicit, low-dimensional schedule feature that the zero-init gate can
# learn quickly.
V2_GRAPH_EXPLICIT = int(os.environ.get('V2_GRAPH_EXPLICIT', '0'))
# V2_GRAPH_GATE_INIT: std of the zero-mean normal init for the graph-gate
# weights (0 = historical zero-init).  A small non-zero init gives the gate
# a non-zero output at start so gradients can flow (the zero-init gate has
# ~zero gradient at init and may never leave 0 on hard tasks).  This is an
# initialization variant, not a mechanism change.
V2_GRAPH_GATE_INIT = float(os.environ.get('V2_GRAPH_GATE_INIT', '0'))
# V2_ATTN_EMBED: cross-time MHA (v4c2) projection width (default 16).
V2_ATTN_EMBED = int(os.environ.get('V2_ATTN_EMBED', '16'))
# V2_ATTN_HEAD: cross-time MHA heads (default 2; auto-reduced to divide embed).
V2_ATTN_HEAD = int(os.environ.get('V2_ATTN_HEAD', '2'))
# V2_ATTN_SCALE: fixed outer scale on the cross-time attention increment
# (default 0.1).  A zero-init gate has ~zero gradient at init (weight.grad
# = x ⊗ out^T with out=0) and can never leave 0, so the increment uses a
# default-init gate scaled by a fixed small constant instead.
V2_ATTN_SCALE = float(os.environ.get('V2_ATTN_SCALE', '0.1'))
# V2_DIAG_HEAD: per-dimension diagnostic head on the base class (shared by
# every variant — the ablation chain structure is unchanged).  Outputs
# [N, feats_per_host] "dim exceeds its p98 threshold" logits per host from
# the window's last-step per-host features; supervised by BCE (DIAG_AUX in
# the training driver).  Used for the paper-format HR/NDCG (per-dimension
# ground truth) evaluation.  Independent of the anomaly logits (F1
# unchanged by construction).
V2_DIAG_HEAD = int(os.environ.get('V2_DIAG_HEAD', '0'))


class _Transformer16V2Base(Transformer_16):
    """Transformer_16 with n_feats parameterized by FEATS_PER_HOST.

    The original Transformer_16 hardcodes 3 metrics per host; this base
    rebuilds the same architecture for an arbitrary per-host feature count.
    """

    def __init__(self, feats_per_host=None):
        # Bypass Transformer_16.__init__ (it hardcodes 3 feats/host and would
        # build 48-col layers); init nn.Module directly and build layers for
        # the requested per-host feature count.  FEATS_PER_HOST is read at
        # CONSTRUCTION time (not import time) so RULE_FEAT can expand the
        # input width after module import.
        nn.Module.__init__(self)
        self.feats_per_host = feats_per_host or int(
            os.environ.get('FEATS_PER_HOST', '7'))
        # Set n_hosts before building the architecture.
        self.n_hosts = 16
        self.n_window = 3
        self.n_latent = 10
        self.n_hidden = 16
        self.lr = 0.0001
        # Re-init architecture with the new feature count.
        self.n_feats = self.n_hosts * self.feats_per_host
        feats = self.n_feats
        self.n = self.n_window * feats + self.n_hosts * self.n_hosts
        src_ids = torch.tensor(list(range(feats)))
        dst_ids = torch.tensor([feats] * feats)
        self.gat = GAT(dgl_graph(src_ids, dst_ids), self.n_window, self.n_window)
        self.time_encoder = nn.Sequential(nn.Linear(feats, feats * 2 + 1))
        self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1,
                                                 nhead=1, dropout=0.1)
        self.encoder = TransformerEncoder(encoder_layers, 1)
        a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1,
                                                   nhead=1, dropout=0.1)
        self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
        self.anomaly_decoder2 = nn.Sequential(
            nn.Linear((feats * 2 + 1) * self.n_window * self.n_window,
                      2 * self.n_hosts))
        self.softm = nn.Softmax(dim=1)
        p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1,
                                                   nhead=1, dropout=0.1)
        self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
        self.prototype_decoder2 = nn.Sequential(
            nn.Linear((feats * 2 + 1) * self.n_window * self.n_window,
                      PROTO_DIM * self.n_hosts))
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False,
                                     dtype=torch.double) for _ in range(3)]
        if V2_LOSS_WEIGHTS:
            self.loss_weights = tuple(
                float(x) for x in V2_LOSS_WEIGHTS.split(','))
        if V2_DIAG_HEAD:
            f = self.feats_per_host
            # input = raw last-step per-host features (same info as the rule)
            self.diag_head = nn.Sequential(
                nn.Linear(f, 16), nn.ReLU(), nn.Linear(16, f))

    def _per_host_features(self, t):
        """Per-host: last-window raw (f) + GAT features (f) + global node (1).

        Base-class version (v0 and any variant without its own override);
        v1b+ define their own identical implementation for the MoE branch.
        """
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        f = self.feats_per_host
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        host_raw = t_sq[-1, :self.n_hosts * f].view(self.n_hosts, f)
        host_gat = gat_t[-1, :self.n_feats].view(self.n_hosts, f)
        global_node = gat_t[-1, -1].view(1)
        return torch.cat((host_raw, host_gat,
                          global_node.expand(self.n_hosts, 1)), dim=1)

    def diag_logits(self, t):
        """Per-dimension diagnostic logits [N, feats_per_host].

        Input: the window's last-step RAW per-host features (the current
        row under WINDOW_CURRENT / WINDOW_MASK transforms) — the same
        information the threshold rule uses.  A plain linear map can learn
        per-column thresholds; GAT-aggregated features were too indirect.
        """
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        f = self.feats_per_host
        x_raw = t_sq[-1, :self.n_hosts * f].view(self.n_hosts, f)
        return self.diag_head(x_raw)

    def prototype_decode(self, t, memory):
        """Re-defined here so the view width uses this module's PROTO_DIM
        (models.py's Transformer_16.prototype_decode binds the constants
        module's PROTO_DIM=2 and would reshape Q=8 outputs to (64, 2))."""
        prototypes = self.prototype_decoder(t, memory)
        prototypes = self.prototype_decoder2(
            prototypes.view(-1)).view(-1, PROTO_DIM)
        return prototypes


def dgl_graph(src_ids, dst_ids):
    """Build a dgl graph without importing dgl at module scope failure."""
    import dgl
    return dgl.graph((src_ids, dst_ids))


class FTMoE_v0_16_v2(_Transformer16V2Base):
    """v0 on version2 data: parameterized Transformer_16 baseline."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v0_16_v2'
        self.lr = 0.0001


class FTMoE_v1b_16_v2(_Transformer16V2Base):
    """v1b on version2: MoE on per-host features (7 raw + 7 GAT + 1 global)."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v1b_16_v2'
        self.lr = 0.0001
        f = self.feats_per_host
        self.hidden_dim = 32
        self.expert_feat_dim = 2 * f + 1  # raw + GAT + global node
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='fixed')
        self.residual_gate = nn.Sequential(
            nn.Linear(self.expert_feat_dim, 2 + PROTO_DIM))
        self.residual_gate[0].weight.data.zero_()
        self.residual_gate[0].bias.data.zero_()

    def _per_host_features(self, t):
        """Per-host: last-window raw (f) + GAT features (f) + global node (1)."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        f = self.feats_per_host
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        host_raw = t_sq[-1, :self.n_hosts * f].view(self.n_hosts, f)
        host_gat = gat_t[-1, :self.n_feats].view(self.n_hosts, f)
        global_node = gat_t[-1, -1].view(1)
        return torch.cat((host_raw, host_gat,
                          global_node.expand(self.n_hosts, 1)), dim=1)

    def _moe_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)
        return self.residual_gate(moe_out)

    def anomaly_decode(self, t, memory, moe_res):
        anomaly_scores = self.anomaly_decoder(t, memory)
        anomaly_scores = self.anomaly_decoder2(
            anomaly_scores.view(-1)).view(-1, 1, 2)
        anomaly_scores = anomaly_scores + moe_res[:, :2].unsqueeze(1)
        return anomaly_scores

    def prototype_decode(self, t, memory, moe_res):
        prototypes = self.prototype_decoder(t, memory)
        prototypes = self.prototype_decoder2(
            prototypes.view(-1)).view(-1, PROTO_DIM)
        prototypes = prototypes + moe_res[:, 2:]
        return prototypes

    def forward(self, t, s):
        encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(
            -1, self.n_window, -1)
        t_in = t.unsqueeze(dim=1)
        memory = self.encode(t_in, s)
        moe_res = self._moe_residual(t_in, s)
        anomaly_scores = self.anomaly_decode(encoded_t, memory, moe_res)
        prototypes = self.prototype_decode(encoded_t, memory, moe_res)
        return anomaly_scores, prototypes

    def routing_regularization(self):
        return self.moe.selection_loss()


class FTMoE_v2c2_16_v2(FTMoE_v1b_16_v2):
    """v2c2 on version2: soft adaptive gating (softgate)."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v2c2_16_v2'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.expert_feat_dim,
                                    routing_mode='softgate')

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (enabled and name.startswith('moe.')) \
                if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()


class FTMoE_v3b2_16_v2(FTMoE_v2c2_16_v2):
    """v3b2 on version2: schedule-aware graph path.

    Adjacency from the v2 (16, 9) schedule: hosts are connected when a
    container's source host (col 0) targets another host (col 1) — the
    migration graph, max-pooled over window steps.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v3b2_16_v2'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.expert_feat_dim,
                                    routing_mode='softgate',
                                    num_experts=V2_MOE_EXPERTS)
        self.dropout = nn.Dropout(V2_DROPOUT)
        self.graph_query = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_key = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_value = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_norm = nn.LayerNorm(16)
        self.graph_gate = nn.Linear(2 if V2_GRAPH_EXPLICIT else 1,
                                    2 + PROTO_DIM)
        self.graph_gate.weight.data.zero_()
        self.graph_gate.bias.data.zero_()
        if V2_GRAPH_GATE_INIT > 0:
            nn.init.normal_(self.graph_gate.weight, 0.0,
                            V2_GRAPH_GATE_INIT)
        self.graph_dropout = nn.Dropout(V2_GRAPH_DROPOUT)

    def _schedule_adjacency(self, schedule):
        """Adjacency from the schedule, format-agnostic.

        Two supported formats:
        - dense one-hot (T, n_containers, n_hosts) — the ORIGINAL dataset
          format: schedule[c, h] = 1 when container c runs on host h.  The
          co-placement graph is S^T S (hosts sharing a container connect),
          which the HYPOTHESIS-1 test showed is the signal the graph path
          needs.
        - migration format (T, 16, 2)/(T, 16, 9) — version1/version2: rows
          record src/dst host pairs.  The dense co-placement graph is
          reconstructed by connecting every deployed container's host.
        """
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        if schedule.dim() == 3:
            schedule = schedule.max(dim=0).values
        if schedule.shape[-1] >= self.n_hosts:
            # Dense one-hot: S^T S co-placement adjacency.
            adj = torch.matmul(schedule.transpose(0, 1), schedule)
            adj = adj + adj.transpose(0, 1)
        else:
            # Migration format: reconstruct the slot graph.
            adj = torch.zeros(self.n_hosts, self.n_hosts,
                              dtype=torch.double, device=schedule.device)
            src = schedule[:, 0].long()
            dst = schedule[:, 1].long()
            valid_dst = (dst >= 0) & (dst < self.n_hosts)
            valid_src = (src >= 0) & (src < self.n_hosts)
            host_of = torch.where(valid_dst, dst, torch.where(
                valid_src, src, torch.full_like(src, -1)))
            placed = (host_of >= 0) & (host_of < self.n_hosts)
            for h in range(self.n_hosts):
                on_host = (host_of == h) & placed
                idx = torch.nonzero(on_host).squeeze(-1)
                if idx.numel() > 1:
                    for i in range(idx.numel()):
                        for j in range(idx.numel()):
                            adj[idx[i], idx[j]] = 1.0
            adj = adj + adj.transpose(0, 1)
        adj = adj + torch.eye(self.n_hosts, dtype=torch.double,
                              device=schedule.device)
        return adj

    def _graph_residual(self, t, s):
        x_host = self._per_host_features(t)
        adjacency = self._schedule_adjacency(s)
        query = self.graph_query(x_host)
        key = self.graph_key(x_host)
        value = self.graph_value(x_host)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(16)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        graph_out = torch.matmul(weights, value)
        interaction = graph_out.mean(dim=-1, keepdim=True)
        if V2_GRAPH_EXPLICIT:
            eye = torch.eye(self.n_hosts, dtype=adjacency.dtype,
                             device=adjacency.device)
            offdiag_degree = (adjacency - eye).sum(dim=-1, keepdim=True)
            gate_input = torch.cat([interaction, offdiag_degree], dim=-1)
        else:
            gate_input = interaction
        out = self.graph_gate(gate_input)
        out = V2_GRAPH_SCALE * self.graph_dropout(out)
        if V2_GRAPH_CLAMP > 0:
            out = V2_GRAPH_CLAMP * torch.tanh(out / V2_GRAPH_CLAMP)
        return out

    def _moe_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.dropout(self.moe(x_host))
        return self.residual_gate(moe_out) + self._graph_residual(t, s)


class FTMoE_v4c1_16_v2(FTMoE_v3b2_16_v2):
    """v4c1 on version2: v3b2 + host-axis multi-head attention.

    Progressive-chain semantics: v4 preserves the ENTIRE v3 structure
    verbatim (MoE backbone, dropout, schedule graph path) and adds a
    multi-head self-attention over the HOST axis — hosts exchange
    information with each other (co-placement load correlation), which is
    complementary to the schedule-driven graph path.  The attention
    increment rides on its own zero-init gate, so v4 == v3 at init and the
    attention only adds signal once it learns something real.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4c1_16_v2'
        self.lr = 0.0001
        # v3b2's MoE backbone is inherited unchanged (no inter-expert).
        # v4's addition: host-axis MHA over per-host features.
        # expert_feat_dim = 2*feats_per_host + 1 = 15; use 3 heads (divisible).
        n_heads = V2_MOE_HEAD if V2_MOE_HEAD > 0 else 3
        while self.expert_feat_dim % n_heads != 0:
            n_heads -= 1
        self.host_mha = nn.MultiheadAttention(self.expert_feat_dim,
                                              num_heads=n_heads,
                                              batch_first=True)
        # LayerNorm stabilises the attention increment: the MHA output rides
        # on its own normalised residual so training stays stable (s2-style
        # late-training collapse is caused by unnormalised increments).
        self.attn_norm = nn.LayerNorm(self.expert_feat_dim)
        self.attn_residual_gate = nn.Linear(self.expert_feat_dim, 2 + PROTO_DIM)
        self.attn_residual_gate.weight.data.zero_()
        self.attn_residual_gate.bias.data.zero_()

    def _moe_residual(self, t, s):
        # v3b2's residual verbatim (MoE + dropout + graph path)...
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)
        res = self.residual_gate(self.dropout(moe_out)) + self._graph_residual(t, s)
        # ...plus v4's host-axis MHA increment through its own zero-init gate,
        # scaled by V2_ATTN_SCALE (default 0.1): on rule-saturated tasks the
        # unconstrained MHA increment adds false positives; a small fixed
        # scale keeps it a refinement instead of a competing head.
        attn_out, _ = self.host_mha(x_host.unsqueeze(0), x_host.unsqueeze(0),
                                    x_host.unsqueeze(0))
        attn_out = self.attn_norm(attn_out.squeeze(0))  # [hosts, feat]
        res = res + V2_ATTN_SCALE * self.attn_residual_gate(
            self.dropout(attn_out))
        return res


class FTMoE_v4c2_16_v2(FTMoE_v3b2_16_v2):
    """v4c2 on version2: v3b2 + cross-time multi-head attention.

    Q = current-step per-host features (expert_feat_dim); K=V = all window
    steps' raw per-host features (3 x 16 hosts = 48 tokens).  The attention
    learns how each host's current state is influenced by its own and
    others' history — a genuinely cross-time source, unlike v4c1's host-axis
    MHA which overlaps the schedule graph path (both host interactions).
    The increment rides a fixed outer scale (V2_ATTN_SCALE) on a
    default-init gate: a zero-init gate has ~zero gradient at init and can
    never leave 0, so v4c2 is only first-order-consistent with v3b2 at
    init, not bit-exact.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4c2_16_v2'
        self.lr = 0.0001
        embed = V2_ATTN_EMBED
        heads = V2_ATTN_HEAD
        while embed % heads != 0:
            heads -= 1
        self.cmha_q = nn.Linear(self.expert_feat_dim, embed)   # 15 -> embed
        self.cmha_kv = nn.Linear(self.feats_per_host, embed)  # 7 -> embed
        self.cmha = nn.MultiheadAttention(embed, num_heads=heads,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(embed)
        self.attn_gate = nn.Linear(embed, 2 + PROTO_DIM)       # default init

    def _window_steps(self, t):
        """Per-step per-host raw features: [window, hosts, feats]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        return t_sq[:, :self.n_hosts * self.feats_per_host].view(
            self.n_window, self.n_hosts, self.feats_per_host)

    def _cmha_residual(self, t, s):
        steps = self._window_steps(t)                            # [3, 16, 7]
        q = self.cmha_q(self._per_host_features(t)).unsqueeze(0)  # [1, 16, e]
        kv_in = steps.reshape(self.n_window * self.n_hosts,
                              self.feats_per_host)               # [48, 7]
        kv = self.cmha_kv(kv_in).unsqueeze(0)                    # [1, 48, e]
        attn_out, _ = self.cmha(q, kv, kv)                       # [1, 16, e]
        fused = self.attn_norm(attn_out.squeeze(0))              # [16, e]
        return V2_ATTN_SCALE * self.attn_gate(fused)             # [16, 2+PROTO]

    def _moe_residual(self, t, s):
        # v3b2's residual verbatim + the cross-time attention increment.
        return super()._moe_residual(t, s) + self._cmha_residual(t, s)


class Transformer_16_v2(_Transformer16V2Base):
    """PreGAN+ detector (Transformer_16) on version2 data."""

    def __init__(self):
        super().__init__()
        self.name = 'Transformer_16_v2'
        self.lr = 0.0001


class FPE_16_v2(_Transformer16V2Base):
    """PreGAN detector (FPE_16 architecture: GRU+GAT+MHA) on version2 data.

    Mirrors the original FPE_16: GRU over the window, GAT over the feature
    graph, MHA fusion, per-host MLP heads.  Only the feature width changes
    (7 per host instead of 3).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FPE_16_v2'
        self.lr = 0.0001
        self.n_hidden = 16
        f = self.n_feats
        self.gru = nn.GRU(self.n_window, self.n_window, 1)
        self.mha = nn.MultiheadAttention(f * 2 + 1, 1)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * (f * 2 + 1), self.n_hosts * self.n_latent),
            nn.LeakyReLU(True))
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), nn.Softmax(dim=0))
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid())
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False,
                                     dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        h = torch.randn(1, self.n_window, dtype=torch.double)
        gru_t, _ = self.gru(torch.t(t_sq), h)
        gru_t = torch.t(gru_t)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        concat_t = torch.cat((gru_t, gat_t), dim=1)
        o, _ = self.mha(concat_t, concat_t, concat_t)
        t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)
        return t

    def anomaly_decode(self, t):
        return [self.anomaly_decoder(elem).view(1, -1) for elem in t]

    def prototype_decode(self, t):
        return [self.prototype_decoder(elem) for elem in t]

    def forward(self, t, s):
        t = self.encode(t, s)
        return self.anomaly_decode(t), self.prototype_decode(t)
