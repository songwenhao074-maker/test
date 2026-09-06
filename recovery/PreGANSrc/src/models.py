import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

## FPE
class FPE_16(nn.Module):
	def __init__(self):
		super(FPE_16, self).__init__()
		self.name = 'FPE_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_16(nn.Module):
	def __init__(self):
		super(Gen_16, self).__init__()
		self.name = 'Gen_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_16(nn.Module):
	def __init__(self):
		super(Disc_16, self).__init__()
		self.name = 'Disc_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


## FPE
class FPE_50(nn.Module):
	def __init__(self):
		super(FPE_50, self).__init__()
		self.name = 'FPE_50'
		self.lr = 0.0001
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 50
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
	def __init__(self):
		super(Attention_50, self).__init__()
		self.name = 'Attention_50'
		self.lr = 0.0008
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		# self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats), 
		# 		nn.Sigmoid())	for i in range(1)]
		# self.encoder_atts = nn.ModuleList(self.atts)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		# for at in self.encoder_atts:
		# 	inp = torch.cat((t.view(-1), s.view(-1)))
		# 	ats = at(inp).reshape(self.n_feats, self.n_feats)
		# 	t = torch.matmul(t, ats)	
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
	def __init__(self):
		super(Gen_50, self).__init__()
		self.name = 'Gen_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
	def __init__(self):
		super(Disc_50, self).__init__()
		self.name = 'Disc_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


############## PreGANPlus Models ##############

# Transformer Model
class Transformer_16(nn.Module):
	def __init__(self):
		super(Transformer_16, self).__init__()
		self.name = 'Transformer_16'
		self.lr = 0.0001
		self.n_hosts = 16
		feats = 3 * self.n_hosts
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.time_encoder = nn.Sequential(
			nn.Linear(feats, feats * 2 + 1), 
		)
		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.encoder = TransformerEncoder(encoder_layers, 1)
		a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
		self.anomaly_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts), 
		)
		self.softm = nn.Softmax(dim=1)
		p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
		self.prototype_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, PROTO_DIM * self.n_hosts), 
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		t = torch.squeeze(t, 1)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		o = torch.cat((t, gat_t), dim=1)
		t = o * math.sqrt(self.n_feats)
		t = self.pos_encoder(t) # window size, batch size (1), feats (3 metrics * 16 hosts)
		memory = self.encoder(t)	
		return memory

	def anomaly_decode(self, t, memory):
		anomaly_scores = self.anomaly_decoder(t, memory)
		anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
		return anomaly_scores

	def prototype_decode(self, t, memory):
		prototypes = self.prototype_decoder(t, memory)
		prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
		return prototypes

	def forward(self, t, s):
		encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
		t = t.unsqueeze(dim=1)
		memory = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(encoded_t, memory)
		prototypes = self.prototype_decode(encoded_t, memory)
		return anomaly_scores, prototypes


## FT-MoE progressive variants ------------------------------------------------
## Incremental build on top of Transformer_16 (PreGAN+ baseline):
##   v0 = Transformer_16 exactly (anchor / sanity check)
##   v1 = + MoE branch (fixed 12-expert routing on host embeddings)
##   v2 = + EAGate adaptive Top-any selection with online add/remove
##   v3 = + schedule-aware graph path (second branch)
##   v4 = + cross multi-head attention fusion (full FT-MoE dual-path)

class _MoE_Progressive(nn.Module):
    """Shared MoE building block for the progressive variants.

    routing_mode:
      'fixed'  -> dense softmax routing over all experts (v1 baseline MoE)
      'eagate' -> adaptive Top-any selection (v2+, paper EAGate)
    Online add/remove (paper Sec. Online Tuning) is enabled only in
    'eagate' mode, mirroring the recovery-level adaptation cadence.
    """

    def __init__(self, hidden_dim, num_experts=12, max_active=8,
                 min_experts=4, max_experts=16, routing_mode='fixed',
                 temperature=0.25, inter_expert=None, expert_width=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.routing_mode = routing_mode
        self.max_active = max_active
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.temperature = temperature
        self.expert_width = expert_width if expert_width > 0 else None
        self.experts = nn.ModuleList([self._make_expert() for _ in range(num_experts)])
        self.expert_keys = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.02)
        self.thresholds = nn.Parameter(torch.zeros(num_experts))
        # Inter-expert attention (v4c1): self-attention over the expert axis
        # of the expert outputs, before routing.  The zero-init projection
        # keeps the branch neutral at initialization.
        self.inter_expert = inter_expert
        if inter_expert is not None:
            # Project expert outputs to a head-divisible space, attend along
            # the expert axis, then project back (zero-init -> neutral at start).
            self.attn_embed = nn.Linear(hidden_dim, hidden_dim * 2)
            self.expert_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=inter_expert, batch_first=True)
            self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.attn_proj.weight.data.zero_()
            self.attn_proj.bias.data.zero_()
            self.attn_norm = nn.LayerNorm(hidden_dim)
            # Learnable residual gate on the attention branch (init 0):
            # v4 == v3 at init; the attention only changes the output once it
            # learns real signal.
            self.attn_gate = nn.Parameter(torch.tensor(0.0))
        # Top-k soft routing parameters (v2c): logits = score_scale * cosine +
        # gumbel noise; temperature anneals from ~1.0 down to ~0.1 over epochs.
        self.score_scale = nn.Parameter(torch.tensor(2.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self._training_steps = 0
        self.register_buffer('activation_counts', torch.zeros(num_experts))
        self.register_buffer('routing_samples', torch.zeros(1))
        self.register_buffer('unrouted_samples', torch.zeros(1))
        self._selection_loss = torch.tensor(0.0)

    def _make_expert(self):
        width = getattr(self, 'expert_width', None)  # None = full width
        if width is None:
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
        return nn.Sequential(
            nn.Linear(self.hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, self.hidden_dim),
        )

    @property
    def num_experts(self):
        return len(self.experts)

    def reset_routing_stats(self):
        with torch.no_grad():
            self.activation_counts.zero_()
            self.routing_samples.zero_()
            self.unrouted_samples.zero_()

    def forward(self, x):
        # x: [hosts, hidden_dim]
        scores = F.cosine_similarity(x.unsqueeze(1), self.expert_keys.unsqueeze(0), dim=-1)
        probabilities = torch.sigmoid(scores)
        thresholds = torch.sigmoid(self.thresholds).unsqueeze(0)
        if self.routing_mode == 'softgate':
            # Soft adaptive gating (v2c2): sigmoid(scale*(sim - threshold))
            # per expert — the paper's activation-probability idea made fully
            # differentiable.  Low-similarity experts are naturally suppressed
            # (soft sparsity) but keep a gradient path; no hard discrete pick.
            gate = torch.sigmoid(self.score_scale * (scores - thresholds))
            routing_mask = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            hard_mask = routing_mask.detach()  # stats / normalization only
            eligible = probabilities >= thresholds
        elif self.routing_mode == 'topk-soft':
            self._training_steps += 1
            if self.training:
                tau = max(0.1, 1.0 * (0.7 ** (self._training_steps / 1000.0)))
            else:
                tau = 0.1
            noise = torch.rand_like(scores)
            noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
            logits = self.score_scale * scores + noise
            k = min(self.max_active, self.num_experts)
            hard_mask = torch.zeros_like(logits)
            topk_idx = torch.topk(logits, k, dim=-1).indices
            hard_mask.scatter_(1, topk_idx, 1.0)
            soft_mask = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            routing_mask = hard_mask + soft_mask - soft_mask.detach()
            eligible = probabilities >= thresholds  # stats only
        elif self.routing_mode == 'fixed':
            # Plain softmax routing: every expert is active, weighted by similarity.
            hard_mask = torch.softmax(scores, dim=-1)
            routing_mask = hard_mask
            eligible = probabilities >= thresholds
        else:
            eligible = probabilities >= thresholds
            hard_mask = torch.zeros_like(scores)
            for row in range(scores.shape[0]):
                selected = torch.nonzero(eligible[row], as_tuple=False).flatten()
                if selected.numel() == 0:
                    selected = torch.topk(scores[row], k=1).indices
                elif selected.numel() > self.max_active:
                    selected = torch.topk(scores[row], k=self.max_active).indices
                hard_mask[row, selected] = 1.0
            soft_mask = torch.sigmoid((probabilities - thresholds) / self.temperature)
            routing_mask = hard_mask + soft_mask - soft_mask.detach()
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        self._attn_out = None
        if self.inter_expert is not None:
            # Inter-expert attention: experts exchange information along the
            # expert axis; zero-init projection keeps the branch neutral at
            # start, and LayerNorm keeps the residual well-conditioned.
            attended, _ = self.expert_attn(self.attn_embed(expert_outputs),
                                           self.attn_embed(expert_outputs),
                                           self.attn_embed(expert_outputs))
            attended = self.attn_proj(attended)             # zero-init
            if hasattr(self, 'attn_gate'):
                # Learnable residual gate (init 0): the attention branch only
                # contributes once it learns real signal — v4 starts as v3.
                attended = attended * self.attn_gate
            self._attn_out = attended                       # exposed for v4
            expert_outputs = self.attn_norm(expert_outputs + attended)
        output = (expert_outputs * scores.unsqueeze(-1) * routing_mask.unsqueeze(-1)).sum(dim=1)
        output = output / hard_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        self._selection_loss = scores.square().mean()
        if self.routing_mode == 'topk-soft':
            # Load-balancing: even fraction of hosts routed to each expert.
            frac = (hard_mask > 0).float().mean(dim=0)
            target = 1.0 / self.num_experts
            self._selection_loss = self._selection_loss + 1.0 * (frac - target).square().mean()
        with torch.no_grad():
            self.activation_counts.add_(hard_mask.detach().sum(dim=0))
            self.routing_samples.add_(scores.shape[0])
            self.unrouted_samples.add_((eligible.sum(dim=1) == 0).sum())
        return output

    def selection_loss(self):
        return self._selection_loss

    def adapt_experts(self):
        """Remove unused experts, add capacity for unmatched inputs (paper online tuning)."""
        if self.routing_samples.item() == 0:
            return False
        changed = False
        removable = [
            index for index, count in enumerate(self.activation_counts.tolist())
            if count == 0 and self.num_experts > self.min_experts
        ]
        removable = removable[:max(0, self.num_experts - self.min_experts)]
        if removable:
            keep = [index for index in range(self.num_experts) if index not in removable]
            for index in sorted(removable, reverse=True):
                del self.experts[index]
            self._replace_router_parameters(self.expert_keys[keep], self.thresholds[keep])
            changed = True
        unrouted_ratio = self.unrouted_samples.item() / self.routing_samples.item()
        if unrouted_ratio >= 0.20 and self.num_experts < self.max_experts:
            parameter = next(self.experts[0].parameters())
            expert = self._make_expert().to(device=parameter.device, dtype=parameter.dtype)
            self.experts.append(expert)
            new_key = torch.randn(1, self.hidden_dim, device=parameter.device, dtype=parameter.dtype) * 0.02
            new_threshold = torch.zeros(1, device=parameter.device, dtype=parameter.dtype)
            self._replace_router_parameters(
                torch.cat((self.expert_keys, new_key), dim=0),
                torch.cat((self.thresholds, new_threshold), dim=0),
            )
            changed = True
        self.reset_routing_stats()
        return changed


class FTMoE_v0_16(Transformer_16):
    """v0: exact PreGAN+ baseline (Transformer_16), renamed checkpoint target."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v0_16'
        self.lr = 0.0001


class FTMoE_v1b_16(Transformer_16):
    """v1b: MoE on per-host features, zero-init residual addition.

    Fixes v1's two flaws: (1) experts now route on each host's own
    feature vector (host features + GAT-transformed features + global node,
    i.e. the paper's X_m), so different hosts can pick different experts;
    (2) the MoE output is added to the PreGAN+ decoder output as a
    zero-initialized residual — at initialization v1b is exactly v0, so the
    baseline path is preserved and the MoE only adjusts it.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v1b_16'
        self.lr = 0.0001
        feats = self.n_feats
        self.hidden_dim = 32
        self.expert_feat_dim = 7  # per-host: 3 raw metrics + 3 GAT features + 1 global node
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='fixed')
        # Zero-init residual on the final logits: at start, v1b == v0.
        self.residual_gate = nn.Sequential(
            nn.Linear(self.expert_feat_dim, 2 + PROTO_DIM),
        )
        self.residual_gate[0].weight.data.zero_()
        self.residual_gate[0].bias.data.zero_()

    def encode(self, t, s):
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        o = torch.cat((t_sq, gat_t), dim=1)
        o = o * math.sqrt(self.n_feats)
        o = self.pos_encoder(o)
        memory = self.encoder(o)
        return memory

    def _per_host_features(self, t):
        """Extract the per-host feature matrix used by the MoE / graph paths.

        For each host: last-window raw metrics (3) + GAT-transformed metrics
        (3) + the global context node (1) -> [hosts, 7].
        """
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        host_raw = t_sq[-1, :self.n_hosts * 3].view(self.n_hosts, 3)
        host_gat = gat_t[-1, :self.n_feats].view(self.n_hosts, 3)
        global_node = gat_t[-1, -1].view(1)
        return torch.cat((host_raw, host_gat, global_node.expand(self.n_hosts, 1)), dim=1)

    def _moe_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)              # [hosts, 7]
        return self.residual_gate(moe_out)      # [hosts, 2+PROTO_DIM]; zero-init

    def anomaly_decode(self, t, memory, moe_res):
        anomaly_scores = self.anomaly_decoder(t, memory)
        anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
        anomaly_scores = anomaly_scores + moe_res[:, :2].unsqueeze(1)
        return anomaly_scores

    def prototype_decode(self, t, memory, moe_res):
        prototypes = self.prototype_decoder(t, memory)
        prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
        prototypes = prototypes + moe_res[:, 2:]
        return prototypes

    def forward(self, t, s):
        encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
        t_in = t.unsqueeze(dim=1)
        memory = self.encode(t_in, s)
        moe_res = self._moe_residual(t_in, s)
        anomaly_scores = self.anomaly_decode(encoded_t, memory, moe_res)
        prototypes = self.prototype_decode(encoded_t, memory, moe_res)
        return anomaly_scores, prototypes

    def routing_regularization(self):
        return self.moe.selection_loss()


class FTMoE_v2b_16(FTMoE_v1b_16):
    """v2b: v1b + EAGate adaptive Top-any expert selection.

    Swaps v1b's fixed softmax routing for the paper's adaptive selection
    (Eq. 2-5): per-expert activation thresholds, Top-any pick rule, and
    online add/remove capacity.  The zero-init residual structure is
    inherited, so at initialization v2b is exactly v0.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v2b_16'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='eagate')

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = enabled and name.startswith('moe.') if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()


class FTMoE_v2c_16(FTMoE_v1b_16):
    """v2c: v1b + continuous Top-k soft routing with annealing + load balance.

    Fixes v2b's EAGate failure mode (threshold routing collapsed to
    all-on/all-off with no fault correlation): selection is now a
    differentiable Top-k with annealed gumbel-softmax, so every expert
    keeps gradients, routing stays diverse, and no hard threshold is needed.
    The zero-init residual structure is inherited (v2c == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v2c_16'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='topk-soft')

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = enabled and name.startswith('moe.') if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()


class FTMoE_v2c2_16(FTMoE_v1b_16):
    """v2c2: v1b + soft adaptive gating (sigmoid on score-threshold, normalized).

    Fully differentiable version of the paper's EAGate activation
    probability: experts below their threshold are softly suppressed rather
    than hard-dropped, so all experts keep gradients and the router stays
    smooth.  Zero-init residual inherited (v2c2 == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v2c2_16'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='softgate')

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = enabled and name.startswith('moe.') if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()


class FTMoE_v3b_16(FTMoE_v2c2_16):
    """v3b: v2c2 + schedule-aware graph path (paper Eq. 1).

    Second branch: hosts connected in the scheduling matrix S (S^T S
    adjacency + self-loops) attend to each other through masked graph
    attention; the aggregated per-host graph features join the MoE branch
    as an extra residual on the logits.  Zero-init residuals are inherited,
    so v3b == v0 at initialization.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v3b_16'
        self.lr = 0.0001
        self.graph_query = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_key = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_value = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_norm = nn.LayerNorm(16)
        self.graph_gate = nn.Linear(16, 2 + PROTO_DIM)
        self.graph_gate.weight.data.zero_()
        self.graph_gate.bias.data.zero_()

    def _schedule_adjacency(self, schedule):
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        schedule = schedule[:self.n_hosts, :self.n_hosts]
        adjacency = torch.matmul(schedule.transpose(0, 1), schedule)
        adjacency = adjacency + adjacency.transpose(0, 1)
        adjacency = adjacency + torch.eye(self.n_hosts, device=schedule.device, dtype=schedule.dtype)
        return adjacency

    def _graph_residual(self, t, s):
        x_host = self._per_host_features(t)
        adjacency = self._schedule_adjacency(s)
        query = self.graph_query(x_host)
        key = self.graph_key(x_host)
        value = self.graph_value(x_host)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(16)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        graph_out = self.graph_norm(torch.matmul(weights, value))
        return self.graph_gate(graph_out)       # [hosts, 2+PROTO_DIM]; zero-init

    def _moe_residual(self, t, s):
        return super()._moe_residual(t, s) + self._graph_residual(t, s)


class FTMoE_v3b2_16(FTMoE_v2c2_16):
    """v3b2: v2c2 + schedule-aware graph path, sparse-graph aware.

    Fixes v3b's failure mode: S^T S adjacency has only ~4 non-zero edges on
    average, so a 16-dim per-host graph feature was mostly zero-padding
    noise.  v3b2 (1) builds the adjacency from the whole window of
    schedules (max-pooled over the 3 intervals, capturing migration paths),
    and (2) outputs a scalar per-host residual — the aggregate "interaction
    strength" with scheduling neighbours — instead of a high-dim vector.
    Zero-init residuals inherited (v3b2 == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v3b2_16'
        self.lr = 0.0001
        self.graph_query = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_key = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_value = nn.Linear(self.expert_feat_dim, 16, bias=False)
        self.graph_norm = nn.LayerNorm(16)
        self.graph_gate = nn.Linear(1, 2 + PROTO_DIM)
        self.graph_gate.weight.data.zero_()
        self.graph_gate.bias.data.zero_()

    def _schedule_adjacency(self, schedule):
        # Window-aggregated schedule: max-pool the 3 intervals' S matrices
        # (each row is a container's source->target assignment), then S^T S.
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        schedule = schedule[:self.n_hosts, :self.n_hosts]
        if schedule.dim() == 3:
            schedule = schedule.max(dim=0).values
        adjacency = torch.matmul(schedule.transpose(0, 1), schedule)
        adjacency = adjacency + adjacency.transpose(0, 1)
        adjacency = adjacency + torch.eye(self.n_hosts, device=schedule.device, dtype=schedule.dtype)
        return adjacency

    def _graph_residual(self, t, s):
        x_host = self._per_host_features(t)
        adjacency = self._schedule_adjacency(s)
        query = self.graph_query(x_host)
        key = self.graph_key(x_host)
        value = self.graph_value(x_host)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(16)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        graph_out = torch.matmul(weights, value)            # [hosts, 16]
        interaction = graph_out.mean(dim=-1, keepdim=True)  # [hosts, 1] scalar
        return self.graph_gate(interaction)                 # [hosts, 2+PROTO_DIM]; zero-init

    def _moe_residual(self, t, s):
        return super()._moe_residual(t, s) + self._graph_residual(t, s)


class FTMoE_v4b_16(FTMoE_v3b2_16):
    """v4b: v3b2 + cross multi-head attention fusion (paper Eq. 6-9).

    The MoE path (Q, per-host) actively attends to the schedule-graph path
    (K=V), so each host's fault-specific features query the scheduling
    context that affects it most.  The fused context is projected to a
    zero-init logit residual — at initialization v4b == v0, and the fusion
    only *refines* the baseline heads rather than replacing them.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4b_16'
        self.lr = 0.0001
        self.moe_proj = nn.Linear(self.expert_feat_dim, 16)  # 7 -> 16 to match graph dim
        self.cmha = nn.MultiheadAttention(16, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Linear(16, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _moe_branch_nodes(self, t):
        """MoE-path node features used as Q in the cross attention."""
        return self.moe_proj(self._per_host_features(t))

    def _graph_branch_nodes(self, t, s):
        """Schedule-graph path node features used as K=V (Eq. 1 output)."""
        x_host = self._per_host_features(t)
        adjacency = self._schedule_adjacency(s)
        query = self.graph_query(x_host)
        key = self.graph_key(x_host)
        value = self.graph_value(x_host)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(16)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, value)             # [hosts, 16]

    def _moe_residual(self, t, s):
        # Full dual-path fusion: MoE output + CMHA(MoE, graph, graph),
        # zero-init residual gates keep the v0 baseline intact at step 0.
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)                      # [hosts, 7]
        q = self._moe_branch_nodes(t).unsqueeze(0)      # [1, hosts, 16]
        kv = self._graph_branch_nodes(t, s).unsqueeze(0)  # [1, hosts, 16]
        fused, _ = self.cmha(q, kv, kv)                 # [1, hosts, 16]
        fused = fused.squeeze(0)                        # [hosts, 16]
        return (self.residual_gate(moe_out) +
                self.fusion_gate(fused))                # [hosts, 2+PROTO_DIM]; zero-init


class FTMoE_v4b2_16(FTMoE_v2c2_16):
    """v4b2: v2c2 + cross multi-head attention over differentiated Q/K/V.

    Fixes v4b's failure mode: there the graph branch produced identical
    nodes for every host (S^T S adjacency carries no host-discriminative
    signal) and CMHA attention collapsed to uniform, so the fusion was pure
    noise.  v4b2 gives CMHA differentiated sources: Q = MoE-path per-host
    features (already fault-discriminative), K=V = raw per-host features
    (window-averaged).  The fused context is added through a zero-init
    residual gate, so v4b2 == v0 at init and the baseline stays intact.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4b2_16'
        self.lr = 0.0001
        # Q/K/V projections onto a common 16-dim space.
        self.cmha_q = nn.Linear(self.expert_feat_dim, 16)
        self.cmha_kv = nn.Linear(10, 16)  # raw(3) + per-host(7) concat
        self.cmha = nn.MultiheadAttention(16, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Linear(16, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _raw_features(self, t):
        """Window-averaged raw per-host features: [hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        raw = t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)
        return raw.mean(dim=0)                              # [hosts, 3]

    def _moe_residual(self, t, s):
        x_host = self._per_host_features(t)                 # [hosts, 7]
        moe_out = self.moe(x_host)                          # [hosts, 7]
        q = self.cmha_q(moe_out).unsqueeze(0)               # [1, hosts, 16]
        kv_src = torch.cat((self._raw_features(t), x_host), dim=-1)  # [hosts, 10]
        kv = self.cmha_kv(kv_src).unsqueeze(0)              # [1, hosts, 16]
        fused, _ = self.cmha(q, kv, kv)                     # [1, hosts, 16]
        fused = fused.squeeze(0)
        return (self.residual_gate(moe_out) +
                self.fusion_gate(fused))                    # [hosts, 2+PROTO_DIM]; zero-init


class FTMoE_v4b3_16(FTMoE_v3b2_16):
    """v4b3: v3b2 + cross multi-head attention as a *third* additive residual.

    v4b/v4b2 regressed vs v3b2 because they replaced the winning graph
    branch with a CMHA that had nothing informative to attend to.  v4b3
    keeps the v3b2 signals (MoE residual + graph interaction residual)
    untouched and adds CMHA only as an extra zero-init residual: Q = MoE
    output, K=V = raw window-averaged host features, projected to a common
    space.  At init v4b3 == v0, and the model can only improve if the
    cross attention actually learns useful interactions.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4b3_16'
        self.lr = 0.0001
        self.cmha_q = nn.Linear(self.expert_feat_dim, 16)
        self.cmha_kv = nn.Linear(10, 16)  # raw(3) + per-host(7) concat
        self.cmha = nn.MultiheadAttention(16, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Linear(16, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _raw_features(self, t):
        """Window-averaged raw per-host features: [hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        raw = t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)
        return raw.mean(dim=0)

    def _cmha_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)                          # [hosts, 7]
        q = self.cmha_q(moe_out).unsqueeze(0)               # [1, hosts, 16]
        kv_src = torch.cat((self._raw_features(t), x_host), dim=-1)  # [hosts, 10]
        kv = self.cmha_kv(kv_src).unsqueeze(0)              # [1, hosts, 16]
        fused, _ = self.cmha(q, kv, kv)                     # [1, hosts, 16]
        return self.fusion_gate(fused.squeeze(0))           # [hosts, 2+PROTO_DIM]; zero-init

    def _moe_residual(self, t, s):
        # v3b2 signals + CMHA additive residual.
        return (super()._moe_residual(t, s) +
                self._cmha_residual(t, s))


class FTMoE_v4b4_16(FTMoE_v3b2_16):
    """v4b4: v3b2 + CMHA fusion with scaled-down residual.

    v4b3's CMHA residual magnitude (~0.28) overwhelmed the decoder output
    (~0.87), unlike v3b2's graph residual (~0.02) which is a gentle,
    stable refinement.  v4b4 applies a 0.1 scale to the CMHA fusion path so
    it behaves like v3b2's refinement rather than a competing head.
    Zero-init residuals inherited (v4b4 == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4b4_16'
        self.lr = 0.0001
        self.cmha_scale = 0.1
        self.cmha_q = nn.Linear(self.expert_feat_dim, 16)
        self.cmha_kv = nn.Linear(10, 16)  # raw(3) + per-host(7) concat
        self.cmha = nn.MultiheadAttention(16, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Linear(16, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _raw_features(self, t):
        """Window-averaged raw per-host features: [hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        raw = t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)
        return raw.mean(dim=0)

    def _cmha_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)                          # [hosts, 7]
        q = self.cmha_q(moe_out).unsqueeze(0)               # [1, hosts, 16]
        kv_src = torch.cat((self._raw_features(t), x_host), dim=-1)  # [hosts, 10]
        kv = self.cmha_kv(kv_src).unsqueeze(0)              # [1, hosts, 16]
        fused, _ = self.cmha(q, kv, kv)                     # [1, hosts, 16]
        return self.cmha_scale * self.fusion_gate(fused.squeeze(0))

    def _moe_residual(self, t, s):
        return (super()._moe_residual(t, s) +
                self._cmha_residual(t, s))


class FTMoE_v4b5_16(FTMoE_v3b2_16):
    """v4b5: v3b2 + feature-level self-attention over informative per-host
    features.

    v4b-v4b4 all fused at the logit level through a zero-init gate, which
    absorbed the gradient before the attention could learn, or attended
    over host-identical sources.  v4b5 runs 4-head self-attention on a
    per-host vector built from *informative* signals (window-averaged raw
    metrics, MoE output, graph interaction scalar).  The residual
    self-attention + LayerNorm keeps the attention on the gradient path;
    only the final logit projection is zero-init (v4b5 == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4b5_16'
        self.lr = 0.0001
        self.feat_dim = 12  # raw(3) + per-host(7) + graph scalar(1) + pad(1); divisible by 4 heads
        self.cmha = nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
        self.cmha_norm = nn.LayerNorm(self.feat_dim)
        self.cmha_scale = 0.1
        self.fusion_gate = nn.Linear(self.feat_dim, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _raw_features(self, t):
        """Window-averaged raw per-host features: [hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        raw = t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)
        return raw.mean(dim=0)

    def _cmha_residual(self, t, s):
        x_host = self._per_host_features(t)             # [hosts, 7]
        raw = self._raw_features(t)                     # [hosts, 3]
        adjacency = self._schedule_adjacency(s)
        query = self.graph_query(x_host)
        key = self.graph_key(x_host)
        value = self.graph_value(x_host)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(16)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        interaction = torch.matmul(weights, value).mean(dim=-1, keepdim=True)  # [hosts, 1]
        h = torch.cat((raw, x_host, interaction), dim=-1)   # [hosts, 11]
        pad = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        h = torch.cat((h, pad), dim=-1)                     # [hosts, 12]
        fused, _ = self.cmha(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        fused = self.cmha_norm(h + fused.squeeze(0))        # [hosts, 12]
        return self.cmha_scale * self.fusion_gate(fused)

    def _moe_residual(self, t, s):
        return super()._moe_residual(t, s) + self._cmha_residual(t, s)


class FTMoE_v4c1_16(FTMoE_v2c2_16):
    """v4c1: v2c2 + inter-expert attention inside the MoE.

    Applies 4-head self-attention along the expert axis of the expert
    outputs (before routing), so experts exchange information per host.
    The projection is zero-init (v4c1 == v0 at init), and the graph path
    is not included — the attention replaces the graph branch.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4c1_16'
        self.lr = 0.0001
        # 14-dim embed is divisible by 2 heads.
        self.moe = _MoE_Progressive(self.expert_feat_dim, routing_mode='softgate', inter_expert=2)

    def _moe_residual(self, t, s):
        x_host = self._per_host_features(t)
        moe_out = self.moe(x_host)
        return self.residual_gate(moe_out)


class FTMoE_v4c2_16(FTMoE_v3b2_16):
    """v4c2: v3b2 + cross-time attention over window steps.

    Q = current-step per-host raw features; K=V = all window steps' per-host
    raw features (3 steps x 16 hosts = 48 tokens).  The attention learns
    how each host's current state is influenced by its own and others'
    history — a genuinely cross-time source, unlike the host-identical
    sources tried in v4b/v4c1.  Zero-init residual gates inherited
    (v4c2 == v0 at init).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4c2_16'
        self.lr = 0.0001
        self.cmha_q = nn.Linear(3, 8)
        self.cmha_kv = nn.Linear(3, 8)
        self.cmha = nn.MultiheadAttention(8, num_heads=1, batch_first=True)
        self.cmha_scale = 0.1
        self.fusion_gate = nn.Linear(8, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()

    def _window_steps(self, t):
        """Per-step per-host raw features: [window, hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        return t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)

    def _cmha_residual(self, t, s):
        steps = self._window_steps(t)                       # [3, 16, 3]
        q = self.cmha_q(steps[-1]).unsqueeze(0)             # [1, 16, 8]
        kv_in = steps.reshape(self.n_window * self.n_hosts, 3)  # [48, 3]
        kv = self.cmha_kv(kv_in).unsqueeze(0)               # [1, 48, 8]
        fused, _ = self.cmha(q, kv, kv)                     # [1, 16, 8]
        return self.cmha_scale * self.fusion_gate(fused.squeeze(0))

    def _moe_residual(self, t, s):
        return super()._moe_residual(t, s) + self._cmha_residual(t, s)


class FTMoE_v4c3_16(FTMoE_v3b2_16):
    """v4c3: v3b2 + cross-time attention with *learnable* fusion gate.

    v4c2's fixed 0.1 scale was a guess; v4c3 lets the model learn how
    strongly the cross-time fusion should influence the logits.  The gate
    starts at 0 (v4c3 == v0 at init) and is bounded via sigmoid.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4c3_16'
        self.lr = 0.0001
        self.cmha_q = nn.Linear(3, 8)
        self.cmha_kv = nn.Linear(3, 8)
        self.cmha = nn.MultiheadAttention(8, num_heads=1, batch_first=True)
        self.fusion_gate = nn.Linear(8, 2 + PROTO_DIM)
        self.fusion_gate.weight.data.zero_()
        self.fusion_gate.bias.data.zero_()
        self.gate_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> starts 0.5, learns 0..1

    def _window_steps(self, t):
        """Per-step per-host raw features: [window, hosts, 3]."""
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        return t_sq[:, :self.n_hosts * 3].view(self.n_window, self.n_hosts, 3)

    def _cmha_residual(self, t, s):
        steps = self._window_steps(t)                       # [3, 16, 3]
        q = self.cmha_q(steps[-1]).unsqueeze(0)             # [1, 16, 8]
        kv_in = steps.reshape(self.n_window * self.n_hosts, 3)  # [48, 3]
        kv = self.cmha_kv(kv_in).unsqueeze(0)               # [1, 48, 8]
        fused, _ = self.cmha(q, kv, kv)                     # [1, 16, 8]
        scale = torch.sigmoid(self.gate_logit)
        return scale * self.fusion_gate(fused.squeeze(0))

    def _moe_residual(self, t, s):
        return super()._moe_residual(t, s) + self._cmha_residual(t, s)


class FTMoE_v1_16(Transformer_16):
    """v1: + per-host MoE branch on top of the unchanged PreGAN+ decoder heads.

    The MoE consumes each host's own TransformerDecoder feature slice, so
    different hosts can route to different experts. Its per-host logit
    corrections are added to the baseline head predictions; when the branch is
    masked out, the model is bit-exact PreGAN+ (drop-in ablation).
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v1_16'
        self.lr = 0.0001
        self.n_feats = 3 * self.n_hosts
        self.hidden_dim = 32
        self.moe = _MoE_Progressive(self.hidden_dim, routing_mode='fixed')
        self.moe_proj = nn.Linear((self.n_feats * 2 + 1) * self.n_window, self.hidden_dim)
        self.moe_a_logit = nn.Linear(self.hidden_dim, 2)
        self.moe_c_logit = nn.Linear(self.hidden_dim, PROTO_DIM)
        # Per-host bias-free logit corrections (zero-init so the branch starts
        # neutral and the baseline loss is unchanged at step 0).
        self.moe_a_bias = nn.Parameter(torch.zeros(2))
        self.moe_c_bias = nn.Parameter(torch.zeros(PROTO_DIM))
        self.moe_gate = 1.0

    def _per_host_features(self, memory):
        # memory: (n_window, 1, feats*2+1) -> (n_hosts, window * feats)
        feat = memory.squeeze(1).reshape(self.n_window, self.n_hosts, -1)
        return feat.permute(1, 0, 2).reshape(self.n_hosts, -1)

    def _moe_memory(self, t, s):
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        o = torch.cat((t_sq, gat_t), dim=1)
        o = o * math.sqrt(self.n_feats)
        o = self.pos_encoder(o)
        memory = self.encoder(o)  # (n_window, 1, feats*2+1)
        nodes = self.moe_proj(self._per_host_features(memory))  # (n_hosts, hidden_dim)
        moe_a = self.moe_a_logit(self.moe(nodes))  # (n_hosts, 2)
        moe_c = self.moe_c_logit(self.moe(nodes))  # (n_hosts, PROTO_DIM)
        return memory, moe_a, moe_c

    def anomaly_decode(self, t, memory, moe_a):
        anomaly_scores = self.anomaly_decoder(t, memory)  # (1, n_window, feats*2+1)
        anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
        anomaly_scores = anomaly_scores + self.moe_gate * (moe_a + self.moe_a_bias).unsqueeze(1)
        return anomaly_scores

    def prototype_decode(self, t, memory, moe_c):
        prototypes = self.prototype_decoder(t, memory)  # (1, n_window, feats*2+1)
        prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
        prototypes = prototypes + self.moe_gate * (moe_c + self.moe_c_bias)
        return prototypes

    def forward(self, t, s):
        encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
        t_in = t.unsqueeze(dim=1)
        memory, moe_a, moe_c = self._moe_memory(t_in, s)
        anomaly_scores = self.anomaly_decode(encoded_t, memory, moe_a)
        prototypes = self.prototype_decode(encoded_t, memory, moe_c)
        return anomaly_scores, prototypes

    def routing_regularization(self):
        return self.moe.selection_loss()


class FTMoE_v2_16(FTMoE_v1_16):
    """v2: + EAGate adaptive Top-any expert selection (paper Eq. 2-5)."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v2_16'
        self.lr = 0.0001
        self.moe = _MoE_Progressive(self.hidden_dim, routing_mode='eagate')

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = enabled and name.startswith('moe.') if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()


class FTMoE_v3_16(FTMoE_v2_16):
    """v3: + schedule-aware graph path (paper Eq. 1) as a second per-host branch.
    Each host's features are refined by attention over scheduling-connected
    neighbours, then added to the MoE-corrected predictions as another
    residual logit correction."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v3_16'
        self.lr = 0.0001
        self.hidden_dim = 32
        self.graph_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.graph_a_logit = nn.Linear(self.hidden_dim, 2)
        self.graph_c_logit = nn.Linear(self.hidden_dim, PROTO_DIM)
        self.graph_gate = 1.0

    def _schedule_adjacency(self, schedule):
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        schedule = schedule[:self.n_hosts, :self.n_hosts]
        adjacency = torch.matmul(schedule.transpose(0, 1), schedule)
        adjacency = adjacency + adjacency.transpose(0, 1)
        adjacency = adjacency + torch.eye(self.n_hosts, device=schedule.device, dtype=schedule.dtype)
        return adjacency

    def _graph_encode(self, nodes, schedule):
        adjacency = self._schedule_adjacency(schedule)
        query = self.graph_query(nodes)
        key = self.graph_key(nodes)
        value = self.graph_value(nodes)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return self.graph_norm(nodes + torch.matmul(weights, value))

    def _moe_memory(self, t, s):
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        o = torch.cat((t_sq, gat_t), dim=1)
        o = o * math.sqrt(self.n_feats)
        o = self.pos_encoder(o)
        memory = self.encoder(o)  # (n_window, 1, feats*2+1)
        nodes = self.moe_proj(self._per_host_features(memory))  # (n_hosts, hidden_dim)
        moe_a = self.moe_a_logit(self.moe(nodes))  # (n_hosts, 2)
        moe_c = self.moe_c_logit(self.moe(nodes))  # (n_hosts, PROTO_DIM)
        graph_nodes = self._graph_encode(nodes, schedule=s)
        graph_a = self.graph_a_logit(graph_nodes)
        graph_c = self.graph_c_logit(graph_nodes)
        return memory, moe_a, moe_c, graph_a, graph_c

    def anomaly_decode(self, t, memory, moe_a, graph_a):
        anomaly_scores = self.anomaly_decoder(t, memory)
        anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
        anomaly_scores = anomaly_scores + self.moe_gate * (moe_a + self.moe_a_bias).unsqueeze(1)
        anomaly_scores = anomaly_scores + self.graph_gate * graph_a.unsqueeze(1)
        return anomaly_scores

    def prototype_decode(self, t, memory, moe_c, graph_c):
        prototypes = self.prototype_decoder(t, memory)
        prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
        prototypes = prototypes + self.moe_gate * (moe_c + self.moe_c_bias)
        prototypes = prototypes + self.graph_gate * graph_c
        return prototypes

    def forward(self, t, s):
        encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
        t_in = t.unsqueeze(dim=1)
        memory, moe_a, moe_c, graph_a, graph_c = self._moe_memory(t_in, s)
        anomaly_scores = self.anomaly_decode(encoded_t, memory, moe_a, graph_a)
        prototypes = self.prototype_decode(encoded_t, memory, moe_c, graph_c)
        return anomaly_scores, prototypes


class FTMoE_v4_16(FTMoE_v3_16):
    """v4: full FT-MoE — cross multi-head attention fuses the MoE and graph
    paths (paper Eq. 6-9); the fused per-host context provides the residual
    logit corrections."""

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_v4_16'
        self.lr = 0.0001
        self.hidden_dim = 32
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fused_a_logit = nn.Linear(self.hidden_dim, 2)
        self.fused_c_logit = nn.Linear(self.hidden_dim, PROTO_DIM)
        self.fused_gate = 1.0

    def _fuse(self, moe_nodes, graph_nodes):
        fused, _ = self.cross_attention(
            moe_nodes.unsqueeze(0), graph_nodes.unsqueeze(0), graph_nodes.unsqueeze(0)
        )
        return self.fusion_norm(moe_nodes + fused.squeeze(0))

    def _moe_memory(self, t, s):
        t_sq = t
        if t_sq.dim() > 2:
            t_sq = t_sq.squeeze(1)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        o = torch.cat((t_sq, gat_t), dim=1)
        o = o * math.sqrt(self.n_feats)
        o = self.pos_encoder(o)
        memory = self.encoder(o)
        nodes = self.moe_proj(self._per_host_features(memory))
        moe_nodes = self.moe(nodes)
        graph_nodes = self._graph_encode(nodes, schedule=s)
        fused = self._fuse(moe_nodes, graph_nodes)
        fused_a = self.fused_a_logit(fused)
        fused_c = self.fused_c_logit(fused)
        return memory, fused_a, fused_c

    def anomaly_decode(self, t, memory, fused_a):
        anomaly_scores = self.anomaly_decoder(t, memory)
        anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
        anomaly_scores = anomaly_scores + self.fused_gate * fused_a.unsqueeze(1)
        return anomaly_scores

    def prototype_decode(self, t, memory, fused_c):
        prototypes = self.prototype_decoder(t, memory)
        prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
        prototypes = prototypes + self.fused_gate * fused_c
        return prototypes

    def forward(self, t, s):
        encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
        t_in = t.unsqueeze(dim=1)
        memory, fused_a, fused_c = self._moe_memory(t_in, s)
        anomaly_scores = self.anomaly_decode(encoded_t, memory, fused_a)
        prototypes = self.prototype_decode(encoded_t, memory, fused_c)
        return anomaly_scores, prototypes


class AdaptiveFaultMoE(nn.Module):
    """Sparse MoE router with the paper's adaptive Top-any expert selection.

    The hard routing decision keeps inference sparse.  A straight-through soft
    gate supplies gradients to the expert keys and activation thresholds while
    retaining the hard decision in the forward pass.
    """

    def __init__(self, hidden_dim, num_experts=12, max_active=8,
                 min_experts=4, max_experts=16, temperature=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_active = max_active
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.temperature = temperature
        self.experts = nn.ModuleList([self._make_expert() for _ in range(num_experts)])
        self.expert_keys = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.02)
        self.thresholds = nn.Parameter(torch.zeros(num_experts))
        self.register_buffer('activation_counts', torch.zeros(num_experts))
        self.register_buffer('routing_samples', torch.zeros(1))
        self.register_buffer('unrouted_samples', torch.zeros(1))
        self._selection_loss = torch.tensor(0.0)

    def _make_expert(self):
        width = getattr(self, 'expert_width', None)  # None = full width
        if width is None:
            return nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
        return nn.Sequential(
            nn.Linear(self.hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, self.hidden_dim),
        )

    @property
    def num_experts(self):
        return len(self.experts)

    def reset_routing_stats(self):
        with torch.no_grad():
            self.activation_counts.zero_()
            self.routing_samples.zero_()
            self.unrouted_samples.zero_()

    def forward(self, x):
        # x: [hosts, hidden_dim]
        scores = F.cosine_similarity(x.unsqueeze(1), self.expert_keys.unsqueeze(0), dim=-1)
        probabilities = torch.sigmoid(scores)
        thresholds = torch.sigmoid(self.thresholds).unsqueeze(0)
        eligible = probabilities >= thresholds
        hard_mask = torch.zeros_like(scores)

        # Top-any: use every eligible expert up to max_active; always select
        # the most similar expert when none passes its threshold.
        for row in range(scores.shape[0]):
            selected = torch.nonzero(eligible[row], as_tuple=False).flatten()
            if selected.numel() == 0:
                selected = torch.topk(scores[row], k=1).indices
            elif selected.numel() > self.max_active:
                selected = torch.topk(scores[row], k=self.max_active).indices
            hard_mask[row, selected] = 1.0

        soft_mask = torch.sigmoid((probabilities - thresholds) / self.temperature)
        routing_mask = hard_mask + soft_mask - soft_mask.detach()
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = (expert_outputs * scores.unsqueeze(-1) * routing_mask.unsqueeze(-1)).sum(dim=1)
        output = output / hard_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        self._selection_loss = scores.square().mean()

        # Statistics are intentionally detached: they control structural
        # updates during online tuning and must not alter the current graph.
        with torch.no_grad():
            self.activation_counts.add_(hard_mask.detach().sum(dim=0))
            self.routing_samples.add_(scores.shape[0])
            self.unrouted_samples.add_((eligible.sum(dim=1) == 0).sum())
        return output

    def selection_loss(self):
        return self._selection_loss

    def _replace_router_parameters(self, expert_keys, thresholds):
        self.expert_keys = nn.Parameter(expert_keys.detach().clone())
        self.thresholds = nn.Parameter(thresholds.detach().clone())
        self.activation_counts = torch.zeros(
            self.num_experts, device=expert_keys.device, dtype=expert_keys.dtype)

    def adapt_experts(self):
        """Remove unused experts and add capacity for unmatched inputs.

        Returns True when the parameter topology changed, so the caller can
        rebuild its optimizer with the current expert parameters.
        """
        if self.routing_samples.item() == 0:
            return False
        changed = False
        removable = [
            index for index, count in enumerate(self.activation_counts.tolist())
            if count == 0 and self.num_experts > self.min_experts
        ]
        # Remove at most the excess above the configured minimum.
        removable = removable[:max(0, self.num_experts - self.min_experts)]
        if removable:
            keep = [index for index in range(self.num_experts) if index not in removable]
            for index in sorted(removable, reverse=True):
                del self.experts[index]
            self._replace_router_parameters(self.expert_keys[keep], self.thresholds[keep])
            changed = True

        unrouted_ratio = self.unrouted_samples.item() / self.routing_samples.item()
        if unrouted_ratio >= 0.20 and self.num_experts < self.max_experts:
            parameter = next(self.experts[0].parameters())
            expert = self._make_expert().to(device=parameter.device, dtype=parameter.dtype)
            self.experts.append(expert)
            new_key = torch.randn(1, self.hidden_dim, device=parameter.device, dtype=parameter.dtype) * 0.02
            new_threshold = torch.zeros(1, device=parameter.device, dtype=parameter.dtype)
            self._replace_router_parameters(
                torch.cat((self.expert_keys, new_key), dim=0),
                torch.cat((self.thresholds, new_threshold), dim=0),
            )
            changed = True
        self.reset_routing_stats()
        return changed


class FTMoE_16(nn.Module):
    """FT-MoE adapted to PreGAN+'s 16-host data and recovery interface.

    The model keeps PreGAN+'s anomaly/prototype outputs, allowing the existing
    GAN scheduler to consume its diagnosis embeddings unchanged.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoE_16'
        self.lr = 0.001
        self.n_hosts = 16
        self.n_feats = 3 * self.n_hosts
        self.n_window = 3
        self.n_latent = 10
        self.hidden_dim = 32
        self.loss_weights = (0.35, 0.50, 0.15)

        # Per-host history is projected before schedule-aware graph messaging.
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.n_window * 3, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.graph_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.moe = AdaptiveFaultMoE(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.prototype_head = nn.Sequential(nn.Linear(self.hidden_dim, PROTO_DIM), nn.Sigmoid())
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def _host_history(self, time_window):
        # [window, hosts * metrics] -> [hosts, window * metrics]
        return time_window.reshape(self.n_window, self.n_hosts, 3).permute(1, 0, 2).reshape(self.n_hosts, -1)

    def _schedule_adjacency(self, schedule):
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        schedule = schedule[:self.n_hosts, :self.n_hosts]
        # In PreGAN+ each row is a container assignment.  S^T S connects
        # hosts that share migration pressure from the active task set.
        adjacency = torch.matmul(schedule.transpose(0, 1), schedule)
        adjacency = adjacency + adjacency.transpose(0, 1)
        adjacency = adjacency + torch.eye(self.n_hosts, device=schedule.device, dtype=schedule.dtype)
        return adjacency

    def _schedule_graph_encode(self, nodes, schedule):
        adjacency = self._schedule_adjacency(schedule)
        query = self.graph_query(nodes)
        key = self.graph_key(nodes)
        value = self.graph_value(nodes)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return self.graph_norm(nodes + torch.matmul(weights, value))

    def encode(self, time_window, schedule):
        nodes = self.temporal_encoder(self._host_history(time_window))
        graph_nodes = self._schedule_graph_encode(nodes, schedule)
        moe_nodes = self.moe(nodes)
        fused, _ = self.cross_attention(
            moe_nodes.unsqueeze(0), graph_nodes.unsqueeze(0), graph_nodes.unsqueeze(0)
        )
        return self.fusion_norm(moe_nodes + fused.squeeze(0))

    def routing_regularization(self):
        return self.moe.selection_loss()

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = enabled and name.startswith('moe.') if enabled else True

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()

    def forward(self, time_window, schedule):
        embedding = self.encode(time_window, schedule)
        anomaly_scores = self.anomaly_head(embedding).unsqueeze(1)
        prototypes = self.prototype_head(embedding)
        return anomaly_scores, prototypes


class FTMoEv2_16(nn.Module):
    """FT-MoE v2: Transformer backbone with MoE + EAGate + cross-attention on top.

    Uses the proven Transformer_16 encoder as the base, then adds the MoE path
    together with a schedule-aware graph encoder and cross-attention fusion.
    The anomaly head follows the Transformer_16 pattern (Softmax-normalized)
    for stable convergence.
    """

    def __init__(self):
        super().__init__()
        self.name = 'FTMoEv2_16'
        self.lr = 0.0001
        self.n_hosts = 16
        feats = 3 * self.n_hosts
        self.n_feats = feats
        self.n_window = 3
        self.n_latent = 10
        self.hidden_dim = 32
        self.loss_weights = (0.35, 0.50, 0.15)

        # ---------- Transformer encoder (same as PreGAN+) ----------
        src_ids = torch.tensor(list(range(self.n_feats)))
        dst_ids = torch.tensor([self.n_feats] * self.n_feats)
        self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
        self.time_encoder = nn.Linear(feats, feats * 2 + 1)
        self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        self.project = nn.Linear(feats * 2 + 1, self.hidden_dim)

        # ---------- Schedule-aware graph path ----------
        self.graph_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)

        # ---------- MoE path (online-tunable) ----------
        self.moe = AdaptiveFaultMoE(self.hidden_dim)

        # ---------- Cross-attention fusion ----------
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=4, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # ---------- Heads (Softmax-based, matching Transformer_16) ----------
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.prototype_head = nn.Sequential(
            nn.Linear(self.hidden_dim, PROTO_DIM), nn.Sigmoid()
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    # ---- helpers -------------------------------------------------
    def _host_history(self, time_window):
        return time_window.reshape(self.n_window, self.n_hosts, 3).permute(1, 0, 2).reshape(self.n_hosts, -1)

    def _schedule_adjacency(self, schedule):
        if not isinstance(schedule, torch.Tensor):
            schedule = torch.as_tensor(schedule, dtype=torch.double)
        schedule = schedule[:self.n_hosts, :self.n_hosts]
        adj = torch.matmul(schedule.transpose(0, 1), schedule)
        adj = adj + adj.transpose(0, 1)
        adj = adj + torch.eye(self.n_hosts, device=schedule.device, dtype=schedule.dtype)
        return adj

    def _graph_encode(self, nodes, schedule):
        adj = self._schedule_adjacency(schedule)
        q = self.graph_query(nodes)
        k = self.graph_key(nodes)
        v = self.graph_value(nodes)
        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(adj <= 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return self.graph_norm(nodes + torch.matmul(weights, v))

    def _transformer_encode(self, t, s):
        # t can be (n_window, n_feats), (1, n_window, n_feats), or (n_window, n_feats)
        # Remove ALL leading singleton dims until we get to 2-D
        t_sq = t
        while t_sq.dim() > 2 and t_sq.shape[0] == 1:
            t_sq = t_sq.squeeze(0)
        graph = torch.cat((t_sq, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        o = torch.cat((t_sq, gat_t), dim=1)
        o = o * math.sqrt(self.n_feats)
        o = self.pos_encoder(o)
        memory = self.transformer_encoder(o)
        # memory: (n_window, feats*2+1) → mean pool across window dimension → project
        pooled = memory.mean(dim=0) if memory.dim() == 2 else memory.mean(dim=1).mean(dim=0)
        nodes = self.project(pooled)     # (hidden_dim,)  — 1-D tensor
        return nodes

    def encode(self, time_window, schedule):
        # Transformer backbone → per-timestep context
        context = self._transformer_encode(time_window, schedule)  # (hidden_dim,)  — 1-D

        # Repeat for each host (Transformer output is global; graph path handles per-host structure)
        nodes = context.unsqueeze(0).expand(self.n_hosts, -1)      # (n_hosts, hidden_dim)
        graph_nodes = self._graph_encode(nodes, schedule)           # (n_hosts, hidden_dim)
        moe_nodes = self.moe(nodes)                                 # (n_hosts, hidden_dim)

        fused, _ = self.cross_attention(
            moe_nodes.unsqueeze(0), graph_nodes.unsqueeze(0), graph_nodes.unsqueeze(0)
        )
        embedding = self.fusion_norm(moe_nodes + fused.squeeze(0))
        return embedding

    # ---- online tuning API ---------------------------------------
    def routing_regularization(self):
        return self.moe.selection_loss()

    def set_online_tuning(self, enabled=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                name.startswith('moe.') if enabled else True
            )

    def begin_routing_window(self):
        self.moe.reset_routing_stats()

    def adapt_experts(self):
        return self.moe.adapt_experts()

    # ---- forward -------------------------------------------------
    def forward(self, time_window, schedule):
        embedding = self.encode(time_window, schedule)
        anomaly_scores = self.anomaly_head(embedding)  # (n_hosts, 2) — softmax-normalized
        prototypes = self.prototype_head(embedding)
        # Return same format as Transformer_16: list of n_hosts tensors
        anomaly_list = []
        proto_list = []
        for h in range(self.n_hosts):
            anomaly_list.append(anomaly_scores[h].view(1, -1))  # (1, 2)
            proto_list.append(prototypes[h])                     # (PROTO_DIM,)
        return anomaly_list, proto_list
