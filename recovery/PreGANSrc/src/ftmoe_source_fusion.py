"""Inactive host-aligned, two-source attention and matched gated controls."""
import torch
from torch import nn
import torch.nn.functional as F
from .ftmoe_fusion_controls import GatedCrossFusion


class SourceAttentionFusion(nn.Module):
    """One query attends time and graph source tokens of the same host."""
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.train(attention.training)

    def forward(self, query, key, value, need_weights=False):
        assert query.shape==key.shape==value.shape
        batch,hosts,hidden = query.shape
        q = query.reshape(batch*hosts,1,hidden)
        k = torch.stack([query,key],dim=2).reshape(batch*hosts,2,hidden)
        v = torch.stack([query,value],dim=2).reshape(batch*hosts,2,hidden)
        result,_ = self.attention(q,k,v,need_weights=False)
        return result.reshape(batch,hosts,hidden),None


class SourceGatedFusion(GatedCrossFusion):
    """The same two source values and parameter count, with featurewise gates."""
    def forward(self, query, key, value, need_weights=False):
        assert query.shape==key.shape==value.shape
        gate = torch.sigmoid(self.query(query)+self.key(key))
        weights = F.dropout(torch.stack([1-gate,gate],dim=-2),p=self.dropout,training=self.training)
        values = torch.stack([self.value(query),self.value(value)],dim=-2)
        return self.output((weights*values).sum(-2)),None
