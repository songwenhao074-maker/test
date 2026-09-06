"""Parameter-matched nodewise gated control for cross-path attention."""
import torch
from torch import nn
import torch.nn.functional as F


class GatedCrossFusion(nn.Module):
    def __init__(self, attention):
        super().__init__()
        assert attention.batch_first and attention._qkv_same_embed_dim
        hidden = attention.embed_dim
        self.dropout = attention.dropout
        with torch.random.fork_rng(devices=[]):
            self.query = nn.Linear(hidden,hidden).to(attention.in_proj_weight)
            self.key = nn.Linear(hidden,hidden).to(attention.in_proj_weight)
            self.value = nn.Linear(hidden,hidden).to(attention.in_proj_weight)
            self.output = nn.Linear(hidden,hidden).to(attention.in_proj_weight)
        with torch.no_grad():
            for index,projection in enumerate([self.query,self.key,self.value]):
                projection.weight.copy_(attention.in_proj_weight[index*hidden:(index+1)*hidden])
                projection.bias.copy_(attention.in_proj_bias[index*hidden:(index+1)*hidden])
            self.output.load_state_dict(attention.out_proj.state_dict())
        self.train(attention.training)

    def forward(self, query, key, value, need_weights=False):
        assert query.shape==key.shape==value.shape
        gate = torch.sigmoid(self.query(query)+self.key(key))
        gate = F.dropout(gate,p=self.dropout,training=self.training)
        return self.output(gate*self.value(value)), None
