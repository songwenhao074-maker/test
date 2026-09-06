"""Read-only cross-attention alignment and inference perturbation diagnostics."""
import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from train_ftmoe_end_to_end import ART, ROOT, create_model, evaluate, guard, load_data, sha, write_json


def attention_weights(module, query, key):
    hidden = module.embed_dim; heads = module.num_heads; depth = hidden//heads
    q = F.linear(query,module.in_proj_weight[:hidden],module.in_proj_bias[:hidden])
    k = F.linear(key,module.in_proj_weight[hidden:2*hidden],module.in_proj_bias[hidden:2*hidden])
    q = q.reshape(query.shape[0],query.shape[1],heads,depth).transpose(1,2)
    k = k.reshape(key.shape[0],key.shape[1],heads,depth).transpose(1,2)
    return ((q @ k.transpose(-2,-1))/math.sqrt(depth)).softmax(-1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    args = parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    check = torch.nn.MultiheadAttention(64,4,batch_first=True).eval()
    fixture = torch.rand(2,16,64)
    with torch.no_grad():
        torch.testing.assert_close(attention_weights(check,fixture,fixture),
            check(fixture,fixture,fixture,average_attn_weights=False)[1],rtol=1e-6,atol=1e-7)
    root = ART/'runs'/args.run
    config = json.loads((root/'configuration.json').read_text())['arguments']
    _,validation,norm,_ = load_data(ROOT/config['data'])
    results = {}
    for seed in [1,2,6]:
        guard(); path = root/f'v4_seed{seed}/best.pt'
        state = torch.load(path,map_location='cpu',weights_only=False)
        model = create_model('v4',seed,norm,**state.get('model_options',{}))
        rows = {}; samples = []
        for case in ['original','zero_cross_only','same_host_cross_only','zero_interaction_only']:
            model.load_state_dict(state['model']); model.eval()
            if case=='zero_interaction_only':
                with torch.no_grad():
                    model.cmha_interaction_proj.weight.zero_(); model.cmha_interaction_proj.bias.zero_()
            def intervene(module,arguments,result):
                if case=='zero_cross_only': return torch.zeros_like(result[0]),None
                if case=='same_host_cross_only':
                    hidden = module.embed_dim
                    values = F.linear(arguments[2],module.in_proj_weight[2*hidden:],module.in_proj_bias[2*hidden:])
                    return module.out_proj(values),None
                if case=='original':
                    weights = attention_weights(module,arguments[0],arguments[1])
                    samples.append({'n':arguments[0].shape[0],
                        'mean_self_attention':float(weights.diagonal(dim1=-2,dim2=-1).mean()),
                        'normalized_entropy':float(-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean()/math.log(weights.shape[-1])),
                        'cross_norm':float(result[0].norm(dim=-1).mean()),
                        'interaction_norm':float(model.cmha_interaction_proj(arguments[0]*arguments[1]).norm(dim=-1).mean())})
            handle = model.cmha.register_forward_hook(intervene)
            rows[case] = evaluate(model,validation)
            handle.remove()
        stats = {k:float(np.average([row[k] for row in samples],weights=[row['n'] for row in samples]))
                 for k in samples[0] if k!='n'}
        results[seed] = {'checkpoint_sha256':sha(path),'epoch':state['epoch'],'attention':stats,'cases':rows}
    report = {'scope':'development-only inference perturbations; no retraining or checkpoint selection',
              'uniform_self_attention':1/16,'results':results}
    write_json(ART/f'{args.run}_cross_attention_diagnostic.json',report)
    print(json.dumps({s:{'attention':r['attention'],'cases':{case:{k:v['mean'][k] for k in ['f1','diagnosis_hr_at_100pct']}
          for case,v in r['cases'].items()}} for s,r in results.items()},indent=2))


if __name__=='__main__': main()
