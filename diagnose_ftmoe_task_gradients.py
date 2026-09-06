"""Training-only gradient alignment; no optimizer steps or model selection."""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from train_ftmoe_end_to_end import ART, ROOT, create_model, guard, load_data, sha, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    run = ART / 'runs' / args.run
    config = json.loads((run/'configuration.json').read_text())['arguments']
    training, _, norm, _ = load_data(ROOT/config['data'])
    indices = torch.randperm(len(training[0]),generator=torch.Generator().manual_seed(801))[:256].split(32)
    results = {}
    for variant in ['v0','v1','v2','v3','v4']:
        guard()
        path = run/f'{variant}_seed1/best.pt'
        state = torch.load(path,map_location='cpu',weights_only=False)
        model = create_model(variant,1,norm,**state.get('model_options',{})).eval()
        model.load_state_dict(state['model'],strict=True)
        parameters = list(model.named_parameters()); rows = []
        for batch in indices:
            x,g,s,y = (value[batch] for value in training)
            output = model(x,s,g); positive = y>0
            detection = .7*F.cross_entropy(output['detection_logits'].reshape(-1,2),positive.long().flatten(),
                                          weight=torch.tensor([.6,2.]))
            classification = .3*F.cross_entropy(output['class_logits'][positive],y[positive]-1)
            p = output['detection_logits'].softmax(-1)[...,1]
            c = output['class_logits'].softmax(-1)
            correct = c.gather(-1,(y.clamp_min(1)-1).unsqueeze(-1)).squeeze(-1)
            ranking = .5*F.softplus(.15+(p[~positive]*c[~positive].max(-1).values)[None,:]
                                    -(p[positive]*correct[positive])[:,None]).mean()
            gradients = {task:torch.autograd.grad(loss,[p for _,p in parameters],retain_graph=True,allow_unused=True)
                         for task,loss in [('detection',detection),('classification',classification),('ranking',ranking)]}
            for group in ['all','encoder','moe','eagate','graph_encoder','cmha']:
                ids = [i for i,(name,_) in enumerate(parameters) if group=='all' or name.startswith(group+'.')]
                if not ids: continue
                vector = {task:torch.cat([(values[i] if values[i] is not None else torch.zeros_like(parameters[i][1])).flatten()
                                          for i in ids]) for task,values in gradients.items()}
                norms = {task:float(value.norm()) for task,value in vector.items()}
                row = {'group':group, 'norms':norms}
                for a,b in [('detection','classification'),('detection','ranking'),('classification','ranking')]:
                    row[a+'_vs_'+b] = float(torch.dot(vector[a],vector[b])/(vector[a].norm()*vector[b].norm()).clamp_min(1e-12))
                rows.append(row)
        groups = {}
        for group in {r['group'] for r in rows}:
            selected = [r for r in rows if r['group']==group]
            groups[group] = {'mean_norm':{k:float(np.mean([r['norms'][k] for r in selected])) for k in selected[0]['norms']}}
            for key in ['detection_vs_classification','detection_vs_ranking','classification_vs_ranking']:
                values = [r[key] for r in selected]
                groups[group][key] = {'mean_cosine':float(np.mean(values)), 'negative_fraction':float(np.mean(np.asarray(values)<0))}
        results[variant] = {'checkpoint':str(path),'sha256':sha(path),'epoch':state['epoch'],'groups':groups}
    write_json(ART/f'{args.run}_task_gradients.json',{'scope':'training only; eval mode; no optimizer updates; local gradient diagnostic',
               'batches':[i.tolist() for i in indices],'results':results})
    print(json.dumps({v:r['groups']['all'] for v,r in results.items()},indent=2))


if __name__=='__main__': main()
