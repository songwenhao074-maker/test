"""Read completed pilot checkpoints to describe training/development fit gaps."""
import argparse
import json
import torch
from train_ftmoe_end_to_end import ART, ROOT, create_model, evaluate, guard, load_data, sha, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True); args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    results = {}
    for run in args.runs:
        root = ART/'runs'/run
        config = json.loads((root/'configuration.json').read_text())['arguments']
        training,validation,norm,_ = load_data(ROOT/config['data'])
        source = root/'v1_seed1'; assert (source/'summary.json').exists()
        results[run] = {}
        for protocol,filename in [('A_last','last.pt'),('B_best','best.pt')]:
            guard()
            state = torch.load(source/filename,map_location='cpu',weights_only=False)
            model = create_model('v1',1,norm,**state.get('model_options',{}))
            model.load_state_dict(state['model'],strict=True)
            results[run][protocol] = {'epoch':state['epoch'],'checkpoint_sha256':sha(source/filename),
                'training':evaluate(model,{'train':training})['mean'],
                'development':evaluate(model,validation)['mean']}
    write_json(ART/args.output,{'scope':'completed seed1 pilot checkpoints; no optimizer updates or selection', 'results':results})
    keys = ['f1','diagnosis_hr_at_100pct','positive_class_macro_f1']
    print(json.dumps({run:{p:{split:{k:v[split][k] for k in keys} for split in ['training','development']}
                      for p,v in rows.items()} for run,rows in results.items()},indent=2))


if __name__=='__main__': main()
