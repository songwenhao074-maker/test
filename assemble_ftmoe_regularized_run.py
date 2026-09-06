"""Assemble protocol 010's exact unchanged v0 controls with completed new runs."""
import argparse
import json
import shutil
import torch
from train_ftmoe_end_to_end import ART, ROOT, aggregate, guard, sha, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True); args = parser.parse_args()
    selection_path = ART/'selection_010_expert_dropout.json'
    selection = json.loads(selection_path.read_text())
    probability = selection['selected_probability']; assert probability>0
    new_root = ART/'runs'/args.run
    configuration = json.loads((new_root/'configuration.json').read_text())
    arguments = configuration['arguments']
    assert arguments['epochs']==60 and arguments['seeds']==[1,2,6]
    assert set(arguments['variants'])=={'v1','v2','v3','v4'}
    assert arguments['expert_dropout']==probability and arguments['moe_context']=='none'
    destination = ART/'runs'/args.output
    assert not destination.exists(), 'Assembly output already exists'
    sources = {}; summaries = []; normalization = None
    for variant in ['v0','v1','v2','v3','v4']:
        for seed in [1,2,6]:
            guard()
            source = (ART/'runs/physical_lr003_e60' if variant=='v0' else new_root)/f'{variant}_seed{seed}'
            summary = json.loads((source/'summary.json').read_text())
            assert summary['variant']==variant and summary['seed']==seed and summary['all_parameters_trainable']
            for key in ['data','epochs','stop_after','learning_rate']:
                assert summary['configuration'][key]==arguments[key], key
            for protocol,filename in [('A_last','last.pt'),('B_best','best.pt')]:
                state = torch.load(source/filename,map_location='cpu',weights_only=False)
                assert state['variant']==variant and state['seed']==seed
                assert state['validation']==summary[protocol]
                assert 1<=state['epoch']<=60 and (protocol!='A_last' or state['epoch']==60)
                options = state.get('model_options',{})
                assert options.get('moe_context','none')=='none'
                assert variant=='v0' or options.get('expert_dropout',0.)==probability
                if normalization is None: normalization = state['normalization']
                assert state['normalization']==normalization
            summaries.append(summary)
            sources[source.name] = {'source':str(source),'files':{name:sha(source/name)
                for name in ['summary.json','last.pt','best.pt','epochs.csv']}}
    destination.mkdir(parents=True)
    lock = {'kind':'exact artifact assembly; no training or checkpoint selection performed',
            'selected_probability':probability,'selection_sha256':sha(selection_path),
            'unchanged_v0_source':'physical_lr003_e60','new_training_run':args.run,'sources':sources,
            'disclosure':'v0 has no experts, so dropout is inapplicable. Its independent 60-epoch seed-paired controls are reused byte-for-byte. All original summaries and training configurations remain unchanged.'}
    write_json(destination/'assembly_provenance.json',lock)
    for name,entry in sources.items():
        target = destination/name; target.mkdir()
        for filename,digest in entry['files'].items():
            source = ROOT/entry['source']/filename
            assert sha(source)==digest
            shutil.copy2(source,target/filename)
            assert sha(target/filename)==digest
    # This configuration describes the assembled comparison, not a new training execution.
    configuration['arguments'] = dict(arguments,run=args.output,variants=['v0','v1','v2','v3','v4'])
    configuration['assembly'] = lock
    write_json(destination/'configuration.json',configuration)
    write_json(destination/'aggregate.json',aggregate(summaries))
    print(json.dumps(aggregate(summaries),indent=2))


if __name__=='__main__': main()
