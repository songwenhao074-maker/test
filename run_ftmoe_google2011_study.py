"""Complete the registered Google2011 single-seed experiment, preserving all outputs."""
import json
from pathlib import Path
import numpy as np
from run_ftmoe_capacity_study import command,inspect_stream,write,ROOT

OUT=ROOT/'artifacts/ftmoe_online/google2011_018'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'trace.json').exists():command(['prepare_google2011_pilot.py'],OUT/'preparation.log')
    folder=OUT/'stream/seed301_steps300'
    if not (folder/'manifest.json').exists():
        command(['prepare_ftmoe_scenario.py','--seed',301,'--steps',300,'--configuration',OUT/'scenario.json','--output',folder],OUT/'generation.log')
    y=np.load(ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical/labels.npy')[:5]
    counts=np.bincount(y.ravel(),minlength=4)
    distribution=inspect_stream(folder,counts/counts.sum())
    for method in 'ABCD':
        output=OUT/'runs'/method
        if (output/'summary.json').exists():continue
        args=['run_ftmoe_online.py','--method',method,'--model-seed',1,'--stream',folder,'--learning-rate',3e-5,'--output',output]
        if (output/'resume.pt').exists():args+=['--resume']
        command(args,OUT/(method+'.log'))
    summaries={m:json.loads((OUT/'runs'/m/'summary.json').read_text()) for m in 'ABCD'}
    result={'status':'complete','distribution':distribution,
            'second_half_f1':{m:s['metrics']['second_half']['f1'] for m,s in summaries.items()},
            'limitation':'single source cohort, single seed pilot; source data are resource measurements, simulated labels are not original failure annotations'}
    write(OUT/'result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',{'error':str(exc)})
        raise
