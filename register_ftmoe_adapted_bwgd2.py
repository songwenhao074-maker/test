"""Freeze a single pilot configuration using training-only disk statistics."""
import hashlib
import json
from pathlib import Path
import datetime
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016'


def main():
    if OUT.exists():
        raise FileExistsError('Registration is immutable: '+str(OUT))
    data=ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical'
    manifest=json.loads((data/'manifest.json').read_text())
    source=data/'container_demand_series.npy'
    disk=np.load(source)[manifest['train_blocks']].reshape(5,202,16,7)[...,4]
    values,counts=np.unique(disk,return_counts=True)
    states=np.searchsorted(values,disk)
    transition=np.zeros((len(values),len(values)),dtype=np.int64)
    np.add.at(transition,(states[:,:-1].ravel(),states[:,1:].ravel()),1)
    initial=counts/counts.sum()
    probability=np.divide(transition,transition.sum(1,keepdims=True),
        out=np.tile(initial,(len(values),1)),where=transition.sum(1,keepdims=True)>0)
    law={'method':'empirical first-order Markov occupancy; synthetic MB, not measured Bitbrain storage',
         'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'train_blocks':manifest['train_blocks'],'values':values.tolist(),
         'initial_probability':initial.tolist(),'transition':probability.tolist(),
         'transition_counts':transition.tolist(),'smoothing':False,
         'rng':'numpy default_rng SeedSequence([replay_seed, creation_id, 16016])',
         'statistics_scope':'training demands only; no model predictions, labels, development or test data',
         'limitation':'per-task marginal and first-order persistence only; cross-resource and cross-task correlation not preserved'}
    protocol={'protocol':16,'name':'single adapted BWGD2 pilot','registered_local':datetime.datetime.now().astimezone().isoformat(),
        'model_seed':1,'replay_seed':301,'steps':300,'guard_steps':1,'methods':list('ABCD'),
        'learning_rate':3e-5,'cpu_positive_clip':[2.,1860.],'ram_multiplier':2.,'io_per_container':1.,
        'capacity_scales':{'CPU_CAP_SCALE':.8,'DISK_CAP_SCALE':.25},
        'native_arrivals_and_vm_selection':True,'synthetic_dynamic_disk':True,'artificial_time_drift':False,
        'keep_original_checkpoint_normalization':True,'keep_original_online_rules':True,
        'distribution_target':{'main_positive_fraction':[.05,.25],'cpu_dominant':True,'all_three_raw_resources_present':True},
        'execution_boundary':'One pilot only. Report before any calibration, normalization change or expansion. If distribution check fails, report without model tuning.',
        'criterion':'D second-half F1 >= 0.65 and D > B; D versus C reported separately',
        'previous_result_preserved':'artifacts/ftmoe_online/pilot',
        'disk_law_sha256':hashlib.sha256((json.dumps(law,indent=2)+'\n').encode()).hexdigest()}
    OUT.mkdir(parents=True)
    for name,obj in [('disk_law',law),('protocol',protocol)]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'registered':str(OUT),'disk_values':law['values'],'mean_MB':float((values*initial).sum())}))


if __name__=='__main__':main()
