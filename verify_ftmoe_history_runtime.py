"""Check that default current training preserves the reusable historical model."""
import ast
import importlib.util
import json
import torch
from train_ftmoe_end_to_end import ROOT, ART, create_model, load_data, loss_fn, sha, write_json


def main():
    torch.set_num_threads(1)
    root = ART/'runs/physical_lr0003_e30'
    configuration = json.loads((root/'configuration.json').read_text())
    verified = {}
    for name, digest in configuration['code_sha256'].items():
        original = root/'source_snapshot'/__import__('pathlib').Path(name).name
        assert sha(original) == digest
        if name != 'train_ftmoe_end_to_end.py':
            assert sha(ROOT/name) == digest, name
        verified[name] = digest
    assert sha(ROOT/configuration['arguments']['data']/'manifest.json') == configuration['manifest_sha256']
    old_path = root/'source_snapshot/train_ftmoe_end_to_end.py'
    spec = importlib.util.spec_from_file_location('historical_trainer', old_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    old_tree = ast.parse(old_path.read_text())
    new_tree = ast.parse((ROOT/'train_ftmoe_end_to_end.py').read_text())
    def function(tree, name):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name==name)
    assert ast.dump(function(old_tree,'load_data')) == ast.dump(function(new_tree,'load_data'))
    # Optimizer and scheduler definitions, temperature and actual updates are unchanged.
    def update_loop(tree):
        train = function(tree, 'train_one')
        loop = next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
        stop = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='validation_metrics' for t in n.targets))
        return ast.dump(ast.Module(body=loop.body[:stop],type_ignores=[]))
    assert update_loop(old_tree) == update_loop(new_tree)
    for name in ['optimizer','scheduler','generator']:
        def assignment(tree):
            return next(n for n in function(tree,'train_one').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert ast.dump(assignment(old_tree)) == ast.dump(assignment(new_tree))
    training, _, normalization, _ = load_data(ROOT/configuration['arguments']['data'])
    x,g,s,y = [t[:8] for t in training]
    results = []
    for variant in ['v0','v1','v2','v3','v4']:
        a = old.create_model(variant,1,normalization)
        b = create_model(variant,1,normalization)
        assert all(torch.equal(p,b.state_dict()[k]) for k,p in a.state_dict().items())
        for model in [a,b]:
            model.train(); model.set_eagate_temperature(1.)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=1e-4)
            torch.manual_seed(10001)
            loss = loss_fn(model,model(x,s,g),y,.7,.3,0.,.01,2.,.5)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        assert all(torch.equal(p,b.state_dict()[k]) for k,p in a.state_dict().items()), variant
        results.append({'variant':variant,'initial_and_one_update_bitwise_identical':True})
    report = {'historical_code_sha256':verified,'data_manifest_sha256':configuration['manifest_sha256'],
        'data_loading_and_training_updates_ast_identical':True,'models':results,
        'difference':'Current reporting uses corrected diagnosis metrics; historical best checkpoints were reselected from all saved epochs under the same corrected score. Validation score never changes optimizer updates or fixed epoch horizon.',
        'scope':'Default options only; the new fusion control intentionally replaces attention.'}
    write_json(ART/'history_runtime_compatibility.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
