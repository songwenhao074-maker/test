"""Check architecture metadata round trips and the registered pilot tie rule."""
import io
import json
import torch
from train_ftmoe_end_to_end import create_model
from select_ftmoe_context_pilot import select_mode


def main():
    torch.set_num_threads(1)
    for mode, probability in [('none',0.), ('local',0.), ('global',0.), ('none',.3), ('global',.3)]:
        options = {'moe_context':mode, 'expert_dropout':probability}
        original = create_model('v4', 1, **options).eval()
        if mode != 'none':
            with torch.no_grad():
                original.moe.context_projection.weight.normal_(std=.1)
        state = {'model': original.state_dict(), 'model_options': options}
        memory = io.BytesIO(); torch.save(state, memory); memory.seek(0)
        state = torch.load(memory, weights_only=False)
        restored = create_model('v4', 1, **state.get('model_options', {})).eval()
        restored.load_state_dict(state['model'], strict=True)
        x = torch.rand(2,16,12,7)
        schedule = torch.eye(16)[None,None].expand(2,12,-1,-1)
        with torch.no_grad():
            expected = original(x,schedule,x)
            actual = restored(x,schedule,x)
        for key in ['detection_logits', 'class_logits']:
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
    assert select_mode({'none': .900, 'local': .901, 'global': .9015}) == 'none'
    assert select_mode({'none': .900, 'local': .910, 'global': .911}) == 'local'
    assert select_mode({'none': .900, 'local': .910, 'global': .914}) == 'global'
    print(json.dumps({'checkpoint_modes_round_trip': True, 'registered_ties': True}))


if __name__ == '__main__':
    main()
