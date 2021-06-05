import os
import json
import torch
import yaml
import numpy as np


def load_json(fn):
    with open(fn, 'r') as f:
        return json.load(f)


def save_json(fn, data):
    with open(fn, 'w') as f:
        json.dump(data, f)


def save_checkpoint(state, save_dir, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # filename = directory + filename
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)


def load_checkpoint(model: torch.nn.Module, ckpt_fn, load_partial=True):
    state_dict = torch.load(ckpt_fn, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except AttributeError as err:
        if not load_partial:
            raise err
        else:
            model_state_dict = model.state_dict()
            no_load = []
            load_states = {}
            for k in model_state_dict:
                if k in state_dict:
                    load_states[k] = state_dict[k]
                else:
                    no_load.append(k)
            model.load_state_dict(load_states)
            print('[WARING] Weights {} connot load.'.format(no_load))


def load_configs(yaml_fn):
    with open(yaml_fn, 'r') as f:
        return yaml.load(f)


def save_results(res):
    np.savetxt('within_eva_results.txt', res, delimiter=',')

    if os.path.exists('submission_within_eva.zip'):
        os.system('rm submission_within_eva.zip')

    os.makedirs('submission_within_eva')
    os.system('mv within_eva_results.txt ./submission_within_eva')
    os.system('zip -r submission_within_eva.zip submission_within_eva')
    os.system('rm -rf submission_within_eva')