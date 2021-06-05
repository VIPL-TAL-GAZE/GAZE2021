import os
import numpy as np
import json


data_root = '/dataset/ETH-XGaze-wpc/v3/xgaze_224/Lamks_results'
output_root = '/mnt/data/dataset/xgaze/lmks'


def save_json(fn, data):
    with open(fn, 'w') as f:
        json.dump(data, f)


def get_non_zero_inds(eye_center_data):
    data_len = eye_center_data.shape[0]
    ind_set = set(range(data_len))
    zero_ind_list = np.argwhere(
        np.linalg.norm(eye_center_data, axis=1) == 0
    ).reshape(-1).tolist()
    zero_ind_set = set(zero_ind_list)
    nonzero_inds = list(ind_set - zero_ind_set)
    return nonzero_inds


def statistic_data(data):
    assert isinstance(data, np.ndarray), type(data)
    # assert data.shape[-1] == 2, data.shape
    # assert data.ndim == 2, data.ndim

    mu = data.mean(axis=0).tolist()
    std = data.std(axis=0).tolist()
    minimum = data.min(axis=0).tolist()
    maximum = data.max(axis=0).tolist()
    return {'mean': mu, 'std': std, 'min': minimum, 'max': maximum}


def process_file(args):
    split_dir = args[0]
    fname = args[1]
    print(*args)
    fn_in = os.path.join(data_root, split_dir, fname)

    data = np.array(np.load(fn_in, allow_pickle=True).item()['landmarks'])

    leye = data[:, 0, 74, :]
    leye_outer = data[:, 0, 66, :]
    leye_inner = data[:, 0, 70, :]
    leye_length = np.linalg.norm(leye_outer - leye_inner, axis=1)

    reye = data[:, 0, 83, :]
    reye_outer = data[:, 0, 79, :]
    reye_inner = data[:, 0, 75, :]
    reye_length = np.linalg.norm(reye_outer - reye_inner, axis=1)

    nonzero_inds = get_non_zero_inds(leye)

    results = {
        'left_eye_center': statistic_data(leye[nonzero_inds]),
        'left_eye_length': statistic_data(leye_length[nonzero_inds]),
        'right_eye_center': statistic_data(reye[nonzero_inds]),
        'right_eye_length': statistic_data(reye_length[nonzero_inds])
    }

    fn_out = os.path.join(output_root, split_dir, fname.replace('npy', 'json'))
    save_json(fn_out, results)
    return results


def collect_data(data_list, key):
    data = {}
    for name in ['mean', 'min', 'max']:
        key_data_list = list(map(lambda x: x[key][name], data_list))
        data[name] = eval('np.array(key_data_list).{}(axis=0).tolist()'.format(name))
    return data


def process_total(fn, data_list):
    data = {}
    for key in ['left_eye_center', 'left_eye_length', 'right_eye_center', 'right_eye_length']:
        data[key] = collect_data(data_list, key)
    print(data)
    save_json(fn, data)


if __name__ == '__main__':
    from multiprocessing import Pool

    p = Pool(32)
    ret_list = []
    for dir_name in os.listdir(data_root):
        dpath = os.path.join(data_root, dir_name)
        dpath_out = os.path.join(output_root, dir_name)
        if not os.path.exists(dpath_out):
            os.mkdir(dpath_out)
        for file_name in os.listdir(dpath):
            args = [dir_name, file_name]
            ret_list.append(p.apply_async(process_file, args=(args,)))
    p.close()
    p.join()

    data_list = [t.get() for t in ret_list]
    fn_out = os.path.join(output_root, 'results.json')
    process_total(fn_out, data_list)
