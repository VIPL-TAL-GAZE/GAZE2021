import os
import pandas as pd
import numpy as np


def make_eye_dataframe(eye_data, mode='left'):
    assert mode in ['left', 'right']
    if mode == 'left':
        cent_ind = 74
        outer_ind = 66
        inner_ind = 70
    else:
        cent_ind = 83
        outer_ind = 79
        inner_ind = 75

    eye_cent = eye_data[:, 0, cent_ind, :].copy()
    eye_outer = eye_data[:, 0, outer_ind, :].copy()
    eye_inner = eye_data[:, 0, inner_ind, :].copy()
    eye_length = np.linalg.norm(eye_outer - eye_inner, axis=1).reshape(-1)

    df = pd.DataFrame({
        'ind': list(range(len(eye_cent))),
        'center_x': eye_cent[:, 0].tolist(),
        'center_y': eye_cent[:, 1].tolist(),
        'length': eye_length.tolist()
    })

    return df


def get_zeros_ind(df: pd.DataFrame):
    return df[(df['center_x'] == 0) & (df['center_y'] == 0)]['ind'].tolist()


def peek_landmark_file(lmk_fn):
    data = np.array(np.load(lmk_fn, allow_pickle=True).item()['landmarks'])

    leye_df = make_eye_dataframe(data, mode='left')
    reye_df = make_eye_dataframe(data, mode='right')

    print(leye_df)
    print(leye_df[(leye_df['center_x'] == 0) & (leye_df['center_y'] == 0)]['ind'].tolist())




if __name__ == '__main__':
    lmk_fn = '/dataset/ETH-XGaze-wpc/v3/xgaze_224/Lamks_results/train/subject0028.npy'

    peek_landmark_file(lmk_fn)
