import os
import cv2
import json
import h5py
import random
import numpy as np
from typing import List


import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_data_loader(data_dir, batch_size, mode, splits, input_size, 
    is_flip, num_workers=4, distributed=True, debug=False):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if mode == 'train':
        is_shuffle = True
        is_load_label = True
        sub_folder_use = 'train'
    elif mode == 'test':
        is_shuffle = False
        is_load_label = False
        sub_folder_use = 'test'
    elif mode == 'eval':
        is_shuffle = False
        is_load_label = True
        sub_folder_use = 'train'
    elif mode == 'test_specific':
        raise NotImplementedError
    else:
        raise ValueError

    refer_list_file = os.path.join("../dataset", splits)
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    data_set = GazeDataset(
        dataset_path=data_dir, 
        keys_to_use=datastore[mode],
        sub_folder=sub_folder_use,
        transform=transform, 
        is_shuffle=is_shuffle, 
        is_flip=is_flip,
        is_load_label=is_load_label,
        debug=debug
    )

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        sampler = torch.utils.data.distributed.DistributedSampler(data_set, shuffle=is_shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
        data_loader = DataLoader(
            data_set,
            batch_sampler=batchsampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    return data_loader


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 is_flip=False, index_file=None, is_load_label=True, debug=False):
        self.path = dataset_path
        self.hdfs = {}
        self.is_flip = is_flip
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        if debug:
            print("Dataset in debug mode.")
            keys_to_use = keys_to_use[:1]

        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        # image = self.transform(image)

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')

            if self.sub_folder in ['train'] and self.is_flip:
                if random.random() < 0.5:
                    image = cv2.flip(image, 1)
                    gaze_label = np.array([gaze_label[0], -gaze_label[1]])

            image = self.transform(image)
            return image, gaze_label
        else:
            image = self.transform(image)
            return image