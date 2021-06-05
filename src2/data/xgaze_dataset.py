import os
import json
import h5py
import random
import cv2
import numpy as np
from typing import List


import torch
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Pad(10),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])


def get_train_loader(data_dir, batch_size, image_scale=1, split_file=None, num_workers=4, is_shuffle=True):
    if split_file is None:
        split_file = 'train_test_split.json'
    # load dataset
    refer_list_file = os.path.join("../datasets", split_file)
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, image_scale=image_scale, is_load_label=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir, batch_size, image_scale=1, split_file=None, num_workers=4, is_shuffle=True):
    if split_file is None:
        split_file = 'train_test_split.json'
    # load dataset
    refer_list_file = os.path.join("../datasets", split_file)
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, image_scale=image_scale, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


def get_data_loader(data_dir, batch_size, image_scale=1, split_file=None, mode='train', num_workers=4, distributed=True, debug=False):

    if mode == 'train':
        is_shuffle = True
        is_load_label = True
        sub_folder_use = 'train'
        transform = transform_train
    elif mode == 'test':
        is_shuffle = False
        is_load_label = False
        sub_folder_use = 'test'
        transform = transform_test
    elif mode == 'eval':
        is_shuffle = False
        is_load_label = True
        sub_folder_use = 'train'
        transform = transform_test
    elif mode == 'test_specific':
        raise NotImplementedError
    else:
        raise ValueError

    if split_file is None:
        split_file = 'train_eval_test_split.json'
    refer_list_file = os.path.join("../datasets", split_file)
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    data_set = GazeDataset(
        mode=mode,
        dataset_path=data_dir,
        keys_to_use=datastore[mode],
        sub_folder=sub_folder_use,
        transform=transform,
        is_shuffle=is_shuffle,
        is_load_label=is_load_label,
        image_scale=image_scale,
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
    def __init__(
            self,
            mode,
            dataset_path: str,
            keys_to_use: List[str] = None,
            sub_folder='',
            transform=None,
            is_shuffle=True,
            index_file=None,
            is_load_label=True,
            image_scale=1,
            debug=False
    ):
        self.path = dataset_path
        self.hdfs = {}
        self.lmks = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.image_scale = image_scale

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
        self.flip = 0.5 if mode == 'train' else 0.
        self.random_crop = True if mode == 'train' else False

        # self.flip = 0
        # self.random_crop = False

        #
        self.lmk_dir = os.path.join('../datasets/xgaze_landmark', self.sub_folder)
        # self.lmk_hdfs = {}
        # for subj_name in self.selected_keys:
        #     lmk_path = os.path.join(lmk_dir, subj_name)
        #     if not os.path.exists(lmk_path):
        #         lmk_path.replace('.h5', '.npy')
        #         lmk_file = np.load(lmk_path, mmap_mode='r')
        #     else:
        #         lmk_file = h5py.File(lmk_path, 'r', swmr=True)
        #     self.lmk_hdfs[subj_name] = lmk_file


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

        self.hdf_lmk = h5py.File(os.path.join(self.lmk_dir, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf_lmk.swmr_mode
        lmk = self.hdf_lmk['landmark'][idx].copy()

        image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        lmk *= self.image_scale

        if self.random_crop:
            image = np.pad(image, [[10*self.image_scale, 10*self.image_scale], [10*self.image_scale, 10*self.image_scale], [0, 0]])
            lmk += [10*self.image_scale, 10*self.image_scale]
            st = np.random.randint(0, 20*self.image_scale+1, size=(2,), dtype=np.int32)
            image = image[st[1]:st[1]+224*self.image_scale, st[0]:st[0]+224*self.image_scale, :]
            lmk -= st

        left_eye_box = get_rect(lmk[42:47], scale=self.image_scale)
        right_eye_box = get_rect(lmk[36:41], scale=self.image_scale)

        flip = np.random.rand()
        if flip < self.flip:
            image = cv2.flip(image, 1)
            right_eye_box, left_eye_box = flip_rect(left_eye_box, 224*self.image_scale), flip_rect(right_eye_box, 224*self.image_scale)

        image = self.transform(image)
        left_eye_box = torch.tensor(left_eye_box, dtype=torch.float32)
        right_eye_box = torch.tensor(right_eye_box, dtype=torch.float32)

        data = {
            'face': image,
            'left_eye_box': left_eye_box,
            'right_eye_box': right_eye_box,
            # 'cam_ind': cam_ind
        }

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')

            if flip < self.flip:
                gaze_label[-1] = -gaze_label[-1]

            return data, gaze_label
        else:
            return data

    # def itracker_data(self, idx):
    #     key, idsx = self.idx_to_kv[idx]
    #     subj = self.selected_keys[key]
    #
    #     self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
    #     assert self.hdf.swmr_mode
    #
    #     # Get face image
    #     image = self.hdf['face_patch'][idx, :]
    #     image = image[:, :, [2, 1, 0]].copy()  # from BGR to RGB
    #
    #     lmk = self.lmk_hdfs[subj][idsx].copy()
    #
    #
    # def __getitem__(self, idx):


def get_rect(points, ratio=1.0, scale=1):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (max(x) - min(x))
    y_expand = 0.1 * (max(y) - min(y))

    x_max, x_min = max(x) + x_expand, min(x) - x_expand
    y_max, y_min = max(y) + y_expand, min(y) - y_expand

    # h:w=1:2
    if (y_max - y_min) * ratio < (x_max - x_min):
        h = (x_max - x_min) / ratio
        pad = (h - (y_max - y_min)) / 2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max - y_min)
        pad = (h * ratio - (x_max - x_min)) / 2
        x_max += pad
        x_min -= pad

    int(x_min), int(x_max), int(y_min), int(y_max)
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    bbox = np.array(bbox)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (224*scale, 224*scale))
    rect = np.concatenate([aSrc, bSrc])

    return rect


def flip_rect(rect, image_width=224):
    x1, y1, x2, y2 = rect
    y1_flip = y1
    y2_flip = y2
    x1_flip = image_width - x2
    x2_flip = image_width - x1
    rect_flip = np.array([x1_flip, y1_flip, x2_flip, y2_flip], dtype=np.int32)
    return rect_flip
