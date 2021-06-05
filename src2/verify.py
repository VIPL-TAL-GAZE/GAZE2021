import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.io import save_results, load_json
# from eth_xgaze import get_data_loader
from data.xgaze_dataset import get_data_loader
from models.gaze.gazenet import GazeNet
from models.gaze.itracker import ITracker, ITrackerAttention, ITrackerMultiHeadAttention
import numpy as np 

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='GazeX')
        parser.add_argument('--data_dir', type=str, default='/home/data/dataset/ETH_XGaze/xgaze_224',
                            help='dataset dir (default: /workspace/datasets/gaze/xgaze_224)')
        # model params 
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='network model type (default: resnet50)')
        # data loader
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        # cuda, seed
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--checkpoint', type=str, default='/home/zhangjiajun/best_net_resnest448A.pth.tar',
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    # init the args
    args = Options().parse()
    ckpt_dir = os.path.split(args.checkpoint)[0]
    opt_fn = os.path.join(ckpt_dir, 'opt.json')
    opt_dict = load_json(opt_fn)
    for k, v in opt_dict.items():
        if k == 'checkpoint': continue

        if not hasattr(args, k):
            setattr(args, k, v)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init dataloader
    test_loader = get_data_loader(
        args.data_dir, 
        args.batch_size,
        args.image_scale,
        mode='test', 
        num_workers=args.workers, 
        distributed=False)

    if args.model == 'GazeNet':
        model = GazeNet(backbones=args.backbones)
    elif args.model == 'ITracker':
        model = ITracker(pretrained=False)
    elif args.model == 'ITrackerAttention':
        model = ITrackerAttention(pretrained=False)
    elif args.model == 'ITrackerMultiHeadAttention':
        model = ITrackerMultiHeadAttention(pretrained=False)
    else:
        raise NotImplementedError(args.model)
    # print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            if args.no_cuda:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.checkpoint))

    num_test = len(test_loader.dataset)
    pred_gaze_all = torch.zeros((num_test, 2))

    model.eval()
    tbar = tqdm(test_loader, desc='\r')

    
    for batch_idx, (data) in enumerate(tbar):
        data_len = len(data['face'])
        data = {k: v.cuda() if args.cuda else v for k, v in data.items()}
        # for k, v in data.items():
        #     print(k, v.shape)
        # if args.backbone == 'resnet_3s':
        #     data = {k: v.cuda() if args.cuda else v for k, v in data.items()}
        # else:
        #     # data = data['face'].cuda() if args.cuda else data
        #     data = data.cuda() if args.cuda else data

        with torch.no_grad():
            output = model(data)

        st = batch_idx*args.batch_size
        pred_gaze_all[st:st+data_len, :] = output

    if pred_gaze_all.shape[0] != num_test:
        print('the test samples save_index ', pred_gaze_all.shape[0], ' is not equal to the whole test set ', num_test)
    print('Tested on : ', pred_gaze_all.shape[0], ' samples')

    save_results(pred_gaze_all.data.numpy())


if __name__ == "__main__":
    main()

