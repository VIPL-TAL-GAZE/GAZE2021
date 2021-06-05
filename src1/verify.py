import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import save_results
from eth_xgaze import get_data_loader
# from models.gaze.gazenet import GazeNet
from models.gaze.gazenet import get_model


from utils import AverageMeter, angular_error




class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='GazeX')
        parser.add_argument('--data-dir', type=str, default='/workspace/datasets/gaze/xgaze_224',
                            help='dataset dir (default: /workspace/datasets/gaze/xgaze_224)')
        # model params 
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='network model type (default: resnet50)')
        # data loader
        parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--splits', type=str, default='train_eval_test_split.json',
                            help='split json')
        parser.add_argument('--input-size', type=int, default=224, help='train input size')
        parser.add_argument('--mode', type=str, default='test', help='infer mode')

        # cuda, seed
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--checkpoint', type=str, default='runs/resnet50/baseline/epoch_24.pth.tar',
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init dataloader
    test_loader = get_data_loader(
        args.data_dir, 
        args.batch_size, 
        mode=args.mode, 
        splits=args.splits,
        input_size=args.input_size,
        is_flip=False,
        num_workers=args.workers, 
        distributed=False)

    # model = GazeNet(backbone=args.backbone)
    model = get_model(name=args.backbone)
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
    index = 0

    model.eval()
    tbar = tqdm(test_loader, desc='\r')

    if args.mode == 'test':
        for batch_idx, data in enumerate(tbar):
            if args.cuda:
                data = data.cuda()
            with torch.no_grad():
                output = model(data)
                pred_gaze_all[index:index + data.size(0), :] = output
                index += data.size(0)

    elif args.mode == 'eval':
        errors = AverageMeter()
        for batch_idx, (data, target) in enumerate(tbar):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)

                gaze_error = np.mean(angular_error(output.cpu().data.numpy(), target.cpu().data.numpy()))
                errors.update(gaze_error.item(), data.size(0))
                pred_gaze_all[index:index + data.size(0), :] = output
                index += data.size(0)
        print('Eval error: {}'.format(errors.avg))
    else:
        raise NotImplementedError

    save_results(pred_gaze_all.data.numpy())


if __name__ == "__main__":
    main()

