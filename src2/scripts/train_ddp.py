import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.backends import cudnn
import tqdm

import sys
sys.path.append('../src2')
from options.train_options import Options
from data.xgaze_dataset import get_data_loader
from utils.modules import Trainer, Logger


def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    # args.lr = args.lr * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.rank, args.world_size))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)

    if args.gpu == 0:
        for k, v in sorted(args.__dict__.items()):
            print(k, v)

    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed(args.seed + args.rank)
    cudnn.deterministic = False
    cudnn.benchmark = True

    train_loader = get_data_loader(
        args.data_dir,
        args.batch_size,
        image_scale=args.image_scale,
        split_file=args.split_file,
        mode='train',
        num_workers=args.workers,
        distributed=True,
        debug=args.debug
    )

    if args.eval:
        eval_loader = get_data_loader(
            args.data_dir,
            args.eval_batch_size,
            image_scale=args.image_scale,
            split_file=args.split_file,
            mode='eval',
            num_workers=args.workers,
            distributed=args.eval_dist,
            debug=args.debug
        )
    else:
        eval_loader = None

    if args.gpu == 0:
        logger = Logger(args)
    else:
        logger = None

    
    trainer = Trainer(args)

    epoch_st = 0
    if args.resume:
        if args.gpu == 0:
            logger.set_epoch(args.last_epoch)
        trainer.eval(args.last_epoch, eval_loader, logger)
        epoch_st = args.last_epoch + 1

    for epoch in tqdm.tqdm(range(epoch_st, args.epochs), desc=args.checkname, disable=args.gpu != 0):
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        if args.gpu == 0:
            logger.set_epoch(epoch)

        trainer.train_one_epoch(train_loader, logger)
        trainer.sync_all_devices()
        if args.gpu == 0:
            trainer.save_ckpt(epoch)
        trainer.sync_all_devices()
        trainer.eval(epoch, eval_loader, logger)
        trainer.update_scheduler()


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    main()
