import os
import torch
from torch.backends import cudnn

import sys
sys.path.append('/home/work/projects/gazex')
from options.train_options import Options
from data.xgaze_dataset import get_data_loader
from utils.modules import TrainerSingle, Logger


def main():
    args = Options().parse()

    for k, v in sorted(args.__dict__.items()):
        print(k, v)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    train_loader = get_data_loader(
        args.data_dir,
        args.batch_size,
        mode='train',
        num_workers=args.workers,
        distributed=False,
        debug=args.debug
    )

    if args.eval:
        eval_loader = get_data_loader(
            args.data_dir,
            args.eval_batch_size,
            mode='eval',
            num_workers=args.workers,
            distributed=False,
            debug=args.debug
        )
    else:
        eval_loader = None

    logger = Logger(args)


    # if args.model in ['GazeNet', 'GazeNetCamera']:
    #     trainer = Trainer(args)
    # elif args.model == 'GazeNetRSN':
    #     if args.use_rsn:
    #         trainer = TrainerRSN(args)
    #     else:
    #         trainer = TrainerRSN2(args)
    trainer = TrainerSingle(args)

    for epoch in range(args.epochs):
        # train_loader.batch_sampler.sampler.set_epoch(epoch)
        logger.set_epoch(epoch)

        trainer.train_one_epoch(train_loader, logger)
        trainer.save_ckpt(epoch)

        trainer.eval(epoch, eval_loader, logger)
        trainer.update_scheduler()


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    main()
