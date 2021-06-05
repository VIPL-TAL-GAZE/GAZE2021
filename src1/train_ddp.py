import os
import time
import argparse
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter

from eth_xgaze import get_data_loader
# from models.gaze.gazenet import GazeNet
from models.gaze.gazenet import get_model
from losses import get_loss
from utils import AverageMeter, save_checkpoint, angular_error, AverageMeterTensor
from utils import save_json


from lr_policy import GradualWarmupScheduler


try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    print('please install amp if using float16 training')


class Options():
    def __init__(self):

        # data settings
        parser = argparse.ArgumentParser(description='Gaze')
        parser.add_argument('--data-dir', type=str, default='/workspace/datasets/gaze/xgaze_224',
                            help='dataset dir (default: /workspace/datasets/gaze/xgaze_224)')

        # model params 
        parser.add_argument('--backbone', type=str, default='hrnet_w64',
                            help='network model type (default: hrnet_w64)')
        parser.add_argument('--pretrained', action='store_true', 
                            default=True, help='load pretrianed mode')

        # training params
        parser.add_argument('--amp', action='store_true',
                            default=False, help='using amp')
        parser.add_argument('--batch-size', type=int, default=192, metavar='N',
                            help='batch size for training (default: 192)')
        parser.add_argument('--epochs', type=int, default=25, metavar='N',
                            help='number of epochs to train (default: 25)')
        parser.add_argument('--start-epoch', type=int, default=0, 
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--resume', action='store_true', default=False,
                            help='Resume training from last saved checkpoint.')
        parser.add_argument('--last-epoch', type=int, default=-1,
                            help='Resume training from last epoch checkpoint.')


        parser.add_argument('--splits', type=str, default='train_eval_test_split.json',
                            help='split json')
        parser.add_argument('--input-size', type=int, default=224, help='train input size')
        parser.add_argument('--is-flip', action='store_true', default=False, help='flip data aug')


        # eval params
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--eval-batch-size', type=int, default=16, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--eval-dist', action='store_true', default=False)

        # loss
        parser.add_argument('--loss', type=str, default='l1',
                            help='loss type (default: l1)')

        # optimizer
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='optimizer (default: sgd)')
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--min-lr', type=float, default=5e-6,
                            help='min learning rate (default: 5e-6)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='sgd momentum (default: 0.9)') 
        parser.add_argument('--weight-decay', type=float, default=1e-4, 
                            metavar ='M', help='SGD weight decay (default: 1e-4)')

        # scheduler
        parser.add_argument('--scheduler', type=str, default='cosine',
                            help='scheduler (default: cosine)')
        parser.add_argument('--step-size', type=int, default=10,
                            help='step size (default: 10)')
        parser.add_argument('--lr-decay-factor', type=float, default=0.1, 
                            help='lr decay factor (default: 0.1)')
        parser.add_argument('--use-warmup', action='store_true', default=False,
                            help='use warmup or not (default: False)')
        parser.add_argument('--warmup-epochs', type=int, default=5, 
                            help='warmup epochs (default: 5)')

        # seed
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

        # checking point
        parser.add_argument('--ckpt-root', type=str, default='runs',
                            help='Root of checkpoints folder.')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')

        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')

        # log
        parser.add_argument('--use-tensorboard', action='store_true', default=False)
        parser.add_argument('--fresh-per-iter', type=int, default=100)

        # debug
        parser.add_argument('--debug', action='store_true', default=False)

        self.parser = parser

    def parse(self, save_configs=True):
        args = self.parser.parse_args()
        if save_configs and args.rank == 0:
            log_dir = os.path.join(args.ckpt_root, args.backbone, args.checkname)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            config_fn = os.path.join(log_dir, 'opt.json')
            save_json(config_fn, args.__dict__)
        return args

def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    get_model(name=args.backbone, pretrained=args.pretrained)
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
        print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True


    train_loader = get_data_loader(
        args.data_dir, 
        args.batch_size, 
        mode='train', 
        splits=args.splits,
        input_size=args.input_size,
        is_flip=args.is_flip,
        num_workers=args.workers, 
        distributed=True,
        debug=args.debug
    )

    if args.eval:
        eval_loader = get_data_loader(
            args.data_dir,
            args.eval_batch_size,
            mode='eval',
            splits=args.splits,
            input_size=args.input_size,
            is_flip=False,
            num_workers=args.workers,
            distributed=args.eval_dist,
            debug=args.debug
        )
    else:
        eval_loader = None


    # model = GazeNet(backbone=args.backbone, pretrained=args.pretrained)
    model = get_model(name=args.backbone, pretrained=args.pretrained)
    if args.gpu == 0:
        print(model)

    criterion = get_loss(args.loss)

    model.cuda(args.gpu)
    criterion.cuda(args.gpu)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=args.weight_decay, 
            amsgrad=False)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=args.weight_decay, 
            amsgrad=False)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    else:
        raise ValueError


    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.lr_decay_factor, 
            last_epoch=-1,
            verbose=False)
    elif args.scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.95)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs-args.warmup_epochs if args.use_warmup else args.epochs, 
            eta_min=args.min_lr, 
            last_epoch=-1, 
            verbose=False)
    else:
        raise ValueError


    if args.use_warmup:
        scheduler_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=args.warmup_epochs,
            after_scheduler=scheduler)


    global total_step
    total_step = 0
    best_epoch = {
        'epoch': 0,
        'error': float('inf'),
        'loss': float('inf'),
    }

    if args.resume:
        assert args.last_epoch > -1, 'Please set an available last-epoch, not {}.'.format(args.last_epoch)
        ckpt_name = 'epoch_' + str(args.last_epoch) + '.pth.tar'
        ckpt_fn = os.path.join(args.ckpt_root, args.backbone, args.checkname, ckpt_name)

        assert os.path.exists(ckpt_fn), 'Checkpoint {} is not exists!'.format(ckpt_fn)

        ckpt = torch.load(ckpt_fn)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        args.start_epoch = ckpt['epoch'] + 1
        total_step = len(train_loader) * ckpt['epoch']
        print('Load checkpoint from', ckpt_fn)

    if args.amp:
        #optimizer = amp_handle.wrap_optimizer(optimizer)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        model = apex.parallel.convert_syncbn_model(model)
        #from apex import amp
        DDP = apex.parallel.DistributedDataParallel
        model = DDP(model, delay_allreduce=True)
        sync_all_devices = lambda: None
    else:
        DDP = DistributedDataParallel
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu])
        # model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        sync_all_devices = dist.barrier

    if args.use_tensorboard and args.gpu == 0:
        log_dir = os.path.join(args.ckpt_root, args.backbone, args.checkname, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print('Save tensorboard log in', log_dir)
        tb = SummaryWriter(log_dir, flush_secs=10)

    def eval(epoch):
        model.eval()

        losses = AverageMeterTensor().cuda(args.gpu)
        errors = AverageMeterTensor().cuda(args.gpu)

        for data, target in tqdm.tqdm(eval_loader, desc='eval'):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            with torch.no_grad():
                output = model(data)

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), target.cpu().data.numpy()))
            errors.update(gaze_error.item(), data.size(0))

            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

        if args.eval_dist:
            # sum all evaluated loss and error from different devices
            loss_and_error = torch.tensor([losses.sum.clone(), errors.sum.clone(), losses.count.clone()],
                                          dtype=torch.float64, device=args.gpu)
            sync_all_devices()
            dist.all_reduce(loss_and_error, dist.ReduceOp.SUM)
            loss_sum, error_sum, count_sum = loss_and_error.tolist()

            loss_avg = loss_sum / count_sum
            error_avg = error_sum / count_sum
        else:
            loss_avg = losses.avg.item()
            error_avg = errors.avg.item()

        if args.gpu == 0:
            print('Epoch: {} / {}, Eval error: {:.5f}, Eval gaze loss: {:.5f}'.format(epoch, args.epochs, error_avg, loss_avg))
            if args.use_tensorboard:
                tb.add_scalar('eval/gaze_loss', loss_avg, epoch)
                tb.add_scalar('eval/error', error_avg, epoch)

            if error_avg < best_epoch['error']:
                best_epoch['epoch'] = epoch
                best_epoch['error'] = error_avg
                best_epoch['loss'] = loss_avg

                state_dict = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                if args.amp:
                    state_dict['amp'] = amp.state_dict()

                filename = 'best_net.pth.tar'
                save_checkpoint(state_dict, args=args, filename=filename)
            print(best_epoch)
        model.train()

    def train(epoch):
        global total_step

        model.train()
        losses = AverageMeter()
        errors = AverageMeter()

        tic = time.time()
        train_loader.batch_sampler.sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            output = model(data)
            total_step += 1

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), target.cpu().data.numpy()))
            errors.update(gaze_error.item(), data.size(0))

            loss = criterion(output, target.float())
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            losses.update(loss.item(), data.size(0))

            # print(total_step)
            if total_step % args.fresh_per_iter == 0:
                # sync_all_devices()
                if args.gpu == 0:
                    iter_per_sec = args.fresh_per_iter / (time.time() - tic) if batch_idx != 0 else 1.0 / (time.time() - tic)
                    tic = time.time()

                    print('Epoch: {} / {}, Iter: {} / {}, Speed: {:.3f} iter/sec, lr: {:.5f}, Train error: {:.5f}, Gaze loss: {:.5f}'. \
                            format(epoch, args.epochs, batch_idx+1, len(train_loader), iter_per_sec, optimizer.param_groups[0]['lr'], errors.avg, losses.avg))

                    if args.use_tensorboard:
                        # total_step = epoch * len(train_loader) + batch_idx
                        tb.add_scalar('train/gaze_loss', losses.avg, total_step)
                        tb.add_scalar('train/error', errors.avg, total_step)
                        tb.add_scalar('train/time', iter_per_sec, total_step)
                        tb.add_scalar('train/lr', optimizer.param_groups[0]['lr'], total_step)
                sync_all_devices()

    def save_checkpoints(epoch):
        state_dict = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }
        if args.amp:
            state_dict['amp'] = amp.state_dict()

        filename = 'epoch_' + str(epoch) + '.pth.tar'
        save_checkpoint(state_dict, args=args, filename=filename)

    for epoch in range(args.start_epoch, args.epochs):
        # sync_all_devices()
        tic = time.time()
        train(epoch)

        if args.eval:
            if args.eval_dist:
                eval(epoch)
            else:
                # sync_all_devices()
                if args.gpu == 0:
                    eval(epoch)
                sync_all_devices()

        # sync_all_devices()
        if args.gpu == 0:
            save_checkpoints(epoch)
        sync_all_devices()

        if args.use_warmup:
            scheduler_warmup.step()
        else:
            scheduler.step()
        elapsed = time.time() - tic

        # sync_all_devices()
        # if args.gpu == 0:
        #     print(f"Epoch: {epoch}, Time cost: {elapsed}")
        # sync_all_devices()

    if args.gpu == 0:
        best_epoch_fn = os.path.join(args.ckpt_root, args.backbone, args.checkname, 'best_epoch_info.json')
        save_json(best_epoch_fn, best_epoch)


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    main()

