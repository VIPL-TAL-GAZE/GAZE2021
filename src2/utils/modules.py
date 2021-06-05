import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import apex
from apex import amp
import tqdm
# import tensorboardX
from torch.utils.tensorboard import SummaryWriter

from models.gaze.gazenet import GazeNet
from models.gaze.itracker import ITracker, ITrackerAttention, ITrackerMultiHeadAttention
from utils.losses import get_loss, get_rsn_loss, AngularLoss, get_ohem_loss
from utils.metrics import AverageMeterTensor, angular_error
from utils.io import save_checkpoint
from utils.drawing import draw_bbox


def get_optimizer(optim_type, parameters, lr, weight_decay):
    if optim_type == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)
    elif optim_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, amsgrad=False)
    elif optim_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'momentum':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_type == 'nesterov':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError


def get_scheduler(scheduler_type, optimizer, step_size, gamma, last_epoch):
    if scheduler_type == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    if scheduler_type == 'mslr':
        assert isinstance(step_size, list)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma, last_epoch=last_epoch)
    if scheduler_type == 'cosinelr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, last_epoch=last_epoch)
    if scheduler_type == 'cosineawr':
        assert isinstance(step_size, list)
        T0, Tm = step_size
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tm, last_epoch=last_epoch)
    if scheduler_type == 'explr':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    else:
        raise NotImplementedError


def train_one_step(model, data):
    output = model(data)
    return output


def get_model(args):
    if args.model == 'GazeNet':
        return GazeNet(args.backbones, pretrained=args.pretrained, dropout=args.dropout)
    elif args.model == 'ITracker':
        return ITracker(pretrained=args.pretrained)
    elif args.model == 'ITrackerAttention':
        return ITrackerAttention(pretrained=args.pretrained)
    elif args.model == 'ITrackerMultiHeadAttention':
        return ITrackerMultiHeadAttention(pretrained=True)
    else:
        raise NotImplementedError


class Trainer:
    def __init__(self, args):
        self.gpu = args.gpu
        self.save_dir = os.path.join(args.ckpt_root, args.checkname)
        self.amp = args.amp

        self.model = get_model(args)

        if self.gpu == 0:
            print(self.model)

        self.model.cuda(self.gpu)

        self.criterion = get_loss(args.loss).cuda(self.gpu)
        self.optimizer = get_optimizer(args.optim, self.model.parameters(), args.lr, args.weight_decay)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer, args.step_size, args.gamma, -1)

        self.total_step = 0
        self.best_epoch = {
            'epoch': 0,
            'error': float('inf'),
            'loss': float('inf')
        }

        if args.resume:
            # assert args.last_epoch > -1, 'Please set an available last-epoch, not {}.'.format(args.last_epoch)
            ckpt_name = 'epoch_' + str(args.last_epoch) + '.pth.tar'
            ckpt_fn = os.path.join(args.ckpt_root, args.checkname, ckpt_name)
            assert os.path.exists(ckpt_fn), 'Checkpoint {} is not exists!'.format(ckpt_fn)

            ckpt = torch.load(ckpt_fn, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.total_step = ckpt['total_step']

            print('Load checkpoint from', ckpt_fn)

        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='02')
            self.model = apex.parallel.convert_syncbn_model(self.model)
            DDP = apex.parallel.DistributedDataParallel
            self.model = DDP(self.model, delay_allreduce=True)
            self.sync_all_devices = lambda: None
        else:
            DDP = DistributedDataParallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=False)
            self.sync_all_devices = dist.barrier

        self.eval_dist = args.eval_dist
        self.ohem = args.use_ohem
        self.ohem_keen_num = 0
        if args.use_ohem:
            self.ohem_keen_num = int(args.ohem_frac * args.batch_size)

    def train_one_epoch(self, data_loader, logger=None):
        self.model.train()

        loss_avger = AverageMeterTensor().cuda(self.gpu)
        error_avger = AverageMeterTensor().cuda(self.gpu)

        for data, target in tqdm.tqdm(data_loader, desc='Train {}'.format(self.gpu), disable=(self.gpu != 0)):
            data = {k: v.cuda(self.gpu) for k, v in data.items()}
            target = target.cuda(self.gpu)
            output = self.model(data)
            self.total_step += 1

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), target.cpu().data.numpy()))
            error_avger.update(gaze_error.item(), len(data_loader))

            loss = self.criterion(output, target)
            if self.ohem:
                ohem_loss, ohem_idx = get_ohem_loss(output, target, keep_num=self.ohem_keen_num)
                loss += ohem_loss

            # use flood
            # loss = torch.abs(loss - 0.005) + 0.005

            self.optimizer.zero_grad()

            if self.amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            loss_avger.update(loss.item(), len(data_loader))

            if logger is not None:
                log_dict = {
                    'gaze_loss': loss_avger.avg,
                    'error': error_avger.avg,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                logger.log_train(log_dict, self.total_step)

                bboxes = torch.cat([data['left_eye_box'][0], data['right_eye_box'][0]], dim=0)
                logger.add_img1(
                    data['face'][0].detach().cpu().numpy(),
                    bboxes.detach().cpu().numpy(),
                    'train', self.total_step
                )

    def save_ckpt(self, epoch, file_name=None):
        state_dict = {
            'epoch': epoch,
            'total_step': self.total_step,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        if self.amp:
            state_dict['amp'] = amp.state_dict()

        if file_name is None:
            file_name = 'epoch_{}.pth.tar'.format(epoch)
        save_checkpoint(state_dict, self.save_dir, file_name)

        print('Save checkpoint in {}'.format(os.path.join(self.save_dir, file_name)))

    def eval(self, epoch, data_loader, logger=None):
        if data_loader is None:
            return

        self.model.eval()

        loss_avg = AverageMeterTensor().cuda(self.gpu)
        error_avg = AverageMeterTensor().cuda(self.gpu)

        for data, target in tqdm.tqdm(data_loader, desc='Eval {}'.format(self.gpu), disable=self.gpu != 0):
            data = {k: v.cuda(self.gpu) for k, v in data.items()}
            target = target.cuda(self.gpu)
            with torch.no_grad():
                output = self.model(data)

            gaze_error = np.mean(angular_error(output.cpu().data.numpy(), target.cpu().data.numpy()))
            error_avg.update(gaze_error.item(), len(data_loader))

            loss = self.criterion(output, target)
            loss_avg.update(loss.item(), len(data_loader))

        if self.eval_dist:
            # sum all evaluated loss and error from different devices
            loss_and_error = torch.tensor([loss_avg.sum.clone(), error_avg.sum.clone(), loss_avg.count.clone()],
                                          dtype=torch.float64, device=self.gpu)
            self.sync_all_devices()
            dist.all_reduce(loss_and_error, dist.ReduceOp.SUM)
            loss_sum, error_sum, count_sum = loss_and_error.tolist()

            loss_avg = loss_sum / count_sum
            error_avg = error_sum / count_sum
        else:
            loss_avg = loss_avg.avg.item()
            error_avg = error_avg.avg.item()

        self.sync_all_devices()
        if logger is not None:
            log_dict = {'gaze_loss': loss_avg, 'error': error_avg}
            logger.log_eval(log_dict)

            if error_avg < self.best_epoch['error']:
                self.best_epoch['epoch'] = epoch
                self.best_epoch['error'] = error_avg
                self.best_epoch['loss'] = loss_avg
                self.save_ckpt(epoch, 'best_net.pth.tar')
            print(str(self.best_epoch))
        self.sync_all_devices()

    def update_scheduler(self):
        self.scheduler.step()


class Logger:
    def __init__(self, args):
        self.log_dir = os.path.join(args.ckpt_root, args.checkname, 'log')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.file_writer = SummaryWriter(self.log_dir, flush_secs=10)
        self.fresh_per_iter = args.fresh_per_iter

        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def log_train(self, log_dict, total_step):
        if total_step % self.fresh_per_iter != 0:
            return

        log_info = '[INFO] Train epoch {} '.format(self._epoch)

        for k, v in sorted(log_dict.items(), key=lambda x: x[0]):
            log_info += '{}: {:.04f} '.format(k, v)
            self.file_writer.add_scalar('gazex/train/{}'.format(k), v, total_step)

        tqdm.tqdm.write(log_info)

    def log_eval(self, log_dict):
        log_info = '[INFO] Eval epoch {} '.format(self._epoch)

        for k, v in sorted(log_dict.items(), key=lambda x: x[0]):
            log_info += '{}: {:.04f} '.format(k, v)
            self.file_writer.add_scalar('gazex/test/{}'.format(k), v, self._epoch)

        print(log_info)

    def add_img(self, img, bbox_s, bbox_r, phase, step):
        if step % self.fresh_per_iter != 0:
            return

        img = ((img.transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        img_s = img.copy()
        img_r = img.copy()
        for bbs, bbr in zip(bbox_s.reshape(-1, 4), bbox_r.reshape(-1, 4)):
            img_s = draw_bbox(img_s, bbs, color='green')
            img_r = draw_bbox(img_r, bbr, color='red')
        # img_s = img_s[np.newaxis, ...]
        # img_r = img_r[np.newaxis, ...]

        self.file_writer.add_image(
            tag='gazex/{}/select'.format(phase),
            img_tensor=img_s,
            global_step=step,
            dataformats='HWC'
        )

        self.file_writer.add_image(
            tag='gazex/{}/rand'.format(phase),
            img_tensor=img_r,
            global_step=step,
            dataformats='HWC'
        )

    def add_img1(self, img, bbox, phase, step):
        if step % self.fresh_per_iter != 0:
            return

        img = ((img.transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        img = img.copy()
        for bb in bbox.reshape(-1, 4):
            img = draw_bbox(img, bb, color='green')
        # img_s = img_s[np.newaxis, ...]
        # img_r = img_r[np.newaxis, ...]

        self.file_writer.add_image(
            tag='gazex/{}/select'.format(phase),
            img_tensor=img,
            global_step=step,
            dataformats='HWC'
        )


