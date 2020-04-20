import argparse

import numpy as np
import config
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label, headpose) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        cond = torch.cat([label, headpose], 1)
        if args.hier == 'top':
            target = top
            out, _ = model(top, cond_headpose_gaze=cond)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top, cond_headpose_gaze=cond)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.7f}'
            )
        )
    return loss


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('--config', type=str, default="./config_eye.json")

    args = parser.parse_args()

    config.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    if args.hier == 'top':
        model = PixelSNAIL(
            [9, 15],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=1,
            cond_headpose_gaze=True
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [18, 30],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            condition=True,
            cond_headpose_gaze=True,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)
    save_path = '/disks/disk2/fjl/checkpoint/vq-vae/pixelsnail/unity_mpii_condition_all'
    os.makedirs(save_path, exist_ok=True)
    model = nn.DataParallel(model).to(device)
    scheduler = None
    if config.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, config.lr, n_iter=len(loader) * config.n_epoch, momentum=None
        )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(config.n_epoch):
        loss = train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'{save_path}/pixelsnail_{args.hier}_eye_{str(i + 1).zfill(3)}.pt',
        )
