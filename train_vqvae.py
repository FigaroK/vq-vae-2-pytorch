import argparse

import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
import config

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--resize', type=bool)
    parser.add_argument('--visual', type=bool, default=False)

    args = parser.parse_args()
    if args.visual:
        from visdom import Visdom
        vis = Visdom(env="vq-vae-2")
    config.load(args.config)

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(config.scale_size),
            transforms.CenterCrop(config.scale_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    path = config.ffhq_img_path
    if args.resize:
        path = config.after_scale_cache_path
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, config.lr, n_iter=len(loader) * config.n_epoch, momentum=None
        )

    for i in range(config.n_epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
