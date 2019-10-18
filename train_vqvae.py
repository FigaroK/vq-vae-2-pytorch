import argparse

import torch
from visualize import _show_imgs, _show_loss
import os
from torch import nn, optim
from torch.utils.data import DataLoader
import config

from torchvision import datasets, transforms, utils
from dataset import _all_image_paths, _split_dataset, list_dataset
from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler


def train(epoch, train_loader, valid_loader, model, optimizer, scheduler, device, best_stats_dict, vis=None):
    global iteration
    global Stop_flag
    global patience
    global min_loss
    global max_patience
    loader = tqdm(train_loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 20

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        model.train()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += (recon_loss.item() + latent_loss.item()) * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if iteration % 50 == 0:
            out, recon_loss, latent_loss, loss = test(valid_loader, model, device)
            if loss < min_loss:
                patience = 0
                min_loss = loss
                torch.save(
                    model.module.state_dict(), f'/disks/disk0/fjl/ffhq/checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt'
                )
                best_stats_dict['best_loss'] = loss
                best_stats_dict['best_iter'] = iteration
            else:
                patience += 1
            if vis:
                _show_imgs(vis, out.cpu(), win_name = "iter", n_row=sample_size)
                _show_loss(vis, iteration, recon_loss, latent_loss, loss)
            if patience > max_patience:
                Stop_flag = True
                break

            # utils.save_image(
            #     out,
            #     f'sample/{str(epoch + 1).zfill(5)}_{str(iteration).zfill(5)}.png',
            #     nrow=sample_size,
            #     normalize=True,
            #     range=(-1, 1),
            # )
            model.train()
        iteration += 1

def test(loader, model, device):

    criterion = nn.MSELoss(reduction='mean')
    latent_loss_weight = 0.25

    sample_size = 20

    mse_sum = 0
    mse_n = 0
    model.zero_grad()
    model.eval()
    sum_latent_loss = 0
    sum_recon_loss = 0
    n_dataset = 0
    
    for i, (img, label) in enumerate(loader):

        img = img.to(device)
        n_dataset += img.shape[0]
        if i == 1:
            sample = img[:sample_size]

        with torch.no_grad():
            out, latent_loss = model(img)
            if i == 1:
                recon_sample = out[:sample_size]
            sum_recon_loss += criterion(out, img)
            sum_latent_loss += latent_loss.sum()
    recon_loss = sum_recon_loss / n_dataset
    latent_loss = sum_latent_loss / n_dataset
    loss = recon_loss + latent_loss_weight * latent_loss
    out = torch.cat([sample, recon_sample], 0)
    return out, recon_loss, latent_loss, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config.json")
    parser.add_argument('--resize', type=bool)
    parser.add_argument('--visual', type=bool, default=False)
    parser.add_argument('--beta', type=str)

    args = parser.parse_args()
    vis = None
    if args.visual:
        from visdom import Visdom
        vis = Visdom(env=f"vq-vae-2_{args.beta}")
    config.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            # transforms.Resize(config.scale_size),
            # transforms.CenterCrop(config.scale_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    path = config.ffhq_img_path
    iteration = 0
    min_loss = 999999
    Stop_flag = False 
    best_stats_dict = dict(best_loss=0, best_iter=0)
    if args.resize:
        path = config.after_scale_cache_path
    paths = _all_image_paths(path)
    train_paths, valid_paths = _split_dataset(paths)
    train_data = list_dataset(train_paths, transform)
    valid_data = list_dataset(valid_paths, transform)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=5)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=5)
    max_patience = len(train_loader) / 50
    patience = 0

    model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, config.lr, n_iter=len(train_loader) * config.n_epoch, momentum=None
        )

    for i in range(config.n_epoch):
        loss = train(i, train_loader,valid_loader, model, optimizer, scheduler, device, best_stats_dict, vis)
