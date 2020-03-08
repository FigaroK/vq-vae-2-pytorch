import argparse
import json
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
                f'lr: {lr:.5f};'
                # f'patience: {patience};'
            )
        )

        if iteration % 50 == 0:
            if valid_loader is None:
                sample = img[:sample_size]
                model.eval()
                with torch.no_grad():
                    out, latent_loss = model(sample)
                    recon_loss = criterion(out, sample)
                    latent_loss = latent_loss.mean()
                    loss = recon_loss + latent_loss_weight * latent_loss
                    out = torch.cat([sample, out], 0)
                torch.save(
                    model.module.state_dict(), f'{model_save_path}/vqvae_best.pt'
                )
                if iteration > best_stats_dict['best_iter']:
                    Stop_flag = True
                    return 
            else:
                out, recon_loss, latent_loss, loss = test(valid_loader, model, device)
                if loss < min_loss:
                    patience = 0
                    min_loss = loss
                    torch.save(
                        model.module.state_dict(), f'{model_save_path}/vqvae_best.pt')
                    best_stats_dict['best_loss'] = loss
                    best_stats_dict['best_iter'] = iteration
                else:
                    patience += 1
            if vis:
                _show_imgs(vis, out.cpu(), win_name = "iter", n_row=sample_size)
                _show_loss(vis, iteration, recon_loss, latent_loss, loss)
            if patience > max_patience:
                Stop_flag = True
                return

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
            sum_recon_loss += criterion(out, img) * img.shape[0]
            sum_latent_loss += latent_loss.mean() * img.shape[0]
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
    model_save_path = os.path.join(f'/disks/disk2/fjl/checkpoint/vq-vae/')
    os.makedirs(model_save_path, exist_ok=True)
    paths = _all_image_paths(path)
    train_paths, valid_paths = _split_dataset(paths, config.test_rate)
    train_data = list_dataset(train_paths, transform)
    valid_loader = None
    if valid_paths:
        valid_data = list_dataset(valid_paths, transform)
        valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=5)
    else:
        config.load('./vq-vae-2_modified_moveAve.json')
        best_stats_dict['best_loss'] = config.best_loss
        best_stats_dict['best_iter'] = config.best_iter
        print(f'best_loss: {str(config.best_loss)}; best_iter: {str(config.best_iter)}')
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=5)
    max_patience = len(train_loader) / 50 * 10
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
        if Stop_flag:
            if valid_loader:
                with open(f"vq-vae-2_{args.beta}", 'w') as f:
                    best_stats_dict['best_loss'] = float(best_stats_dict['best_loss'].cpu().numpy())
                    best_stats_dict['best_iter'] = int(best_stats_dict['best_iter'].cpu().numpy())
                    json_data = json.dumps(best_stats_dict, sort_keys=True, indent=4, separators=(',', ': '))
                    f.write(json_data)
                    f.close()
            break
