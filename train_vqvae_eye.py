import argparse
import json
import torch
from visualize import _show_imgs, _show_loss
import os
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
import config

from torchvision import datasets, transforms, utils
from dataset import _all_image_paths, _split_dataset, list_dataset, H5_EYE_dataset
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

    for i, (img, right_eye_img, left_label, right_label, left_headpose, right_headpose) in enumerate(loader):
        model.zero_grad()
        model.train()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step(loss)
        optimizer.step()

        all_loss = recon_loss + latent_loss_weight * latent_loss
        mse_sum += all_loss * img.shape[0]
        mse_n += img.shape[0]
        total_loss = mse_sum /mse_n

        lr = optimizer.param_groups[0]['lr']


        if iteration % 50 == 0:
            if valid_loader is None:
                sample = img[:sample_size]
                model.eval()
                with torch.no_grad():
                    val_out, val_latent_loss = model(sample)
                    best_stats_dict['val_recon_loss']= criterion(val_out, sample)
                    best_stats_dict['val_latent_loss']= val_latent_loss.mean()
                    best_stats_dict['val_loss'] = best_stats_dict['val_recon_loss'] + latent_loss_weight * best_stats_dict['val_latent_loss']
                    save_out = torch.cat([sample, val_out], 0)
                if best_stats_dict['val_loss'] < best_stats_dict['best_loss']:
                    torch.save(
                        model.module.state_dict(), f'{model_save_path}/vqvae_eye_best.pt'
                    )
                if iteration > best_stats_dict['best_iter']:
                    torch.save(
                        model.module.state_dict(), f'{model_save_path}/vqvae_eye_last.pt'
                    )
                    Stop_flag = True
                    return 
            else:
                save_out, best_stats_dict['val_recon_loss'], best_stats_dict['val_latent_loss'], best_stats_dict['val_loss'] = test(valid_loader, model, device)
                if best_stats_dict['val_loss'] < min_loss:
                    patience = 0
                    min_loss = best_stats_dict['val_loss']
                    torch.save(
                        model.module.state_dict(), f'{model_save_path}/vqvae_best.pt')
                    best_stats_dict['best_loss'] = min_loss
                    best_stats_dict['best_iter'] = iteration
                else:
                    patience += 1
            if vis:
                _show_imgs(vis, save_out.cpu(), win_name = "iter", n_row=sample_size)
                _show_loss(vis, iteration, val_recon_loss, val_latent_loss, val_loss)
            if patience > max_patience:
                Stop_flag = True
                return

            if iteration % 1000 == 0:
                utils.save_image(
                    save_out,
                    f'sample/{str(epoch + 1).zfill(5)}_{str(iteration).zfill(5)}.png',
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
            model.train()
        loader.set_description(
            (
                f"epoch:{epoch + 1}; mse:({recon_loss.item():.4f} {best_stats_dict['val_recon_loss'].item():.4f});"
                f"latent:({latent_loss.item():.4f} {best_stats_dict['val_latent_loss'].item():.4f});all:({total_loss.item():.4f} {best_stats_dict['val_loss'].item():.4f});"
                f"best:({best_stats_dict['best_iter']} {best_stats_dict['best_loss']:.4f})"
                f"lr: {lr:.5f};"
                f"patience: {patience};"
            )
        )
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
    
    for i, (img, right_eye_img, left_label, right_label, left_headpose, right_headpose) in enumerate(loader):

        img = img.to(device)
        n_dataset += img.shape[0]

        with torch.no_grad():
            out, latent_loss = model(img)
            if i == 1:
                sample = img[:sample_size]
                recon_sample = out[:sample_size]
            sum_recon_loss += criterion(out, img) * img.shape[0]
            sum_latent_loss += latent_loss.mean() * img.shape[0]
    recon_loss = sum_recon_loss / n_dataset
    latent_loss = sum_latent_loss / n_dataset
    # loss = recon_loss + latent_loss_weight * latent_loss
    loss = recon_loss + latent_loss_weight * latent_loss
    out = torch.cat([sample, recon_sample], 0)
    return out, recon_loss, latent_loss, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config_eye.json")
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

    # transform = transforms.Compose(
    #     [
    #         # transforms.Resize(config.scale_size),
    #         # transforms.CenterCrop(config.scale_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    path = config.img_path
    iteration = 0
    min_loss = 999999
    Stop_flag = False 
    best_stats_dict = dict(best_loss=0, best_iter=0)
    if args.resize:
        path = config.after_scale_cache_path
    model_save_path = os.path.join(f"/disks/disk2/fjl/checkpoint/vq-vae_{args.beta}/")
    os.makedirs(model_save_path, exist_ok=True)
    # paths = _all_image_paths(path)
    paths = glob.glob(f"{path}/*.h5")
    train_paths, valid_paths = _split_dataset(paths, config.test_rate)
    train_data = H5_EYE_dataset(train_paths)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.n_loaders_train)
    valid_loader = None
    if valid_paths:
        valid_data = H5_EYE_dataset(valid_paths)# , transform)
        valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=config.n_loaders_test)
    else:
        config.load('./vq-vae-2_modified_moveAve.json')
        best_stats_dict['best_loss'] = config.best_loss
        best_stats_dict['best_iter'] = config.best_iter
        print(f'best_loss: {str(config.best_loss)}; best_iter: {str(config.best_iter)}')
    patience = 0
    max_patience = len(train_loader) * 10

    model = nn.DataParallel(VQVAE(in_channel=1)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, config.lr, n_iter=len(train_loader) * config.n_epoch, momentum=None
        )
    elif config.sched == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(config.n_epoch):
        if config.sched == 'plateau':
            print(scheduler.num_bad_epochs)
            loss = train(i, train_loader,valid_loader, model, optimizer, None, device, best_stats_dict, vis)
            scheduler.step(best_stats_dict["val_loss"])
        else:
            loss = train(i, train_loader,valid_loader, model, optimizer, scheduler, device, best_stats_dict, vis)
        if Stop_flag:
            if valid_loader:
                stats_dict = {}
                with open(f"vq-vae-2_{args.beta}.json", 'w') as f:
                    stats_dict['best_loss'] = float(best_stats_dict['best_loss'])
                    stats_dict['best_iter'] = int(best_stats_dict['best_iter'])
                    stats_dict['sched'] = config.sched
                    json_data = json.dumps(stats_dict, sort_keys=True, indent=4, separators=(',', ': '))
                    f.write(json_data)
                    f.close()
            break
