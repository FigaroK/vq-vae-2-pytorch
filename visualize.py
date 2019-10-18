#!/usr/bin/env python3

import cv2 as cv
import os
from visdom import Visdom
import numpy as np
import time
import torch

def _show_imgs(vis, imgs, win_name, n_row):
    imgs = (imgs * 0.5 + 0.5) * 255
    imgs = (torch.clamp(imgs, 0, 255)).to(dtype=torch.uint8)
    opts = dict(caption="ffhq")
    vis.images(imgs, opts=opts, nrow=n_row, padding=3, win=win_name)

def _load_img(path, mode=cv.IMREAD_COLOR):
    img = cv.imread(path, mode)
    img = cv.resize(img, (256, 256))
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    return img

def _show_loss(vis, x_iter, recon_loss, latent_loss, loss):
    x_iter = np.asarray([x_iter])
    vis.line(X=x_iter, Y=np.asarray([recon_loss.cpu().detach()]), update='append', win='recon_loss', opts=dict(markersymbol='circle', title="recon_loss", legend=["recon_loss_l1"]))
    vis.line(X=x_iter, Y=np.asarray([latent_loss.cpu().detach()]), update='append', win='latent_loss', opts=dict(markersymbol='circle', title="latent_loss", legend=["latent_loss_l2"]))
    vis.line(X=x_iter, Y=np.asarray([loss.cpu().detach()]), update='append', win='loss', opts=dict(markersymbol='circle', title="loss", legend=["loss_l2"]))

def denormalize(imgs):
    imgs = (imgs * 0.5 + 0.5)

if __name__ == "__main__":
    root_path = "/disks/disk0/fjl/ffhq/images1024x1024"
    vis = Visdom(env="vq-vae-2")
    for root, dirs, f in os.walk(root_path):
        for img in [i for i in f if i.endswith(".png") and not i.startswith('.')]:
            src_path = os.path.join(root, img)
            dst = _load_img(src_path)
            dst_path = src_path.replace('1024x1024', '256x256')
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
            cv.imwrite(dst_path, dst)

