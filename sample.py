import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    model.eval()
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache) # 采样时，从第一行（第一次值都为0）开始输入，而后输入目标像素之前所有生成过的所有像素。
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device):
    ckpt = torch.load(checkpoint)

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE(in_channel=1)

    elif model == 'pixelsnail_top':
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
        )

    elif model == 'pixelsnail_bottom':
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
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--vqvae', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae_cycle_eye/vqvae_best.pt")
    parser.add_argument('--top', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae/pixelsnail/pixelsnail_top_eye_002.pt")
    parser.add_argument('--bottom', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae/pixelsnail/pixelsnail_bottom_eye_004.pt")
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--filename', type=str, default='eye_epoch_2.png')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    filename = os.path.join("./sample", args.filename)

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch, [9, 15], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [18, 30], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, filename, normalize=True, range=(-1, 1))
