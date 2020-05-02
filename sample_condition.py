import argparse
import sys
sys.path.append('/disks/disk2/fjl/Proj/Python_Proj')
# from gaze_pytorch.dataset import eyediap_dataset
from gaze_pytorch.utils_old import rad2deg, angle_err
from dataset import rgb2gray, torch2cv2, H5_EYE_dataset
import os
import models
import h5py

import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

import numpy as np
from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from dataset import get_gaze_pose
from glob import glob

        
@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None, cond_headpose_gaze=None):
    model.eval()
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
    # for i in range(size[0]):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cond_headpose_gaze=cond_headpose_gaze, cache=cache) # 采样时，从第一行（第一次值都为0）开始输入，而后输入目标像素之前所有生成过的所有像素。
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample
    return row

@torch.no_grad()
def sample_by_dataset(test_loader, vqvae_model, pixelsnail_top, pixelsnail_bottom, gaze_predictor):
    # loader = tqdm(test_loader)
    loader = test_loader
    vqvae_model.eval()
    pixelsnail_top.eval()
    pixelsnail_bottom.eval()
    gaze_predictor.eval()
    err_sum_a = 0
    err_sum_b = 0
    err_sum_c = 0
    num = 0
    recon_imgs = []
    recon_gazes = []
    pred_gazes = []
    headposes = []
    for i, (img, label, headpose) in enumerate(loader):
        print(f"{i} / {len(loader)}")
        img = img.cuda()
        num += img.shape[0]
        label = label.cuda()
        headpose = headpose.cuda()
        pred_gaze = gaze_predictor(img, headpose)
        cond_gazepose = torch.cat([pred_gaze, headpose], 1)
        top_index = sample_model(pixelsnail_top, device, img.shape[0], [9, 15], args.temp, cond_headpose_gaze=cond_gazepose)
        bottom_index = sample_model(pixelsnail_bottom, device, img.shape[0], [18, 30], args.temp, condition=top_index, cond_headpose_gaze=cond_gazepose)
        decoded_sample = model_vqvae.decode_code(top_index, bottom_index)
        decoded_sample = decoded_sample.clamp(-1, 1)
        # save_image(decoded_sample, filename, normalize=True, range=(-1, 1), nrow=4)
        recon_gaze = gaze_predictor(torch2cv2(decoded_sample).cuda(), headpose)
        recon_imgs.append(torch2cv2(decoded_sample, normalize=False))
        # recon_gazes.append(recon_gaze.cpu().numpy())
        pred_gazes.append(pred_gaze.cpu().numpy())
        headposes.append(headpose.cpu().numpy())

        a = rad2deg(angle_err(recon_gaze, pred_gaze))
        b = rad2deg(angle_err(recon_gaze, pred_gaze))
        print('recon pre err: ', a.mean())
        a = a.sum()
        err_sum_a += a
    with h5py.File("/disks/disk2/fjl/dataset/mpii_recon.h5", 'w') as f:
        f['left_gaze'] = np.vstack(pred_gazes)
        f['left_eye_img'] = np.vstack(recon_imgs)
        f['left_headpose'] = np.vstack(headposes)
    print('recon pre err: ', err_sum_a / num)

def load_model(model, checkpoint, device):
    ckpt = torch.load(checkpoint)

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE(in_channel=3)

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
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=1,
            cond_headpose_gaze=True
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
            condition=True,
            cond_headpose_gaze=True
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(prefix_chars='@')
    parser.add_argument('@@batch', type=int, default=64)
    # parser.add_argument('@@vqvae', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae_unity_unityEyes_dis_from_MPII/vqvae_best.pt")
    parser.add_argument('@@vqvae', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae_unity_unityEyes_dis_from_MPII/vqvae_best.pt")
    parser.add_argument('@@top', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae/pixelsnail/unity_mpii_condition_all/pixelsnail_top_eye_006_20w.pt")
    parser.add_argument('@@bottom', type=str, default="/disks/disk2/fjl/checkpoint/vq-vae/pixelsnail/unity_mpii_condition_all/pixelsnail_bottom_eye_008.pt")
    parser.add_argument('@@predictor', type=str, default="/disks/disk2/fjl/checkpoint/gaze/cross_single_eye/gazenet_3_1.tar")
    parser.add_argument('@@temp', type=float, default=1.0)
    parser.add_argument('@@gpu', type=str, default='1')
    parser.add_argument('@@gaze', type=str) # gaze(pitch, yaw)
    parser.add_argument('@@pose', type=str) # headpose(pitch, yaw)
    parser.add_argument('@@filename', type=str, default='headpose_pp.png')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # if hasattr(args, "gaze") and hasattr(args, "pose"):
    #     gaze = list(eval(args.gaze))
    #     pose = list(eval(args.pose))
    #     gaze.extend(pose)
    #     cond_gazepose = torch.tensor(gaze).unsqueeze(0).to(torch.float32)
    #     cond_gazepose = cond_gazepose / 180 * np.pi
    #     cond_gazepose = cond_gazepose.repeat(args.batch, 1)
    #     cond_gazepose = cond_gazepose.to(torch.float32).to('cuda')
    # else:
    #     gaze, pose = [0,0], [0,0]
    #     gaze[0] = np.random.normal(-10, 11)
    #     gaze[1] = np.random.normal(0, 20)
    #     pose[0] = np.random.normal(7, 35)
    #     pose[1] = np.random.normal(5, 30)
    #     cond_gazepose = torch.tensor(gaze.extend(pose)).unsqueeze(0)
    #     cond_gazepose = cond_gazepose / 180 * np.pi
    #     cond_gazepose.cuda()
    # cond_gazepose = get_gaze_pose('pose').to('cuda')
    # filename = os.path.join("./sample", args.filename + f"{gaze[0]}_{gaze[1]}_{gaze[2]}_{gaze[3]}.png")
    filename = os.path.join("./sample", args.filename)

    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)
    model_predictor = torch.load(args.predictor)
    test_dataset = H5_EYE_dataset(glob("/disks/disk2/fjl/dataset/MPII_H5_single_evaluation/P*.h5"))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch)
    sample_by_dataset(test_loader, model_vqvae, model_top, model_bottom, model_predictor)


    
    # top_sample = sample_model(model_top, device, args.batch, [9, 15], args.temp, condition=cond_gazepose)
    # bottom_sample = sample_model(
    #     model_bottom, device, args.batch, [18, 30], args.temp, condition=top_sample
    # )

    # decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    # decoded_sample = decoded_sample.clamp(-1, 1)

    # save_image(decoded_sample, filename, normalize=True, range=(-1, 1), nrow=4)
