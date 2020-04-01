import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
import config
from dataset import ImageFileDataset, CodeRow, load_paths_from_txt, H5_EYE_dataset
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        loader = tqdm(loader)

        for i, (img, right_eye_img, left_label, right_label, left_headpose, right_headpose) in enumerate(loader):
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for f, top, bottom, headpose in zip(left_label, id_t, id_b, left_headpose):
                row = CodeRow(top=top, bottom=bottom, label=f, headpose=headpose)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                loader.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--beta', type=str)
    parser.add_argument('--config', type=str, default="./config_eye.json")

    args = parser.parse_args()
    config.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'# str(config.gpu_id)
    paths = load_paths_from_txt([config.txt_path])

    device = 'cuda'
    # transform = transforms.Compose(
    #     [
    #         # transforms.Resize(args.size),
    #         # transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    dataset = H5_EYE_dataset(paths)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    model = VQVAE(in_channel=1)
    model.load_state_dict(torch.load(config.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.beta, map_size=map_size)

    extract(env, loader, model, device)
