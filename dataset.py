import os
import pickle
import numpy as np
from collections import namedtuple
from PIL import Image
from multiprocessing import Pool
import functools
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.imgs[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

class list_dataset(datasets.ImageFolder):
    def __init__(self, paths, transform):
        self.transform = transform
        self.paths = paths

    def __getitem__(self, index):
        return self.transform(_load_img(self.paths[index])), np.asarray([1])

    def __len__(self):
        return len(self.paths)


def _load_imgs(img_path, name = 'eye_image'):
    img = cv2.imread(img_path, 0)
    return np.asarray([img]).astype(np.float32)

def _load_img(path):
    img = Image.open(path)
    return img

def _geth5file(filename, use_face, stats):
    stats["length"] -= 1
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        data_dict['left_eye'] = f['left_eye'].value
        data_dict['right_eye'] = f['right_eye'].value
        data_dict['t_left'] = f['left_label'].value
        data_dict['t_right'] = f['right_label'].value
        data_dict['head_pose_l'] = f['left_head_poses'].value
        data_dict['head_pose_r'] = f['right_head_poses'].value
        if use_face:
            data_dict['face']= f['face'].value
    print(stats["length"])
    return data_dict

def _getnpzfile(filename, use_face, stats):
    stats["length"] -= 1
    data_npz = np.load(filename, allow_pickle=True)
    data_dict = data_npz['dataset'][()]
    assert isinstance(data_dict, dict)
    data_dict['left_eye'] = _load_imgs(data_dict['left_eye'])
    data_dict['right_eye'] = _load_imgs(data_dict['right_eye'])
    if not use_face:
        data_dict.pop('face', None)
    else:
        data_dict['face'] = _load_imgs(data_dict['face'], name='face')
    print(stats["length"])
    return data_dict

def _split_dataset(paths_list, valid_rate=0.2):
    if valid_rate <= 0:
        return paths_list, None
    len_dataset = len(paths_list)
    total_idx = np.arange(len_dataset)
    validation_size = len_dataset * valid_rate
    validation_idx = np.random.choice(len_dataset, validation_size, replace=False)
    train_idx = np.setdiff1d(total_idx, validation_idx)
    train_paths = []
    valid_paths = []
    for i in train_idx:
        train_paths.append(paths_list[i])
    for i in validation_idx:
        valid_paths.append(paths_list[i])
    return train_paths, valid_paths

def _all_image_paths(path, exts=['.png', '.jpg']):
    image_paths = []
    for root, dirs, f in os.walk(path):
        for img in f:
            ext = os.path.splitext(img)[1]
            if ext in exts and not img.startswith('.'):
                src_path = os.path.join(root, img)
                image_paths.append(src_path)
    return image_paths
