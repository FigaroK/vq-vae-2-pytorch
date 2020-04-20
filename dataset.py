import os
import cv2
import pickle
import h5py
import numpy as np
import random
from collections import namedtuple
from PIL import Image
from multiprocessing import Pool
import functools
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.utils import save_image
import lmdb
import math


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'label', 'headpose'])

def identity(x):
    return x

def denormalize(x):
    return (x * 0.5 + 0.5) * 255

def torch2cv2(x):
    x = denormalize(x)
    x = x.cpu().numpy().astype(np.uint8)
    gray_list = []
    for i in x:
        rgb = np.transpose(i, [1,2,0])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_list.append(gray[np.newaxis, np.newaxis, :])
    gray_list = np.vstack(gray_list).astype(np.float32)
    a = torch.from_numpy(cv2Image(gray_list))
    save_image(a, "./test.png", normalize=True, range=(-1, 1), nrow=4)
    return a
    
def get_gaze_pose(mode):
    gaze = np.zeros([2, 4, 4])
    pitch = np.arange(-30, 50, 20).astype(np.float32)[:, np.newaxis]
    yaw = np.arange(-30, 50, 20).astype(np.float32)[np.newaxis, :] 
    pitch_repeat = np.repeat(pitch, 4, axis = 1)[np.newaxis, :]
    yaw_repeat = np.repeat(yaw, 4, axis = 0)[np.newaxis, :]
    if mode == 'gaze':
        gaze_pose = np.vstack([pitch_repeat, yaw_repeat, gaze])
    if mode == 'pose':
        gaze_pose = np.vstack([gaze, pitch_repeat, yaw_repeat])
    if mode == 'all':
        gaze_pose = np.vstack([pitch_repeat, yaw_repeat, pitch_repeat, yaw_repeat])
    gaze_pose = np.transpose(gaze_pose.reshape(4, 16), [1,0])
    a = torch.from_numpy(gaze_pose).to(torch.float32) / 180 * np.pi
    print(a)
    print(a.shape)
    return a

def load_paths_from_txt(txt_paths):
    paths = []
    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            data = f.read()
            data = data.strip('\n')
            paths.extend(data.split('\n'))
    return paths

def d_3(result):
    result = torch.from_numpy(result)
    data = torch.zeros([3])

    data[0] = (-1) * (torch.cos(result[0])) * (torch.sin(result[1]))

    data[1] = (-1) * (torch.sin(result[0]))

    data[2] = (-1) * (torch.cos(result[0])) * (torch.cos(result[1]))

    return data  # size 128*3

def d_2(label):
    theta = np.arcsin((-1) * label[1]) # 以相机为主视人，上正，下负. 输入是单位向量
    phi = np.arctan2((-1) * label[0], (-1) * label[2]) # 以相机为主视人，向右看是正，左看是-
    return np.asarray([theta, phi]).astype(np.float32)

def cv2Image(x):
    x = (x / 255 - 0.5) / 0.5
    if x.ndim == 2:
        return x[np.newaxis, :]
    return x

def rgb2gray(x):
    x = np.transpose(x, [1,2,0])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2Image(x)

class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.imgs[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path, transform_label=identity):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.transform_label = transform_label

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
            label = self.transform_label(row.label)
            headpose = row.headpose

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), label, headpose

# class LMDBDataset(Dataset):
#     def __init__(self, path):
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
# 
#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)
# 
#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
# 
#     def __len__(self):
#         return self.length
# 
#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = str(index).encode('utf-8')
# 
#             row = pickle.loads(txn.get(key))
# 
#         return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

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
    validation_size = math.ceil(len_dataset * valid_rate)
    valid_paths = random.sample(paths_list, validation_size)
    train_paths = [ i for i in paths_list if i not in valid_paths ]
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

class H5_EYE_dataset(Dataset):
    def __init__(self, h5_path_list, transform=cv2Image, label_trans=identity):
        assert h5_path_list[0].endswith('.h5')
        self.transform = transform
        self.label_trans = label_trans
        self.train_files = [h5py.File(file_name, 'r') for file_name in h5_path_list]
        self.dataset = _eyediap_get_mat(self.train_files)
    
    def __len__(self):
        return self.dataset[-1]

    def __getitem__(self, index):
        left = self.dataset[0][index].astype(np.float32)
        # right = self.dataset[1][index].astype(np.float32)
        left_eye_img = self.transform(left)
        # right_eye_img = self.transform(right)
        left_label = self.label_trans(self.dataset[1][index].astype(np.float32))
        # right_label = self.label_trans(self.dataset[3][index].astype(np.float32))
        left_headpose = self.dataset[2][index].astype(np.float32)
        # right_headpose = self.dataset[5][index].astype(np.float32)
        return left_eye_img, left_label, left_headpose
        

def _eyediap_get_mat(files):
    left_gazes = np.vstack([files[idx]['left_gaze'] for idx in range(len(files))])
    left_headposes = np.vstack([files[idx]['left_headpose'] for idx in range(len(files))])
    images_l = np.vstack([files[idx]['left_eye_img'] for idx in range(len(files))])

    num_instances = images_l.shape[0]
    # dimension = images_l.shape[1]

    print ("%s images loaded" % (num_instances))
    print ("shape of 'images' is %s" % (images_l.shape,))

    return images_l, left_gazes, left_headposes, num_instances

