import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
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

class gaze_dataset(data.Dataset):
    def __init__(self, h5_path_list, use_face=False, num_workers=0):
        stats = dict(length=len(h5_path_list))
        self.use_face = use_face
        self.num_workers = num_workers
        h5_path_list = list(filter(lambda x : len(x) > 5, h5_path_list))
        ext = os.path.splitext(h5_path_list[0])[1]
        if ext == '.npz':
            _getfile = _getnpzfile
        elif ext == '.h5':
            _getfile = _geth5file
        else:
            raise TypeError('fileType error')
        self.dataset = []
        num = len(h5_path_list)
        pool = Pool(self.num_workers)
        cache_dataset = functools.partial(_getfile, use_face=use_face, stats=stats)
        datasets = pool.map(cache_dataset, h5_path_list)
        self.dataset.extend(datasets)

    def __getitem__(self, index):
        f  = self.dataset[index]
        left_eye_img = f['left_eye']
        right_eye_img = f['right_eye']
        left_label = f['t_left']
        right_label = f['t_right']
        left_headpose = f['head_pose_l']
        right_headpose = f['head_pose_r']
        if self.use_face:
            face = f.get('face', None)
            return left_eye_img, right_eye_img, left_label, right_label, left_headpose, right_headpose, face
        else:
            return left_eye_img, right_eye_img, left_label, right_label, left_headpose, right_headpose

    def __len__(self):
        return len(self.dataset)


def _load_imgs(img_path, name = 'eye_image'):
    img = cv2.imread(img_path, 0)
    return np.asarray([img]).astype(np.float32)

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
