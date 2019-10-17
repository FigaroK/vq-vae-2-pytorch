#!/usr/bin/env python3

import cv2 as cv
from visdom import Visdom
import numpy as np

def _show_img(vis, img, win_name, keep_history=False):
    vis.image(img, caption=win_name, store_history=keep_history)

def _load_img(path, mode=cv.IMREAD_COLOR):
    img = cv.imread(path, mode)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    return img

if __name__ == "__main__":
    root_path = "/disks/disk0/fjl/ffhq/"
