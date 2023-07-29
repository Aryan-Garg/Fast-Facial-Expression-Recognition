#!/usr/bin/env python

import os
import sys
sys.append('../')

import numpy as np 
import cv2
import pandas as pd
from tqdm import tqdm
import glob


def csv_to_imgs(csv_path, img_path, img_size, img_type):
    """
    csv_path: path to the csv file
    img_path: path to the folder where images are stored
    img_size: size of the image
    img_type: type of the image
    """
    df = pd.read_csv(csv_path)
    for i in tqdm(range(df.shape[0])):
        img_name = df.iloc[i, 0]
        img = cv2.imread(os.path.join(img_path, img_name), 0)
        # img = cv2.resize(img, (img_size, img_size))
        # cv2.imwrite(os.path.join(img_path, img_name.split('.')[0] + img_type), img)

if __name__ == '__main__':
    csv_to_imgs('../dataset_FER/train.csv', '../data/train', 48, '.jpg')