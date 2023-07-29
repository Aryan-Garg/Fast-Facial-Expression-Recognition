#!/usr/bin/env python

import os
import sys
sys.path.append('../')

import numpy as np 
import pandas as pd
from tqdm import tqdm
import glob
import cv2


def create_image_from_rmo_list(pixel_lst):
    img = np.zeros((48, 48), dtype=np.uint8)
    for i in range(48):
        for j in range(48):
            img[i][j] = int(pixel_lst[i * 48 + j])
    return img


def make_if_not_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print('Directory already exists.')
        # Clean based on user input
        while True:
            user_input = input('Clean directory? [y/n]: ')
            if user_input == 'y':
                files = glob.glob(dir_path + '*')
                for f in files:
                    os.remove(f)
                break
            elif user_input == 'n':
                break
            else:
                print('Invalid input. Try again.')


if __name__ == '__main__':
    DATA_PATH = 'dataset_FER/'
    train_csv = pd.read_csv(DATA_PATH + 'renamed_train.csv')
    test_csv = pd.read_csv(DATA_PATH + 'test.csv')

    SAVE_TRAIN_DIR = DATA_PATH + 'train/'
    SAVE_TEST_DIR = DATA_PATH + 'test/'
    make_if_not_dir(SAVE_TRAIN_DIR)
    make_if_not_dir(SAVE_TEST_DIR)

    print('Saving train images...')
    for i in tqdm(range(len(train_csv))):
        img = create_image_from_rmo_list(train_csv['pixels'][i].split(' '))
        annotation = train_csv['emotion'][i]
        cv2.imwrite(SAVE_TRAIN_DIR + str(i) + '.png', img) # png to not degrade quality as in jpg!!
        with open(DATA_PATH + 'train_annotations.txt', 'a') as f:
            f.write(str(annotation))

    print('Saving test images...')
    for i in tqdm(range(len(test_csv))):
        img = create_image_from_rmo_list(test_csv['pixels'][i].split(' '))
        cv2.imwrite(SAVE_TEST_DIR + str(i) + '.png', img)