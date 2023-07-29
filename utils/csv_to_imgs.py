#!/usr/bin/env python

import os
import sys
sys.path.append('../')
import shutil

import numpy as np 
import pandas as pd
from tqdm import tqdm
import glob
import cv2

class2emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion2class = dict((v,k) for k,v in class2emotion.items())


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
        # take user input to remove for existing directory
        ans = input("Do you want to remove the existing directory (y/n):")
        if ans == 'y':
            # remove existing directory
            shutil.rmtree(dir_path, ignore_errors=True)
            print('Directory removed.')
            # create new directory
            os.makedirs(dir_path)
        else:
            # exit the script
            print('Existing directory was not removed. Exiting script.')
            sys.exit()



if __name__ == '__main__':
    DATA_PATH = 'dataset_FER/'
    train_csv = pd.read_csv(DATA_PATH + 'renamed_train.csv')
    test_csv = pd.read_csv(DATA_PATH + 'test.csv')

    SAVE_TRAIN_DIR = DATA_PATH + 'train/'
    SAVE_TEST_DIR = DATA_PATH + 'test/'
    make_if_not_dir(SAVE_TRAIN_DIR)
    make_if_not_dir(SAVE_TEST_DIR)

    print('Saving train images...')
    cnt_classes = {0: 0, 1: 0, 2: 0, 3: 0, 4:0, 5:0, 6:0}
    for i in tqdm(range(len(train_csv))):
        img = create_image_from_rmo_list(train_csv['pixels'][i].split(' '))
        annotation = train_csv['emotion'][i]
        emotion = class2emotion[annotation]
        if not os.path.exists(SAVE_TRAIN_DIR + f"{emotion}"):
            os.makedirs(SAVE_TRAIN_DIR + f"{emotion}")
        cv2.imwrite(SAVE_TRAIN_DIR + f"{emotion}/{emotion}_" + str(cnt_classes[annotation]) + '.png', img)
        cnt_classes[train_csv['emotion'][i]] += 1

        with open(DATA_PATH + 'train_annotations.txt', 'a') as f:
            f.write(str(annotation))

    print('Saving test images...')
    for i in tqdm(range(len(test_csv))):
        img = create_image_from_rmo_list(test_csv['pixels'][i].split(' '))
        cv2.imwrite(SAVE_TEST_DIR + str(i) + '.png', img)