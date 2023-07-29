#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import fastai
from fastai.vision.all import *
from fastai.vision.widgets import *


def get_files():
    train_files = get_image_files("dataset_FER/train")
    test_files = get_image_files("dataset_FER/test")
    return train_files, test_files


def label_func(fname):
    return fname.split('_')[0]


def get_dataloader(batch_size):
    dls = ImageDataLoaders.from_name_func(".", 
                                        train_files, 
                                        label_func,
                                        item_tfms=Resize(224),
                                        bs=batch_size)
    return dls


def fine_tune(lr, epochs, dls=None):
    assert dls is not None, "dls is None"

    print("[+] Fine tuning...")
    # create learner
    learn = vision_learner(dls, "vit_base_patch16_224", metrics=error_rate)
    
    # find good starting learning rate
    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    print("lr_find (min, steep, valley, slide):", lrs.minimum, lrs.steep, lrs.valley, lrs.slide)
    plt.savefig("lr_find_vit_patch16.png")

    # finetune
    learn.fine_tune(epochs, lr)

    # save model
    learn.export('ckpts/export_vit_base_patch16_224.pkl')


def predict(test_files):
    print(f"[+] Predicting {len(test_files)} images...")
    learn = load_learner('ckpts/export_vit_base_patch16_224.pkl')
    test_dl = learn.dls.test_dl(test_files)
    preds,idx,decoded = learn.get_preds(dl=test_dl,with_decoded=True)
    # preds contain probabilites of each class --> will use for ensembling later
    val = []
    i = 0
    for fns in test_files:
        result={}
        result['file'] = fns.name
        result['emotion'] = learn.dls.vocab[int(decoded[i])]
        i=i+1
        val.append(result)

    # create a dataframe
    print("[+] Creating submission.csv...")
    df = pd.DataFrame(val)
    print(df.head())
    df.to_csv('submission_vit_patch16.csv', index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epochs', type=int, default=10)
    argparser.add_argument('-b', '--batch_size', type=int, default=64)
    argparser.add_argument('-l', '--learning_rate', type=float, default=1e-2)

    args = argparser.parse_args()
    print("Args:", args)

    train_files, test_files = get_files()
    dls = get_dataloader(args.batch_size)
    # print(dls.show_batch(max_n=9)
    print("[+] Dataloader vocab:", dls.vocab)

    fine_tune(args.learning_rate, args.epochs, dls)
    predict(test_files)
