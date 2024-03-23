import os, getpass
import os.path as osp
import numpy as np
import argparse
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = '.'
add_path(osp.join(root_dir, 'lib'))
add_path(osp.join(root_dir, 'util'))
add_path(osp.join(root_dir, 'tools'))


class Config:
    imgDir = '/root/autodl-tmp/Crowdhuman/train'
    imgDir2 = '/root/autodl-tmp/Crowdhuman/val'

    output_dir = 'DH_head'
    snapshot_dir = osp.join(output_dir, 'model_dump')
    eval_dir = osp.join(output_dir, 'eval_dump')
    
    train_image_folder, val_image_folder = imgDir, imgDir2
    train_json = r'train.json'
    eval_json = r'val.json'
    anno_file = r'crowdhuman_val.odgt'
    train_file = r'crowdhuman_train.odgt'

    asymmetrical_low = 0.1
    score_thr = 1
    ign_thr = 0.7
    watershed = 5
    iou_thr = 0.4
    floor_thr = 0.05
    iter_nums = 1

config = Config()
