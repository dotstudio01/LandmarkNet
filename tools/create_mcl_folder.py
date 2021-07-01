import cv2
import argparse
import os
import glob
import shutil
import numpy as np
import json
import sys
sys.path.append('../')
from utils.dataloaders import KITTI
from transforms3d.quaternions import quat2mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create mcl data folder')
    parser.add_argument('scene', type=str)
    parser.add_argument('target_dir', type=str)
    parser.add_argument('--data_dir', type=str, default='/storage/dataset/KITTI')
    # parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    dataset = KITTI(args.data_dir, args.scene, split=args.split)

    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'labels'), exist_ok=True)
    pose_file = open(os.path.join(args.target_dir, 'pose.txt'), 'w')
    start_idx = max(0, args.start)
    if(args.end == -1):
        end_idx = len(dataset)
    else:
        end_idx = min(args.end, len(dataset))

    exif = {}
    for idx in range(start_idx, end_idx):
        frame_info = dataset.image_list[idx]
        color_filename = frame_info[0]
        label_filename = frame_info[1]
        pose = frame_info[2]

        # 复制原始图像
        shutil.copy(color_filename, os.path.join(args.target_dir, 'images',  '%06d.png' % idx))
        shutil.copy(label_filename, os.path.join(args.target_dir, 'labels',  '%06d.png' % idx))
        pose_file.write('%06d ' % idx)
        for i in range(6):
            pose_file.write('%.6f ' % pose[i])
        pose_file.write('%.6f\n' % pose[6])