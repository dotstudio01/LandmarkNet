
import sys
sys.path.append('../')
from transforms3d.quaternions import quat2mat
from utils.dataloaders import ImageListDataset
import cv2
import argparse
import os
import glob
import shutil
import numpy as np
import json

class UndistortHelper:
    def __init__(self, K, distortion, width, height):
        # self.K = K
        # self.distortion = distortion
        new_K = K
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, None, new_K, (width, height), cv2.CV_32FC1)
        self.map1 = map1
        self.map2 = map2

    def __call__(self, image, interpolation=cv2.INTER_NEAREST):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_NEAREST)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create OpenSFM data folder')
    parser.add_argument('scene', type=str)
    parser.add_argument('target_dir', type=str)
    parser.add_argument('--data_dir', type=str, default='/storage/dataset/tj_campus')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--split', type=str, default='full')
    args = parser.parse_args()

    dataset = ImageListDataset(args.data_dir, args.scene, 'test_image_list.txt')

    K = np.array([[1024.7, 0, 635.8],
            [0, 1028.1, 357.2],
            [0, 0, 1]])
    distortion = np.array([-0.3084, 0.1032, 0, 0])
    undistort = UndistortHelper(K, distortion, 1280, 720)

    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'images'), exist_ok=True)
    pose_file = open(os.path.join(args.target_dir, 'pose.txt'), 'w')
    start_idx = max(0, args.start)
    if(args.end == -1):
        end_idx = len(dataset)
    else:
        end_idx = min(args.end, len(dataset))

    exif = {}
    h, w = 360, 640
    for idx in range(start_idx, end_idx):
        frame_info = dataset.image_list[idx]
        heatmap_filename = os.path.join(args.log_dir, 'results', 'heatmap', '%06d.png' % idx)
        color_filename = frame_info['color_filename']
        label_filename = frame_info['label_filename']
        pose = frame_info['pose']

        # 复制原始图像
        # shutil.copy(color_filename, os.path.join(args.target_dir, 'images',  '%06d.png' % idx))
        color = cv2.imread(color_filename)
        color_undistorted = undistort(color)
        color_undistorted = cv2.resize(color_undistorted, (w, h))
        cv2.imwrite(os.path.join(args.target_dir, 'images',  '%06d.png' % idx), color_undistorted)

        pose_file.write('%06d ' % idx)
        # exit(0)
        pose_file.write(' '.join(list(map(lambda x: '%.6f' % x, pose))))
        pose_file.write('\n')
    pose_file.close()
