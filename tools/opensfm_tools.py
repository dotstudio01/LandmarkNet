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
    parser = argparse.ArgumentParser(description='Create OpenSFM data folder')
    parser.add_argument('scene', type=str)
    parser.add_argument('target_dir', type=str)
    parser.add_argument('--data_dir', type=str, default='/storage/dataset/KITTI')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--split', type=str, default='full')
    args = parser.parse_args()

    dataset = KITTI(args.data_dir, args.scene, split=args.split)

    os.makedirs(args.target_dir, exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'poses'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'labels'), exist_ok=True)
    # 仅供查看
    os.makedirs(os.path.join(args.target_dir, 'heatmaps_blend'), exist_ok=True)
    start_idx = max(0, args.start)
    if(args.end == -1):
        end_idx = len(dataset)
    else:
        end_idx = min(args.end, len(dataset))

    exif = {}
    for idx in range(start_idx, end_idx):
        frame_info = dataset.image_list[idx]
        heatmap_filename = os.path.join(args.log_dir, 'results', 'heatmap', '%06d.png' % idx)
        color_filename = frame_info[0]
        label_filename = frame_info[1]
        pose = frame_info[2]

        # 复制原始图像
        shutil.copy(color_filename, os.path.join(args.target_dir, 'images',  '%06d.png' % idx))

        # 复制标签图像
        shutil.copy(label_filename, os.path.join(args.target_dir, 'labels',  '%06d.png' % idx))

        # 加载label，生成mask
        label = cv2.imread(label_filename)
        mask = np.zeros_like(label)
        mask = mask + 255
        mask[label==23] = 0
        mask_filename = os.path.join(args.target_dir, 'masks',  '%06d.png' % idx + '.png')
        cv2.imwrite(mask_filename, mask)

        # 加载heatmap，缩放
        heatmap = cv2.imread(heatmap_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        heatmap = cv2.resize(heatmap, (label.shape[1], label.shape[0]))
        heatmap_filename = os.path.join(args.target_dir, 'heatmaps',  '%06d.png' % idx + '.png')
        cv2.imwrite(heatmap_filename, heatmap)

        # 计算并保存heatmap_blend
        # image = cv2.imread(color_filename)
        # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap_blend = cv2.addWeighted(image, 0.5, heatmap_color, 0.5, 0)
        heatmap_blend_filename = os.path.join(args.target_dir, 'heatmaps_blend',  '%06d.png' % idx + '.png')
        # cv2.imwrite(heatmap_blend_filename, heatmap_blend)
        shutil.copy(os.path.join(args.log_dir, 'results', 'heatmap_blend', '%06d.png' % idx), heatmap_blend_filename)

        # 保存位姿
        pose_filename = os.path.join(args.target_dir, 'poses', '%06d.png.npy' % idx)
        pose_T = np.eye(4, dtype=np.float32)
        pose_T[:3, :3] = quat2mat(pose[3:])
        pose_T[:3, 3] = pose[:3]
        np.save(pose_filename, pose_T)
        gps = {}
        gps.update({'latitude': np.float64(pose[0]) / 6371000.0 * 180.0 / np.pi})
        gps.update({'altitude': np.float64(pose[1])})
        gps.update({'longitude': np.float64(pose[2] / 6371000.0 * 180.0 / np.pi)})
        gps.update({'hdop': np.float64(5)})
        exif['%06d.png' % idx] = {'gps': gps}
    f = open(os.path.join(args.target_dir, 'exif_overrides.json'), 'w')
    f.write(json.dumps(exif))



    f = open(os.path.join(args.target_dir, 'image_groups.txt'), 'w')
    idx = start_idx
    split_id = 0
    while(idx < end_idx):
        batch_start = idx
        if(idx + 150 < end_idx):
            batch_end = idx + 100
        else:
            batch_end = end_idx
        for i in range(batch_start, batch_end):
            f.write('%06d.png %03d\n' % (i, split_id))
        if(batch_end < end_idx):
            split_id = split_id + 1
            idx = idx + 100
        else:
            break
