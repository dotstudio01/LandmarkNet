import numpy as np
import torch

import sys
sys.path.append('../')
from utils.dataloaders import DeepLoc, KITTI, SevenScenes, CambridgeLandmarks

def get_pose_stats(dataset, data_dir, scene):
    # dataset
    if(dataset == '7Scenes'):
        h, w = 128, 160
        train_dataset = SevenScenes(data_dir, scene, True)
    elif(dataset == 'KITTI'):
        h, w = 128, 368
        train_dataset = KITTI(data_dir, scene, True, skip_images=True)
    elif(dataset == 'DeepLoc'):
        h, w = 128, 192
        train_dataset = DeepLoc(data_dir, True)
    elif(dataset == 'CambridgeLandmarks'):
        h, w = 128, 256
        train_dataset = CambridgeLandmarks(data_dir, scene, True)
    else:
        raise NotImplementedError

    poses = []
    train_dataset.skip_images=True
    for image, label, pose in train_dataset:
        poses.append(pose)
    poses = np.vstack(poses)
    mean = np.mean(poses, axis=0)
    std = np.std(poses, axis=0)
    return mean, std

def normalize_pose(pose, mean, std):
    return (pose - mean) / std

def unnormalize_pose(pose, mean ,std):
    return pose * std + mean

class NormalizePose(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, pose):
        return (pose - self.mean) / self.std