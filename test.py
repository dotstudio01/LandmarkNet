import argparse
from collections import OrderedDict
import cv2
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import sys
sys.path.append('../')
from utils.dataloaders import DeepLoc, KITTI, SevenScenes, CambridgeLandmarks, TUM, ImageListDataset
from utils.math.quaternion_math import quaternion_angular_error
from utils.tools import Logger, visualize_features, IOUMeter, gen_color_bar, visualize_features_batch
import utils.transforms.cv_transforms as transforms
from torchvision.transforms import Normalize

from model.loss import PoseCriterion, FeatureCriterion
from model.residual_attention_network import ResidualAttentionModel_92
# from model.fpn import FeaturePyramidNetwork
# from model.pan import PyramidAttentioNetwork
# from model.resnet import resnet50, resnet101
from model.cnn_model import CNNModel

from tools.pose_tools import get_pose_stats, NormalizePose
import shutil


class Evaluator(object):
    def __init__(self, model, train_dataset, test_dataset, args):
        self.config = args
        self.logdir = args.logdir
        assert(os.path.isfile(self.config.checkpoint))
        assert os.path.exists(self.logdir)
        # 计算数据集的均值和方差
        if args.normalize_pose:
            mean, std = get_pose_stats(args.dataset, args.data_dir, args.scene)
        else:
            mean = np.zeros(7, dtype=np.float32)
            std = np.ones(7, dtype=np.float32)
        train_dataset.pose_transform = NormalizePose(mean, std)
        test_dataset.pose_transform = NormalizePose(mean, std)
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

        self.model = model
        # self.iouMeter = IOUMeter(num_classes)
        loc_func = None if self.config.cuda else lambda storage, loc: storage
        checkpoint = torch.load(
            self.config.checkpoint, map_location=loc_func)
        self.model.load_state_dict(checkpoint)
        if self.config.cuda:
            self.model.cuda()
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        
        # 测试的时候，一张一张地输入
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size * args.seq_len,
                                                  shuffle=False, num_workers=args.num_workers,)
        self.logger = Logger(os.path.join(self.logdir, 'test.log'))
        self.logger.write('Dataset %s\ntest_images %d\n' %
                          (args.dataset, len(test_dataset)))

        self.t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        self.q_criterion = quaternion_angular_error
        self.traj_filename = os.path.join(self.logdir, 'estimate.txt')
        self.gt_filename = os.path.join(self.logdir, 'gt.txt')

        os.makedirs(os.path.join(self.logdir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, 'results', 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, 'results', 'heatmap'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'heatmap_color'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, 'results', 'heatmap_blend'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label_color'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label_blend'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label_gt'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label_gt_color'), exist_ok=True)
        # os.makedirs(os.path.join(self.logdir, 'results', 'label_gt_blend'), exist_ok=True)

    def save_features(self, batch_idx, images, features_pred, labels_pred=None):
        back_transform = transforms.Compose(
            [Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
                Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])
        for i in range(images.shape[0]):
            idx = self.config.batch_size * batch_idx + i
            image = back_transform(images[i])
            feature_pred = features_pred[i]
            feature_pred_shape = feature_pred.shape
            feature_pred = feature_pred.view(feature_pred_shape[0], -1)
            mean_value = torch.mean(torch.abs(feature_pred), axis=1)
            order = [j for j in range(feature_pred_shape[0])]
            order = sorted(order, key=lambda x: mean_value[x], reverse=True)
            indices = [order[j] for j in range(3)]
            indices = []
            for indice in range(3):
                indices.append(indice)
            feature_pred = feature_pred.view(*feature_pred_shape)
            image, heatmap, heatmap_color, heatmap_blend = visualize_features(image, feature_pred, indices=indices, with_mean=False, scale_factor=0)
            
            cv2.imwrite(os.path.join(self.logdir, 'results', 'image',
                                     '%06d.png' % idx), image)
            cv2.imwrite(os.path.join(self.logdir, 'results', 'heatmap',
                                     '%06d.png' % idx), heatmap)
            # cv2.imwrite(os.path.join(self.logdir, 'results', 'heatmap_color',
            #                          '%06d.png' % idx), heatmap_color)
            cv2.imwrite(os.path.join(self.logdir, 'results', 'heatmap_blend',
                                    '%06d.png' % idx), heatmap_blend)
            # if labels_pred is not None:
            #     label_pred = labels_pred[i]
            #     label_pred = label_pred.cpu().numpy()
            #     label_pred_color = colormap[label_pred]
            #     label_blend = cv2.addWeighted(image, 0.5, label_pred_color, 0.5, 0)
            #     cv2.imwrite(os.path.join(self.logdir, 'results', 'label',
            #                              '%06d.png' % idx), label_pred)
            #     cv2.imwrite(os.path.join(self.logdir, 'results', 'label_color',
            #                              '%06d.png' % idx), label_pred_color)
            #     cv2.imwrite(os.path.join(self.logdir, 'results', 'label_blend',
            #                              '%06d.png' % idx), label_blend)

    def save_multi_stage_features(self, batch_idx, images, features):
        batch_size = features[0].shape[0]
        h, w = features[0].shape[2:]
        features_mean = []
        for feature in features:
            feature_resized = torch.nn.functional.upsample_bilinear(feature, (h, w))
            feature_mean = torch.mean(torch.abs(feature_resized), axis=1)
            # feature_mean = feature_mean.view(batch_size, -1, h, w)
            features_mean.append(feature_mean)
        features_mean = torch.stack(features_mean, dim=1)
        # print(features_mean.shape)
        self.save_features(batch_idx, images, features_mean)

    def save_features_batch(self, batch_idx, images, features_pred):
        image_vis, heatmap_vis, heatmap_blend_vis = visualize_features_batch(images, features_pred)
        cv2.imwrite(os.path.join(self.logdir, 'results', 'image',
                                    '%06d.png' % batch_idx), image_vis)
        cv2.imwrite(os.path.join(self.logdir, 'results', 'heatmap',
                                    '%06d.png' % batch_idx), heatmap_vis)
        cv2.imwrite(os.path.join(self.logdir, 'results', 'heatmap_blend',
                                '%06d.png' % batch_idx), heatmap_blend_vis)

    def export(self, poses_pred, poses_gt):
        with open(self.traj_filename, 'w') as f:
            for idx, pose in enumerate(poses_pred):
                f.write(str(idx))
                for cell in pose:
                    f.write(' %.6f' % cell)
                f.write('\n')
        f.close()
        print('{:s} saved'.format(self.traj_filename))

        with open(self.gt_filename, 'w') as f:
            for idx, pose in enumerate(poses_gt):
                f.write(str(idx))
                for cell in pose:
                    f.write(' %.6f' % cell)
                f.write('\n')
        f.close()
        print('{:s} saved'.format(self.gt_filename))

    def feature_callback(self, model, input, output):
        feature = output.detach()
        self.features.append(feature)

    def eval(self, use_grad_feature=False):
        self.model.eval()
        # self.iouMeter.reset()
        self.model.attention_module1.register_forward_hook(self.feature_callback)
        self.model.attention_module2_2.register_forward_hook(self.feature_callback)
        # self.model.attention_module3.register_forward_hook(self.feature_callback)
        self.model.attention_module3_3.register_forward_hook(self.feature_callback)
        # self.model.residual_block6.register_forward_hook(self.feature_callback)
        poses_gt = []
        poses_estimate = []
        for batch_idx, items in enumerate(self.test_loader):
            if self.config.with_label:
                images = items[0]
                labels = items[1]
                poses = items[2]
            else:
                images = items[0]
                poses = items[1]
            if self.config.cuda:
                images = images.cuda()
                if self.config.with_label:
                    labels = labels.cuda()
                poses = poses.cuda()
            if images.shape[0] != self.config.batch_size * self.config.seq_len:
                continue
            images = Variable(images, requires_grad=True)
            imgs = images.view(1, *images.shape)

            self.features = []
            _, poses_pred, features_pred = self.model(imgs)

            poses_pred = poses_pred.view(*poses_pred.shape[1:])
            features_pred = features_pred.view(*features_pred.shape[1:])
            # 更新mIOU
            # semantic_pred = semantic_pred.cpu().numpy()
            # labels = labels.cpu().numpy()
            # label_pred = np.argmax(semantic_pred, axis=1)
            # self.iouMeter.add_image(label_pred, labels)
            if use_grad_feature:
                loss = torch.mean(features_pred)
                # loss = torch.mean(poses_pred)
                loss.backward()
                features_pred = images.grad
            # Do not forget to convert back
            poses = torch.mul(poses, self.std) + self.mean
            poses_pred = torch.mul(poses_pred, self.std) + self.mean

            images = images.detach()
            poses = poses.detach()
            poses_pred = poses_pred.detach()
            features_pred = features_pred.detach()
            # save images and feature map
            self.save_features(batch_idx, images, features_pred)
            # self.save_multi_stage_features(batch_idx, images, self.features)
            # self.save_features_batch(batch_idx, images, features_pred)
            poses_gt.append(poses.cpu().numpy())
            poses_estimate.append(poses_pred.cpu().numpy())
            if (batch_idx + 1) % 20 == 0:
                print('batch:%d/%d' % (batch_idx, len(self.test_loader)))

        qs = [q[:, 3:]/(np.linalg.norm(q[:, 3:], axis=-1, keepdims=True)+1e-12)
              for q in poses_estimate]
        poses_estimate = [np.concatenate((poses_estimate[i][:, :3], q), axis=-1)
                      for i, q in enumerate(qs)]
        poses_gt = np.vstack(poses_gt)
        poses_estimate = np.vstack(poses_estimate)

        t_loss = np.asarray([self.t_criterion(p, t) for p, t in zip(poses_estimate[:, :3],
                                                                    poses_gt[:, :3])])
        q_loss = np.asarray([self.q_criterion(p, t) for p, t in zip(poses_estimate[:, 3:],
                                                                    poses_gt[:, 3:])])

        errs = OrderedDict({
            "Error in translation(median)": "{:5.3f}".format(np.median(t_loss)),
            "Error in translation(mean)": "{:5.3f}".format(np.mean(t_loss)),
            "Error in rotation(median)": "{:5.3f}".format(np.median(q_loss)),
            "Error in rotation(mean)": "{:5.3f}".format(np.mean(q_loss)),
            # "mIOU": "{:5.3f}".format(self.iouMeter.mIOU()),
        })
        self.logger.write(str(errs))
        self.export(poses_estimate, poses_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="it's usage tip.", description="help info.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=2)

    # test
    parser.add_argument("--cuda", type=int, default=1)
    # random seed
    parser.add_argument('--seed', type=int, default=10)
    # log
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--logdir", type=str, default="log")
    # dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--scene", type=str)
    parser.add_argument('--normalize_pose', type=int, default=0)
    parser.add_argument('--with_label', type=bool, default=True)
    # split
    parser.add_argument('--split', type=str, required=True)

    args = parser.parse_args()
    # 设置随机种子
    # safe to call this method when cuda is unavailable according to official documentation
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # dataset
    if(args.dataset == '7Scenes'):
        h, w = 128, 160
        train_dataset = SevenScenes(args.data_dir, args.scene, True)
        test_dataset = SevenScenes(args.data_dir, args.scene, False)
    elif(args.dataset == 'KITTI'):
        h, w = 128, 384
        train_dataset = KITTI(args.data_dir, args.scene, True)
        test_dataset = KITTI(args.data_dir, args.scene, split=args.split)
        num_classes = 34
        colormap = train_dataset.colormap
        cv2.imwrite(os.path.join(args.logdir, 'colorbar.png'), gen_color_bar(colormap))
    elif(args.dataset == 'DeepLoc'):
        h, w = 128, 192
        train_dataset = DeepLoc(args.data_dir, True)
        test_dataset = DeepLoc(args.data_dir, False)
    elif(args.dataset == 'CambridgeLandmarks'):
        h, w = 128, 256
        train_dataset = CambridgeLandmarks(args.data_dir, args.scene, True)
        if(args.split=='train'):
            test_dataset = CambridgeLandmarks(args.data_dir, args.scene, True)
        elif(args.split=='test'):
            test_dataset = CambridgeLandmarks(args.data_dir, args.scene, False)
        else:
            raise NotImplementedError
    elif(args.dataset == 'TUM'):
        # 原始 480 * 640
        # h, w = 240, 320
        h, w = 256, 320
        train_dataset = TUM(args.data_dir, args.scene, True)
        if(args.split=='train'):
            test_dataset = TUM(args.data_dir, args.scene, True)
        elif(args.split=='test'):
            test_dataset = TUM(args.data_dir, args.scene, False)
        else:
            raise NotImplementedError
    elif(args.dataset == 'ImageListDataset'):
        h, w = 192, 320
        train_dataset = ImageListDataset(args.data_dir, args.scene, 'train_image_list.txt')
        if(args.split == 'train'):
            test_dataset = ImageListDataset(args.data_dir, args.scene, 'train_image_list.txt')
        elif(args.split == 'test'):
            test_dataset = ImageListDataset(args.data_dir, args.scene, 'test_image_list.txt')
        elif(args.split == 'full'):
            test_dataset = ImageListDataset(args.data_dir, args.scene, 'image_list.txt')
        args.with_label = False
    else:
        raise NotImplementedError

    train_dataset.color_transform = transforms.Compose(
        [transforms.Resize((h, w), cv2.INTER_LINEAR), transforms.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset.color_transform = transforms.Compose(
        [transforms.Resize((h, w), cv2.INTER_LINEAR), transforms.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset.label_transform = transforms.Compose([transforms.Resize((h, w), interpolation=cv2.INTER_NEAREST), transforms.Label2Tensor()])
    test_dataset.label_transform = transforms.Compose([transforms.Resize((h, w), interpolation=cv2.INTER_NEAREST), transforms.Label2Tensor()])

    # model
    model = ResidualAttentionModel_92(h, w, dropout_rate=0)
    # model = CNNModel(resnet101(pretrained=False))    

    # evaluator
    evaluator = Evaluator(model, train_dataset, test_dataset, args)
    evaluator.eval()
