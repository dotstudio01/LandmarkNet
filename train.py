import argparse
from collections import OrderedDict
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SequentialSampler, Sampler

import sys
sys.path.append('../')
from utils.dataloaders import DeepLoc, KITTI, SevenScenes, CambridgeLandmarks, TUM, ImageListDataset
from utils.math.quaternion_math import quaternion_angular_error
from utils.tools import Logger, IOUMeter, AverageMeter, visualize_features_batch
import utils.transforms.cv_transforms as transforms
from torchvision.transforms import Normalize

from model.loss import PoseCriterion, FeatureCriterion
from model.residual_attention_network import ResidualAttentionModel_92
# from model.fpn import FeaturePyramidNetwork
# from model.pan import PyramidAttentioNetwork
from model.resnet import ResNet
from model.cnn_model import CNNModel

from tools.pose_tools import get_pose_stats, NormalizePose
import shutil


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
    if(isinstance(m, torch.nn.LSTM)):
        torch.nn.init.xavier_uniform_(m.all_weights[0][0], gain=1.2)
        torch.nn.init.xavier_uniform_(m.all_weights[0][1], gain=1.2)

        torch.nn.init.xavier_uniform_(m.all_weights[1][0], gain=1.2)
        torch.nn.init.xavier_uniform_(m.all_weights[1][1], gain=1.2)

        torch.nn.init.xavier_uniform_(m.all_weights[2][0], gain=1.2)
        torch.nn.init.xavier_uniform_(m.all_weights[2][1], gain=1.2)

# 注意区别SequantialSampler
# 序列采样，方便LSTM训练
# LSTM输入形状(batch_size, seq_len, input_size)
# 每个批次采样batch_size * seq_len张图像
class SequenceSampler(Sampler):
    def __init__(self, data_source, batch_size, seq_len, shuffle=False, drop_last=True):
        self.data_source = data_source
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if not self.drop_last:
            raise NotImplementedError

    def __iter__(self):
        # case len(self.data_source = 22), seq_len = 5
        # start_idx = [0, 1, 2, 3]
        start_idx = list(range(len(self.data_source) // self.seq_len))
        if self.shuffle:
            np.random.shuffle(start_idx)

        for batch_idx in range(len(self)):
            yield start_idx[batch_idx // self.seq_len] * self.seq_len + batch_idx % self.seq_len


    def __len__(self):
        return  len(self.data_source) // (self.batch_size * self.seq_len) * (self.batch_size * self.seq_len)

class Trainer(object):
    def __init__(self, model, criterions, train_dataset, val_dataset, args):
        self.config = args

        # 建立Logger和SummaryWriter
        self.logdir = args.logdir
        if not os.path.isfile(args.checkpoint):
            os.makedirs(self.logdir, exist_ok=False)
            # 保存模型和训练的代码，方便后续查看
            shutil.copy('train.sh', os.path.join(self.logdir, 'train.sh'))
            shutil.copy('train.py', os.path.join(self.logdir, 'train.py'))
            shutil.copytree('model', os.path.join(self.logdir, 'model'))
        else:
            assert os.path.exists(self.logdir)
        self.logger = Logger(os.path.join(self.logdir, 'train.log'))
        self.logger.write('Dataset %s \ntrain_images %d\nval_images %d\n' %
                     (args.dataset, len(train_dataset), len(val_dataset)))
        self.writer = SummaryWriter(os.path.join(self.logdir, 'runs'))
        # 计算数据集的均值和方差
        if args.normalize_pose:
            mean, std = get_pose_stats(args.dataset, args.data_dir, args.scene)
        else:
            mean = np.zeros(7, dtype=np.float32)
            std = np.ones(7, dtype=np.float32)
        train_dataset.pose_transform = NormalizePose(mean, std)
        val_dataset.pose_transform = NormalizePose(mean, std)
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

        # 加载model，并根据需要移入GPU
        self.model = model
        self.pose_criterion = criterions[0]
        self.feature_criterion = criterions[1]
        self.semantic_criterion = criterions[2]
        # self.iouMeter = IOUMeter(34)
        if self.config.GPUs > 0:
            if self.config.GPUs > 1:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=range(self.config.GPUs))
            self.model.cuda()
            self.pose_criterion.cuda()
            self.feature_criterion.cuda()
            self.semantic_criterion.cuda()
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        # 设置optimizer和scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [20, 40, 80, 120], np.sqrt(1))
        self.start_epoch = 0
        # 检查是否有checkpoint需要导入
        if os.path.isfile(args.checkpoint):
            self.logger.write('Load checkpoint %s' % args.checkpoint)
            ckpt = torch.load(args.checkpoint)
            self.logger.write('Epoch %d' % ckpt['epoch'])
            self.start_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(ckpt['optim_state_dict'])
            self.logger.write('Checkpoint loaded')

        train_sampler = SequenceSampler(train_dataset, args.batch_size, args.seq_len, shuffle=True)
        val_sampler = SequenceSampler(val_dataset, args.batch_size, args.seq_len, shuffle=False)
        # 定义DataLoader
        # shuffle is done in sampler, no need to activate shuffle here.
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size * args.seq_len,
                                                   shuffle=False, num_workers=args.num_workers,
                                                   sampler=train_sampler)
        # 测试的时候，一张一张地输入
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=args.batch_size * args.seq_len,
                                                  shuffle=False, num_workers=args.num_workers,
                                                  sampler=val_sampler)
        self.t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        self.q_criterion = quaternion_angular_error
        self.t_error_best = 9999
        self.q_error_best = 9999

    def eval(self, epoch):
        self.model.eval()
        # self.iouMeter.reset()
        poses_gt = []
        poses_estimate = []
        for batch_idx, items in enumerate(self.val_loader):
            if self.config.with_label:
                images = items[0]
                labels = items[1]
                poses = items[2]
            else:
                images = items[0]
                poses = items[1]

            # labels = labels.view(self.config.batch_size, self.config.seq_len, *labels.shape[2:])
            # poses = poses.view(self.config.batch_size, self.config.seq_len, *poses.shape[2:])
            if args.GPUs > 0:
                images = images.cuda()
                if self.config.with_label:
                    labels = labels.cuda()
                poses = poses.cuda()
            images = images.view(self.config.batch_size, self.config.seq_len, *images.shape[1:])
            with torch.no_grad():
                _ ,poses_pred, features_pred = self.model(images)
            images = images.view(self.config.batch_size * self.config.seq_len, *images.shape[2:])
            poses_pred = poses_pred.view(self.config.batch_size * self.config.seq_len, *poses_pred.shape[2:])
            features_pred = features_pred.view(self.config.batch_size * self.config.seq_len, *features_pred.shape[2:])
            # # 更新mIOU
            # semantic_pred = semantic_pred.cpu().numpy()
            # labels = labels.cpu().numpy()
            # label_pred = np.argmax(semantic_pred, axis=1)
            # self.iouMeter.add_image(label_pred, labels)
            # Do not forget to convert back
            if batch_idx == 0:
                _, heatmap, heatmap_blend = visualize_features_batch(images, features_pred, indices=[], with_mean=True)

                heatmap_HWC = np.zeros_like(heatmap_blend)
                heatmap_HWC[:, :, 0] = heatmap_blend[:, :, 2]
                heatmap_HWC[:, :, 1] = heatmap_blend[:, :, 1]
                heatmap_HWC[:, :, 2] = heatmap_blend[:, :, 0]
                self.writer.add_image('heatmap_blend', heatmap_HWC, global_step=epoch, dataformats='HWC')
            poses = torch.mul(poses, self.std) + self.mean
            poses_pred = torch.mul(poses_pred, self.std) + self.mean

            poses_gt.append(poses.cpu().numpy())
            poses_estimate.append(poses_pred.cpu().numpy())
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
        if [np.mean(t_loss), np.mean(q_loss)] < [self.t_error_best, self.q_error_best]:
            self.t_error_best = np.mean(t_loss)
            self.q_error_best = np.mean(q_loss)

            filename = os.path.join(self.logdir, 'best.pth.tar'.format(epoch))

            checkpoint_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
            checkpoint_dict.update(
                {'optim_state_dict': self.optimizer.state_dict()})
            torch.save(checkpoint_dict, filename)
        return errs

    def save_checkpoint(self, epoch):
        filename = os.path.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        checkpoint_dict = {'epoch': epoch,
                           'model_state_dict': self.model.state_dict()}
        checkpoint_dict.update(
            {'optim_state_dict': self.optimizer.state_dict()})
        torch.save(checkpoint_dict, filename)

    def train(self):
        start_epoch = self.start_epoch
        total_epoch = self.start_epoch + args.n_epochs
        for epoch in range(start_epoch, total_epoch):
            self.logger.write("epoch %d" % epoch)
            self.model.train()
            self.writer.add_scalar('optim/lr', self.scheduler.get_lr()[0], global_step=epoch)
            pose_loss_meter = AverageMeter()
            abs_pose_loss_meter = AverageMeter()
            rel_pose_loss_meter = AverageMeter()
            feature_loss_meter = AverageMeter()
            # semantic_loss_meter = AverageMeter()
            for batch_idx, items in enumerate(self.train_loader):
                if self.config.with_label:
                    images = items[0]
                    labels = items[1]
                    poses = items[2]
                else:
                    images = items[0]
                    poses = items[1]

                if args.GPUs > 0:
                    images = images.cuda()
                    if self.config.with_label:
                        labels = labels.cuda()
                    poses = poses.cuda()

                self.optimizer.zero_grad()
                images = images.view(self.config.batch_size, self.config.seq_len, *images.shape[1:])
                rel_poses_pred, abs_poses_pred, features_pred = self.model(images)

                abs_poses_gt = poses
                poses = poses.view(poses.shape[0] // 2, 2, *poses.shape[1:])
                rel_poses_gt = poses[:, 1] - poses[:, 0]

                # rel_poses_pred = rel_poses_pred.view(self.config.batch_size * self.config.seq_len // 2, *rel_poses_pred.shape[2:])
                abs_poses_pred = abs_poses_pred.view(self.config.batch_size * self.config.seq_len, *abs_poses_pred.shape[2:])
                features_pred = features_pred.view(self.config.batch_size * self.config.seq_len, *features_pred.shape[2:])
                # no need to convert back here, just do it in evaluation
                # poses_pred = torch.mul(poses_pred_raw, self.std) + self.mean
                abs_pose_loss = self.pose_criterion(abs_poses_pred, abs_poses_gt)
                # rel_pose_loss = self.pose_criterion(rel_poses_pred, rel_poses_gt)
                # pose_loss = abs_pose_loss + rel_pose_loss * 100
                pose_loss = abs_pose_loss

                if self.config.with_label:
                    feature_loss = self.feature_criterion(features_pred, labels)
                else:
                    feature_loss = 0
                scale = 0.01
                feature_loss = feature_loss * scale

                loss = pose_loss + feature_loss
                loss.backward()
                abs_pose_loss_meter.update(abs_pose_loss.item(), n=images.shape[0])
                # rel_pose_loss_meter.update(rel_pose_loss.item(), n=images.shape[0])
                pose_loss_meter.update(pose_loss.item(), n=images.shape[0])
                if self.config.with_label:
                    feature_loss_meter.update(feature_loss.item() / scale, n=images.shape[0])
                self.optimizer.step()

                if ((batch_idx + 1) % 5 == 0):
                    self.logger.write("Epoch [%d/%d], Iter [%d/%d] Abs Pose Loss %.4f Rel Pose Loss %.4f Pose Loss: %.4f Feature Loss: %.4f" % (
                        epoch, total_epoch, batch_idx, len(self.train_loader), abs_pose_loss_meter.avg, rel_pose_loss_meter.avg, pose_loss_meter.avg, feature_loss_meter.avg))
            self.scheduler.step()
            self.writer.add_scalar('optim/abs_pose_loss', abs_pose_loss_meter.avg, global_step=epoch)
            self.writer.add_scalar('optim/rel_pose_loss', rel_pose_loss_meter.avg, global_step=epoch)
            self.writer.add_scalar('optim/pose_loss', pose_loss_meter.avg, global_step=epoch)
            self.writer.add_scalar('optim/feature_loss',feature_loss_meter.avg, global_step=epoch)
            if self.config.val_freq > 0:
                if ((epoch + 1) % self.config.val_freq == 0):
                    self.logger.write('val %d' % (epoch))
                    errs = self.eval(epoch)
                    self.logger.write(str(errs))
                    self.writer.add_scalar('err/trans_median', float(
                        errs['Error in translation(median)']), global_step=epoch)
                    self.writer.add_scalar('err/trans_mean', float(
                        errs['Error in translation(mean)']), global_step=epoch)
                    self.writer.add_scalar('err/rot_median', float(
                        errs['Error in rotation(median)']), global_step=epoch)
                    self.writer.add_scalar('err/rot_mean', float(
                        errs['Error in rotation(mean)']), global_step=epoch)
            if ((epoch + 1) % self.config.save_freq == 0):
                self.save_checkpoint(epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="it's usage tip.", description="help info.")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=2)

    # random seed
    parser.add_argument('--seed', type=int, default=10)
    # train
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument("--GPUs", type=int, default=1)
    # log
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="log")
    # dataset
    parser.add_argument("--dataset", type=str,
                        choices=['7Scenes', 'RobotCar', 'KITTI', 'DeepLoc', 'CambridgeLandmarks', 'TUM', 'ImageListDataset'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--scene", type=str)
    parser.add_argument('--normalize_pose', type=int, default=0)
    # loss
    parser.add_argument("--sx", type=float, default=0)
    parser.add_argument("--sq", type=float, default=0)

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
        val_dataset = SevenScenes(args.data_dir, args.scene, False)
    elif(args.dataset == 'KITTI'):
        h, w = 128, 384
        train_dataset = KITTI(args.data_dir, args.scene, True)
        val_dataset = KITTI(args.data_dir, args.scene, False)
    elif(args.dataset == 'DeepLoc'):
        h, w = 128, 192
        train_dataset = DeepLoc(args.data_dir, True)
        val_dataset = DeepLoc(args.data_dir, False)
    elif(args.dataset == 'CambridgeLandmarks'):
        h, w = 128, 256
        train_dataset = CambridgeLandmarks(args.data_dir, args.scene, True)
        val_dataset = CambridgeLandmarks(args.data_dir, args.scene, False)
    elif(args.dataset == 'TUM'):
        # 原始 480 * 640
        # h, w = 240, 320
        h, w = 256, 320
        train_dataset = TUM(args.data_dir, args.scene, True)
        val_dataset = TUM(args.data_dir, args.scene, True)
    elif(args.dataset == 'ImageListDataset'):
        h, w = 192, 320
        FLAG_COLOR = 1 << 0
        FLAG_DEPTH = 1 << 1
        FLAG_LABEL = 1 << 2
        FLAG_POSE = 1 << 3
        train_dataset = ImageListDataset(args.data_dir, args.scene, 'train_image_list.txt',
                                         flag=FLAG_COLOR | FLAG_LABEL | FLAG_POSE)
        val_dataset = ImageListDataset(args.data_dir, args.scene, 'test_image_list.txt',
                                       flag=FLAG_COLOR | FLAG_LABEL | FLAG_POSE)
    else:
        raise NotImplementedError

    if len(train_dataset[0]) == 2:
        args.with_label = False
    elif len(train_dataset[0]) == 3:
        args.with_label = True
    else:
        raise NotImplementedError

    train_dataset.color_transform = transforms.Compose(
        [transforms.Resize((h, w), cv2.INTER_LINEAR), transforms.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_dataset.color_transform = transforms.Compose(
        [transforms.Resize((h, w), cv2.INTER_LINEAR), transforms.ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if args.with_label:
        train_dataset.label_transform = transforms.Compose(
            [transforms.Resize((h // 16, w // 8), cv2.INTER_NEAREST), transforms.Label2Tensor()])
        val_dataset.label_transform = transforms.Compose(
            [transforms.Resize((h // 16, w // 8), cv2.INTER_NEAREST), transforms.Label2Tensor()])
    # model
    # model = ResidualAttentionModel_92(h, w, dropout_rate=args.dropout_rate)
    model = ResNet(h, w, dropout_rate=args.dropout_rate)
    # model = CNNModel(resnet101(pretrained=False))
    # model = CNNModel(PyramidAttentioNetwork(34, h, w))
    # loss
    pose_criterion = PoseCriterion(args.sx, args.sq)
    feature_criterion = FeatureCriterion()
    semantic_criterion = torch.nn.CrossEntropyLoss()
    criterions = [pose_criterion, feature_criterion, semantic_criterion]

    # trainer
    trainer = Trainer(model, criterions, train_dataset, val_dataset, args)
    trainer.train()
