import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseCriterion(nn.Module):
    def __init__(self, sx=0, sq=0):
        super(PoseCriterion, self).__init__()
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=False)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=False)
        self.loss_func = nn.L1Loss()

    def forward(self, pose_pred, pose_gt):
        x_gt = pose_gt[:, :3]
        q_gt = pose_gt[:, 3:]

        x_pred = pose_pred[:,:3]
        q_pred = pose_pred[:, 3:]
        
        x_loss = self.loss_func(x_gt, x_pred) * torch.exp(-self.sx)
        q_loss = self.loss_func(q_gt, q_pred) * torch.exp(-self.sq)
        loss = x_loss + q_loss
        return loss

class FeatureCriterion(nn.Module):
    def __init__(self):
        super(FeatureCriterion, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, features, labels):
        feature_shape = features.shape
        # features_x = features.clone()
        # features_y = features.clone()
        # for i in range(feature_shape[2]):
        #     features_x[:, :, i] = features_x[:, :, i] * i
        # for i in range(feature_shape[3]):
        #     features_y[:, :, :, i] = features_y[:, :, :, i] * i
        
        # features = features.view(feature_shape[0], feature_shape[1], -1)
        # features_x = features_x.view(feature_shape[0], feature_shape[1], -1)
        # features_y = features_y.view(feature_shape[0], feature_shape[1], -1)

        # var_x = torch.var(features_x, dim=2)
        # var_y = torch.var(features_y, dim=2)
        # loss_var = torch.sum(torch.sum(var_x, dim=1) + torch.sum(var_y, dim=1))

        # loc_x = torch.sum(features_x, dim=2) / torch.sum(features, dim=2)
        # loc_y = torch.sum(features_y, dim=2) / torch.sum(features, dim=2)
        # loss_sep = 0
        
        # for i in range(feature_shape[1]):
        #     for j in range(i+1, feature_shape[1]):
        #         dis_2 = (loc_x[:, i]-loc_x[:, j])*(loc_x[:, i]-loc_x[:, j]) + (loc_y[:, i]-loc_y[:, j])*(loc_y[:, i]-loc_y[:, j])
        #         loss_sep += torch.sum(torch.exp(-dis_2))

        features = features.view(feature_shape)
        loss_ignore = 0
        for ch in range(feature_shape[1]):
            for ignore_id in [7, 8, 21, 23]:
                loss_ignore += torch.sum(torch.abs(features[:,ch][labels==ignore_id])) 
        # loss = loss_var + loss_sep + loss_ignore
        loss = loss_ignore
        return loss
