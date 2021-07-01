import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, backbone):
        super(CNNModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(1000, 7)

    def forward(self, x, with_features=False):
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size * seq_len, *x.shape[2:])
        poses, features = self.backbone(x)
        poses = self.fc(poses)
        poses = poses.view(batch_size, seq_len, -1)
        if(isinstance(features, tuple)):
            features_pred = features[-2].view(batch_size, seq_len, *features[-2].shape[1:])
        else:
            features_pred = features.view(batch_size, seq_len, *features.shape[1:])
        return poses, features_pred