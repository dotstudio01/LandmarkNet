'''FeaturePyramidNetwork in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet50, resnet101

class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        weight = self.bn_upsample_1(self.conv_upsample_1(x1_merge))
        x_master = x_master * self.relu(weight)

        self.weight = weight
        #
        out = self.relu(x_master + x_gpb)

        return out

class PyramidAttentioNetwork(nn.Module):
    def __init__(self, num_classes, h, w, backbone_arch='resnet50', pretrained=True):
        super(PyramidAttentioNetwork, self).__init__()
        self.num_classes = num_classes
        self.h = h
        self.w = w

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        BatchNorm = nn.BatchNorm2d
        
        if backbone_arch == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
        elif backbone_arch == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.fpa = FPA(channels=1024)

        # pose branch
        self.pose_branch_conv = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)
        )
        self.pose_branch_fc = nn.Linear(16 * int(h * w / 16 / 16), 7)

        # Semantic Branch
        self.semantic_branch = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        )
    
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        # Bottom-up using backbone
        low_level_features = self.backbone(x)
        c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
        c5 = low_level_features[4]

        c4 = self.fpa(c4)
        
        self.feature = self.fpa.weight
        # Pose
        pose_feature = self.pose_branch_conv(c4)
        pose_feature = pose_feature.view(pose_feature.shape[0], -1)
        pose = self.pose_branch_fc(pose_feature)

        # Semantic
        semantic = self._upsample(self.semantic_branch(c5), self.h, self.w)

        return pose, semantic

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    model = PyramidAttentioNetwork(32, 128, 384, backbone_arch="resnet50")
    input = torch.rand(2, 3, 128, 384)
    output = model(input)
    print(output[1].size())