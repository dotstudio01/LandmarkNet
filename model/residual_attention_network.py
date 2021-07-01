import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from .attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar


class ResidualAttentionModel_92(nn.Module):
    # for input size 224
    def __init__(self, h, w, dropout_rate=0.5):
        super(ResidualAttentionModel_92, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 32)
        self.attention_module1 = AttentionModule_stage1(32, 32, size1=(
            int(h/4), int(w/4)), size2=(int(h/8), int(w/8)), size3=(int(h/16), int(w/16)))
        self.residual_block2 = ResidualBlock(32, 64, 2)
        self.attention_module2 = AttentionModule_stage2(64, 64, size1=(int(h/8), int(w/8)), size2=(int(h/16), int(w/16)))
        self.attention_module2_2 = AttentionModule_stage2(64, 64, size1=(int(h/8), int(w/8)), size2=(int(h/16), int(w/16)))

        # lstm branch
        self.lstm = nn.LSTM(int(2 * 64 * w * h / 8 / 8), hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)
        self.l_fc = nn.Linear(256, 7)
        # maybe fully connected layers needed here
        
        # residual block branch
        self.residual_block3 = ResidualBlock(64, 128, 2)
        self.attention_module3 = AttentionModule_stage3(128, 128, size1=(int(h/16), int(w/16)))
        self.attention_module3_2 = AttentionModule_stage3(128, 128, size1=(int(h/16), int(w/16)))
        self.attention_module3_3 = AttentionModule_stage3(128, 128, size1=(int(h/16), int(w/16)))
        self.residual_block4 = ResidualBlock(128, 256, 2)
        self.residual_block5 = ResidualBlock(256, 256)
        self.residual_block6 = ResidualBlock(256, 256)

        self.residual_branch_fc = nn.Linear(256 * int(w * h / 32 / 32), 1024)

        self.dropout = nn.Dropout(dropout_rate)

        # final fully connected layer
        self.r_fc = nn.Linear(1024, 7)


    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size * seq_len, *x.shape[2:])
        # import pdb
        # pdb.set_trace()
        out = self.conv1(x)
        out = self.mpool1(out)

        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)

        # 保存feature
        feature = out.view(batch_size, seq_len, *out.shape[1:])
        # lstm branch
        l_out = feature.view(batch_size, seq_len // 2, -1)
        l_out, _ = self.lstm(l_out)
        l_out = self.l_fc(l_out)
        rel_pose = l_out.reshape(batch_size, seq_len // 2, -1)

        # residual attention branch
        r_out = feature.view(batch_size * seq_len, *feature.shape[2:])

        r_out = self.residual_block3(r_out)

        r_out = self.attention_module3(r_out)
        r_out = self.attention_module3_2(r_out)
        r_out = self.attention_module3_3(r_out)
        r_out = self.residual_block4(r_out)
        r_out = self.residual_block5(r_out)
        r_out = self.residual_block6(r_out)
        r_out = r_out.view(r_out.shape[0], -1)
        r_out = self.residual_branch_fc(r_out)
        r_out = self.dropout(r_out)
        r_out = self.r_fc(r_out)
        abs_pose = r_out.view(batch_size, seq_len, -1)

        return rel_pose, abs_pose, feature
