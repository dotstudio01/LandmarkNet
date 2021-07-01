import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, h, w, dropout_rate=0.5):
        super(ResNet, self).__init__()

        base_model = models.resnet18(pretrained=False)

        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model[-2].register_forward_hook(self.feature_callback)

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        
        self.relu = nn.ReLU(inplace=False)

        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc_pose = nn.Linear(2048, 7, bias=True)

        init_modules = [self.fc_last, self.fc_pose]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # nn.init.normal_(self.fc_last.weight, 0, 0.01)
        # nn.init.constant_(self.fc_last.bias, 0)
        #
        # nn.init.normal_(self.fc_position.weight, 0, 0.5)
        # nn.init.constant_(self.fc_position.bias, 0)
        #
        # nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
        # nn.init.constant_(self.fc_rotation.bias, 0)

    def feature_callback(self, model, input, output):
        self.feature = output

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size * seq_len, *x.shape[2:])
        out = self.base_model(x)
        out = out.view(out.size(0), -1)
        out = self.fc_last(out)

        out = self.relu(out)
        out = self.dropout(out)
        abs_pose = self.fc_pose(out)

        rel_pose = None
        feature = self.feature

        abs_pose = abs_pose.view(batch_size, seq_len, -1)
        feature = feature.view(batch_size, seq_len, *feature.shape[1:])
        return rel_pose, abs_pose, feature
