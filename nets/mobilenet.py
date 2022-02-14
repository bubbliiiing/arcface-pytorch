import torch
import torch.nn as nn


def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, dropout_keep_prob, embedding_size, pretrained):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 1),    # 3
            conv_dw(32, 64, 1),   # 7

            conv_dw(64, 128, 2),  # 11
            conv_dw(128, 128, 1),  # 19

            conv_dw(128, 256, 2),  # 27
            conv_dw(256, 256, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),  # 43 + 16 = 59
            conv_dw(512, 512, 1), # 59 + 32 = 91
            conv_dw(512, 512, 1), # 91 + 32 = 123
            conv_dw(512, 512, 1), # 123 + 32 = 155
            conv_dw(512, 512, 1), # 155 + 32 = 187
            conv_dw(512, 512, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2), # 219 +3 2 = 241
            conv_dw(1024, 1024, 1), # 241 + 64 = 301
        )

        self.bn2        = nn.BatchNorm2d(1024, eps=1e-05)
        self.dropout    = nn.Dropout(p=dropout_keep_prob, inplace=True)
        self.linear     = nn.Linear(1024 * self.fc_scale, embedding_size)
        self.features   = nn.BatchNorm1d(embedding_size, eps=1e-05)
        if pretrained:
            self.load_state_dict(torch.load("model_data/mobilenet_v1_weights.pth"), strict = False)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.features(x)
        return x

def get_mobilenet(dropout_keep_prob, embedding_size, pretrained):
    return MobileNetV1(dropout_keep_prob, embedding_size, pretrained)
