from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv   = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn     = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Residual_Block(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Residual_Block, self).__init__()
        self.conv       = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw    = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project    = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual   = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Residual_Block(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv   = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn     = BatchNorm2d(out_c)
        self.prelu  = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        # 112,112,3 -> 56,56,64
        self.conv1      = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        # 56,56,64 -> 56,56,64
        self.conv2_dw   = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        # 56,56,64 -> 28,28,64
        self.conv_23    = Residual_Block(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3     = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # 28,28,64 -> 14,14,128
        self.conv_34    = Residual_Block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4     = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # 14,14,128 -> 7,7,128
        self.conv_45    = Residual_Block(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5     = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.sep        = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.sep_bn     = nn.BatchNorm2d(512)
        self.prelu      = nn.PReLU(512)

        self.GDC_dw     = nn.Conv2d(512, 512, kernel_size=7, bias=False, groups=512)
        self.GDC_bn     = nn.BatchNorm2d(512)

        self.features   = nn.Conv2d(512, embedding_size, kernel_size=1, bias=False)
        self.last_bn    = nn.BatchNorm2d(embedding_size)
  
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)

        x = self.sep(x)
        x = self.sep_bn(x)
        x = self.prelu(x)
        
        x = self.GDC_dw(x)
        x = self.GDC_bn(x)

        x = self.features(x)
        x = self.last_bn(x)
        return x


def get_mbf(embedding_size, pretrained):
    if pretrained:
        raise ValueError("No pretrained model for mobilefacenet")
    return MobileFaceNet(embedding_size)
