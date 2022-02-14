#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.arcface import Arcface

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Arcface(num_classes=10575, backbone="mobilenetv1", mode="predict").to(device)
    summary(model, input_size=(3, 112, 112))
    