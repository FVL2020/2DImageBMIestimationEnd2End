from SENET import SEBottleneck
from CBAM import CBAMBottleneck
# from seresnext import seresnext101_32x8d
import torch.nn as nn
from Densenet import *
from SKNet import SKNet
import torchvision.models as models



def SKNet101(num_classes=1):
    model = SKNet(num_classes, [3, 4, 23, 3])
    return model


def AlexNet(num_classes=1):
    model = models.alexnet(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
    )
    return model


def VGG16(num_classes=1):
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, num_classes),
    )
    return model


def GoogleNet(num_classes=1):
    model = models.googlenet(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes)
    )
    return model


def MobileNet(num_classes=1):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes)
    )
    return model


def Densenet121(num_classes=1):
    model = models.densenet121(pretrained=True)
    model.classifier =  nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes)
        # nn.Linear(1024, num_classes)
    )
    return model

def Resnext101(num_classes=1):
    model = models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 128),
        nn.ReLU(True),
        nn.Linear(128, num_classes),
    )
    return model


def Resnext50(num_classes=1):
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048,512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, num_classes),
    )
    return model



# 计算 flops 和 parmas
# print(Resnet101())
# Nets = [SEDensenet121(), AlexNet(),MobileNet(), VGG16(), GoogleNet(), Resnext50(), Resnet101()]
# Nets_names = ['SEDensenet121()', 'AlexNet()', 'MobileNet()','VGG16()', 'GoogleNet()', 'Resnext50()', 'Resnet101()']
# input = torch.randn(1, 3, 224, 224)
# for Net, Name in zip(Nets,Nets_names):
#     num_params = 0
#     for param in Net.parameters():
#         num_params += param.numel()
#     print(Name, num_params / 1e6)
#     flops, params = profile(Net, inputs=(input,))
#     print(Name, flops / 1e9, params / 1e6)


# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(Densenet121(), inputs=(input, ))
# print('Densenet:', flops/1e9, params/1e6)
# print('alexnet: ', flops/1e6, params/1e6)
# flops, params = profile(models.alexnet(), inputs=(input, ))
# print('AlexNet: ', flops/1e6, params/1e6)
#
# print(models.alexnet())
# print(AlexNet())