import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


#
#
# class BasicBlock(nn.Module):
#     def __init__(self,inplanes,planes,stride=1):
#         super(BasicBlock,self).__init__()
#         self.conv1 = conv3x3(inplanes,planes,stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         if inplanes != planes:  # 1*1卷积 改变通道数,防止跳连接时channel不同
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
#                 nn.BatchNorm2d(planes)
#             )
#         else: # 通道相同的直接连就可以了
#             self.downsample = lambda x:x
#
#         self.stride = stride
#
#     def forward(self,x):
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#         return out
#
# class PreActBasicBlock(BasicBlock): # 继承BasicBlock
#     def __init__(self,inplanes,planes,stride):
#         super(PreActBasicBlock,self).__init__(inplanes,planes,stride)
#         if inplanes != planes:
#             self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride, bias=False))
#         else:
#             self.downsample = lambda x: x
#         self.bn1 = nn.BatchNorm2d(inplanes)
#
#     def forward(self,x):
#         residual = self.downsample(x)
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#
#         out += residual
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self,block,n_size,num_classes=10):
#         super(ResNet,self).__init__()
#         self.inplane = 16
#         self.conv1 = nn.Conv2d(3,self.inplane,kernel_size=3,stride=1,padding=1,bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplane)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.layer1 = self._make_layer(block, 16, n_size,1)
#         self.layer2 = self._make_layer(block, 32, n_size, 2)
#         self.layer3 = self._make_layer(block, 64, n_size, 2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)  # H*W -> 1*1
#         self.fc = nn.Linear(64,num_classes)
#
#         self.initialize()
#
#     def initialize(self):  # 卷积层和标准化层的初始化
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m,nn.BatchNorm2d):
#                 nn.init.constant_(m.weight,1)
#                 nn.init.constant_(m.bias,0)
#
#     def _make_layer(self,block,planes,blocks,stride): # block--Basicblock  planes--channel，blocks--block的数量，stride--步长
#         strides = [stride] + [1]*(blocks - 1)
#         # [stride , 1 ,1,1,1,...]  数组中除了第一个是stride ，其他stride 都是 1
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inplane,planes,stride))
#             self.inplane = planes   # 第一次之后channel都不变
#
#         return nn.Sequential(*layers)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0),-1)
#         x = self.fc(x)
#
#         return x
#
# def resnet110(**kwargs):
#     model = ResNet(BasicBlock, 18, **kwargs)
#     return model

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):  # channel一般是1
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # // 结果取整
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # 把结果限制到0-1之间，并将output作为scale乘到U的C个通道上
        )

    def forward(self, x):
        # print(type(x))
        b, c, _, _ = x.size()  # batchsize*C*H*W
        y = self.avg_pool(x).view(b, c)  # 经过一个avgpooling层之后 H*W -> 1*1
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # 把 y的形状变成和x一样，即把y扩展到每一个channel，作为一个scale与x相乘


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group=1, base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):  # 大网络用这个
    expansion = 4  #

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

