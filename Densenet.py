import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from torch.jit.annotations import List
import torch.utils.checkpoint as cp
from SENET import SELayer
from CBAM import CBAM
from SKNet import SKLayer
from utils.skeleton_feat import genSkeletons


class _DenseLayer(nn.Module):  # 这个就跟ResNet 的bottleneck一样
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=True, mode=None):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        # if mode == 'cbam':
        #     self.add_module('cbam', CBAM(growth_rate))
        # elif mode == 'se':
        #     self.add_module('se', SELayer(channel=growth_rate)),

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)  # 把outputs 和之前的feature concat起来，
        # 注意这里是先把features concat起来，因为传进来的features是之前的所以的features的一个append的tensor，
        # 这里的forward是在DenseBlock那里进行的，原版其实只有一个forward
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=True,
                 mode=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                mode=mode,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        # self.mode = mode
        # if self.mode == 'se':
        #     self.SELayer = SELayer(num_input_features+num_layers*growth_rate)
        # elif self.mode == 'cbam':
        #     self.CBAMLayer = CBAM()

    def forward(self, init_features):
        features = [init_features]
        # print(self.items())
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        out = torch.cat(features, 1)
        # print(type(out))
        # if self.mode == 'se':
        #     out = self.SELayer(out)
        # elif self.mode == 'cbam':
        #     out = self.CBAMLayer(out)
        return out


class _Transition(nn.Sequential):  # 过渡层，防止channel太多
    def __init__(self, num_input_features, num_output_features, mode=None):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # if mode == 'se':
        #     self.add_module('se', SELayer(channel=num_output_features)),

class KDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1, memory_efficient=True, deep_stem=True,
                 mode=None):
        super(KDenseNet, self).__init__()

        if deep_stem:
            self.features = nn.Sequential(OrderedDict([
                ('conv0_3', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2,
                                      padding=1, bias=False)),
                ('norm0_3', nn.BatchNorm2d(num_init_features)),
                ('relu0_3', nn.ReLU(inplace=True)),

                ('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm1', nn.BatchNorm2d(num_init_features)),
                ('relu1', nn.ReLU(inplace=True)),

                ('conv2', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm2', nn.BatchNorm2d(num_init_features)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # self.features.add_module('se', SELayer(num_init_features))

        num_features = num_init_features + 55
        self.after = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                mode=mode
            )
            self.after.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # if mode == 'se':
            #     self.features.add_module('SELayer%d' % (i + 1), SELayer(num_features))
            # elif mode == 'cbam':
            #     self.features.add_module('CBAMLayer%d' % (i + 1), CBAM(num_features))
            # elif mode == 'sk':
            #     self.features.add_module('SKLayer%d' % (i + 1), SKLayer(num_features, G=1))

            if i != len(block_config) - 1:  # 如果不是最后一层
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, mode=mode)
                self.after.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

                if mode == 'se':
                    self.after.add_module('SELayer%d' % (i + 1), SELayer(num_features))
                elif mode == 'cbam':
                    self.after.add_module('CBAMLayer%d' % (i + 1), CBAM(num_features))
                elif mode == 'sk':
                    self.after.add_module('SKLayer%d' % (i + 1), SKLayer(num_features, G=1))

        self.after.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 2, num_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 4, num_features // 8),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 8, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, kpts=None):
        # 4C
        # if kpts is not None:
        #     x = torch.cat((x, kpts),1)
        features = self.features(x)

        # features = self.after(features)
        if kpts is not None:
            out = self.after(torch.cat((features, kpts), 1))
        out = F.relu(out, inplace=True)

        # out = nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False)(out)
        # out = nn.ReLU()(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out




class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1, memory_efficient=True, deep_stem=False,
                 mode=None):
        super(DenseNet, self).__init__()

        if deep_stem:
            self.features = nn.Sequential(OrderedDict([
                ('conv0_3', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2,
                                      padding=1, bias=False)),
                ('norm0_3', nn.BatchNorm2d(num_init_features)),
                ('relu0_3', nn.ReLU(inplace=True)),

                ('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm1', nn.BatchNorm2d(num_init_features)),
                ('relu1', nn.ReLU(inplace=True)),

                ('conv2', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm2', nn.BatchNorm2d(num_init_features)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # self.features.add_module('se', SELayer(num_init_features))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                mode=mode
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # if mode == 'se':
            #     self.features.add_module('SELayer%d' % (i + 1), SELayer(num_features))
            # elif mode == 'cbam':
            #     self.features.add_module('CBAMLayer%d' % (i + 1), CBAM(num_features))
            # elif mode == 'sk':
            #     self.features.add_module('SKLayer%d' % (i + 1), SKLayer(num_features, G=1))

            if i != len(block_config) - 1:  # 如果不是最后一层
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, mode=mode)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

                if mode == 'se':
                    self.features.add_module('SELayer%d' % (i + 1), SELayer(num_features))
                elif mode == 'cbam':
                    self.features.add_module('CBAMLayer%d' % (i + 1), CBAM(num_features))
                elif mode == 'sk':
                    self.features.add_module('SKLayer%d' % (i + 1), SKLayer(num_features, G=1))

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(num_features, num_features//2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_features // 2, num_features // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(num_features // 4, num_features // 8),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 1),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(num_features, num_features//2, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features // 2, num_features // 4, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features // 4, num_features // 8, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(num_features // 8, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        # out = nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False)(out)
        # out = nn.ReLU()(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        # out = torch.unsqueeze(torch.squeeze(out), 1)
        # print(out.shape)
        return out


def _densenet(arch, growth_rate, block_config, num_init_features, mode,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, mode=mode, **kwargs)
    return model


def Densenet121(mode=None, **kwargs):
    __name__ = 'Densenet121'
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, mode,
                     **kwargs)


def Densenet169(mode=None, **kwargs):
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, mode,
                     **kwargs)


def SEDensenet121(mode='se', **kwargs):
    return _densenet('sedensenet121', 32, (6, 12, 24, 16), 64, mode,
                     **kwargs)


def SEDensenet169(mode='se', **kwargs):
    return _densenet('sedensenet121', 32, (6, 12, 32, 32), 64, mode,
                     **kwargs)


def KDensenet121(model='se',**kwargs):
    return KDenseNet(32, (6, 12, 24, 16), 64, mode=model,**kwargs)


def CBAMDensenet121(mode='cbam', **kwargs):
    return _densenet('cbamdensenet121', 32, (6, 12, 24, 16), 64, mode,
                     **kwargs)


def SKDensenet121(mode='sk',**kwargs):
    return _densenet('skdensenet121', 32, (6, 12, 24, 16), 64, mode,
                     **kwargs)


# print(KDensenet121())