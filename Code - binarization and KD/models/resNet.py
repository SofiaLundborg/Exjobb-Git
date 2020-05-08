'''
Resnet Implementation Heavily Inspired by the torchvision resnet implementation
'''

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from binaryUtils import myConv2d, myMaxPool2d
import torch
import math


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def my_conv3x3(in_planes, out_planes, input_size, stride=1, net_type='full_precision', bias=False, factorized_gamma=False):
    """3x3 convolution with padding"""
    return myConv2d(in_planes, out_planes, input_size, kernel_size=3, stride=stride,
                    padding=1, net_type=net_type, bias=bias, factorized_gamma=factorized_gamma)


class BasicBlockForTeacher(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockForTeacher, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'cifar10':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        x = F.relu(x)
        x_to_shortcut = x
        out = self.bn1(self.conv1(x))
        out = F.relu(out)

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)
        i_layer += 1

        out += res_shortcut

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockNaive(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=18, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockNaive, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes
        self.shortcut = nn.Sequential()

        # if stride != 1 or in_planes != planes:
        #     if option == 'cifar10':
        #         self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
        #     else:
        #         self.shortcut = nn.Sequential(
        #              nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        #              nn.BatchNorm2d(self.expansion * planes)
        #         )

        if stride != 1 or in_planes != planes:
            if n_layers == 20:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                    myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                             net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                    nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = x
        else:
            x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = F.relu(out)

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)

        i_layer += 1

        out += res_shortcut

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockWithRelu(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=20, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockWithRelu, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes
        self.input_size_temp = input_size.copy()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if n_layers == 20:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                    myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                             net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = F.relu(x)
        else:
            x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = F.relu(out)

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)

        i_layer += 1

        out += res_shortcut

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockAbs(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=20, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockAbs, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes
        self.input_size_temp = input_size.copy()

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if n_layers == 20:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                    myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                             net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = abs(x) * self.move_average_factor
        else:
            x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = F.relu(out)

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)

        i_layer += 1

        out += res_shortcut

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockAbsDoubleShortcut(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=20, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockAbsDoubleShortcut, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if n_layers == 20:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                    myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                             net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = abs(x) * self.move_average_factor
        else:
            x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = F.relu(out)
        else:
            out_mid = out

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)

        if self.conv1.conv2d.weight.do_binarize:
            out = (out + out_mid)*(1/math.sqrt(2)) + res_shortcut
        else:
            out += res_shortcut

        i_layer += 1

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockNaiveDoubleShortcut(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockNaiveDoubleShortcut, self).__init__()
        self.input_size_temp = input_size.copy()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                myConv2d(in_planes, self.expansion * planes, self.input_size_temp, kernel_size=1, stride=stride, padding=0,
                         net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = x
        else:
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            #out = F.relu(out)
            out = out
        else:
            out_mid = out

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))
        res_shortcut = self.shortcut(x_to_shortcut)

        if self.conv1.conv2d.weight.do_binarize:
            out = (out + out_mid)*(1/math.sqrt(2)) + res_shortcut
        else:
            out += res_shortcut

        i_layer += 1

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockBiReal(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=20, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockBiReal, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride != 1 or in_planes != planes:
                if n_layers == 20:
                    self.shortcut = LambdaLayer(
                        lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                else:
                    self.shortcut = nn.Sequential(
                        myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                                 net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                        nn.BatchNorm2d(self.expansion * planes)
                    )
    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = x
        else:
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = out
            out += self.shortcut(x_to_shortcut)
            x_to_shortcut = out

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))
        res_shortcut = self.shortcut(x_to_shortcut)

        if self.conv1.conv2d.weight.do_binarize:
            #out = (out + out_mid)*(1/math.sqrt(2)) + res_shortcut
            out += res_shortcut
        else:
            out += res_shortcut

        i_layer += 1

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockReluDoubleShortcut(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, n_layers=20, net_type='full_precision', factorized_gamma=False):
        super(BasicBlockReluDoubleShortcut, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes
        self.input_size_temp = input_size.copy()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if n_layers == 20:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                    myConv2d(in_planes, self.expansion * planes, [-1], kernel_size=1, stride=stride, padding=0,
                             net_type=net_type, bias=False, factorized_gamma=factorized_gamma),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, inp):
        x, i_layer, feature_layers_to_extract, features, cut_network = inp

        if cut_network:
            if i_layer > cut_network:
                return inp

        if self.conv1.conv2d.weight.do_binarize:
            x_to_shortcut = F.relu(x)
        else:
            x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if self.conv2.conv2d.weight.do_binarize:
            out_mid = out
        else:
            out = F.relu(out)

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        res_shortcut = self.shortcut(x_to_shortcut)

        if self.conv1.conv2d.weight.do_binarize:
            out = (out + out_mid)*(1/math.sqrt(2)) + res_shortcut
        else:
            out += res_shortcut

        i_layer += 1

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.out_size = planes * 4

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

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

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNetReluFirst(nn.Module):
    def __init__(self, block, layers, net_type='full_precision', dataset="cifar10", num_classes=10, in_planes=None, factorized_gamma=False):
        super(ResNetReluFirst, self).__init__()
        self.dataset = dataset
        self.net_type = net_type
        self.factorized_gamma = factorized_gamma
        self.n_layers = sum(layers)*2 + 2

        if in_planes:
            self.in_planes = in_planes
        elif self.n_layers == 20:
            self.in_planes = 16
        else:
            self.in_planes = 64

        if dataset == "cifar10":
            num_classes = 10
            input_size = [32]
        elif dataset == "ImageNet":
            num_classes = 1000
            input_size = [224]

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        ip = self.in_planes
        if "cifar" in dataset:
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=3, stride=1, padding=1, net_type='full_precision',
                                  bias=False, factorized_gamma=factorized_gamma)
        else:
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=7, stride=2, padding=3, net_type='full_precision',
                                  bias=False, factorized_gamma=factorized_gamma)

        if self.n_layers == 20:
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)
            self.layer4 = None
        else:
            self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)
            self.maxpool = myMaxPool2d(kernel_size=3, stride=2, padding=1, input_size=input_size)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1 = self._make_layer(block, ip, input_size, layers[0], stride=1, net_type=net_type)
        self.layer2 = self._make_layer(block, ip * 2, input_size, layers[1], stride=2, net_type=net_type)
        self.layer3 = self._make_layer(block, ip * 4, input_size, layers[2], stride=2, net_type=net_type)
        if self.n_layers == 18:
            self.layer4 = self._make_layer(block, ip * 8, input_size, layers[3], stride=2, net_type=net_type)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # No update of weights
        for p in list(self.parameters()):
            p.requires_grad = False

    def _make_layer(self, block, planes, input_size, num_blocks, stride, net_type):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, input_size, stride, self.n_layers, net_type, factorized_gamma=self.factorized_gamma))
            if i == 0: self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, feature_layers_to_extract=None, cut_network=None):

        features = OrderedDict()

        out = self.bn1(self.conv1(x))
        i_layer = 1

        if cut_network == i_layer:
            return out

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        if self.layer4:
            out = self.maxpool(out)

        inp = [out, i_layer, feature_layers_to_extract, features, cut_network]
        output = self.layer1(inp)
        i_layer = output[1]
        if cut_network:
            if cut_network <= i_layer:
                return output[0]
        output = self.layer2(output)
        i_layer = output[1]
        if cut_network:
            if cut_network <= i_layer:
                return output[0]

        output = self.layer3(output)
        i_layer = output[1]
        if cut_network:
            if cut_network <= i_layer:
                return output[0]

        out, i_layer, feature_layers_to_extract, features, cut_network = output

        if self.layer4:
            out, i_layer, feature_layers_to_extract, features, cut_network = self.layer4([out, i_layer, feature_layers_to_extract, features, cut_network])
            out = F.relu(out)
            if cut_network:
                if cut_network <= i_layer:
                    return output[0]
            out = self.avgpool(out)
        else:
            out = self.relu(out)
            out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        # Fully connected layer to get to the class
        out = self.linear(out)

        if feature_layers_to_extract:
            # soft_output = out.detach()
            return features, out
        return out


class CifarModel():
    @staticmethod
    def resnet20Naive(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockNaive, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20WithRelu(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockWithRelu, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20Abs(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockAbs, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20AbsDoubleShortcut(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockAbsDoubleShortcut, [3, 3, 3], net_type, dataset=dataset, factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20ReluDoubleShortcut(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockReluDoubleShortcut, [3, 3, 3], net_type, dataset=dataset, factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20ForTeacher(net_type, dataset='cifar10', **kwargs):
        return ResNetReluFirst(BasicBlockForTeacher, [3, 3, 3], net_type, dataset=dataset, **kwargs)
    @staticmethod
    def resnet20NaiveDoubleShortcut(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockNaiveDoubleShortcut, [3, 3, 3], net_type, dataset=dataset, factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20BiReal(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockBiReal, [3, 3, 3], net_type, dataset=dataset, factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet18Naive(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockNaive, [2, 2, 2, 2], net_type, dataset=dataset,
                               factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet18ReluDoubleShortcut(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockReluDoubleShortcut, [2, 2, 2, 2], net_type, dataset=dataset,
                               factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet18ForTeacher(net_type, dataset='cifar10', **kwargs):
        return ResNetReluFirst(BasicBlockForTeacher, [2, 2, 2, 2], net_type, dataset=dataset, **kwargs)


resnet_models = {
        "resnet20Naive": CifarModel.resnet20Naive,
        "resnet20WithRelu": CifarModel.resnet20WithRelu,
        "resnet20Abs": CifarModel.resnet20Abs,
        "resnet20AbsDoubleShortcut": CifarModel.resnet20AbsDoubleShortcut,
        "resnet20ReluDoubleShortcut": CifarModel.resnet20ReluDoubleShortcut,
        "resnet20NaiveDoubleShortcut": CifarModel.resnet20NaiveDoubleShortcut,
        "resnet20ForTeacher": CifarModel.resnet20ForTeacher,
        "resnet20BiReal": CifarModel.resnet20BiReal,
        "resnet18Naive": CifarModel.resnet18Naive,
        "resnet18ReluDoubleShortcut": CifarModel.resnet18ReluDoubleShortcut,
        "resnet18ForTeacher": CifarModel.resnet18ForTeacher,
}
