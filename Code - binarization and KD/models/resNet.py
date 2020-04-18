'''
Resnet Implementation Heavily Inspired by the torchvision resnet implementation
'''

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from binaryUtils import myConv2d
import matplotlib.pyplot as plt
import torch
import math

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def my_conv3x3(in_planes, out_planes, input_size, stride=1, net_type='full_precision', bias=False, factorized_gamma=False):
    """3x3 convolution with padding"""
    return myConv2d(in_planes, out_planes, input_size, kernel_size=3, stride=stride,
                    padding=1, net_type=net_type, bias=bias, factorized_gamma=factorized_gamma)


class BasicBlock(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision'):
        super(BasicBlock, self).__init__()
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
        out = F.relu(self.bn1(self.conv1(x)))
        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


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



        #if self.conv2.conv2d.weight.do_binarize:  # divide all values less than 0 by 2 to be similar to relu-addition
        #    out[out < 0] = out[out < 0]*0.5

        # if plot:
        #     ax_combined.hist(out.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color, label='relu')
        #     ax_combined.hist(out_abs.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_abs, label='abs*0.6')
        #     ax_combined.hist(out_no_relu.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_no_relu, label='none')
        #     ax_combined.legend(frameon=False)
        #     plt.show()

        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()

        return [out, i_layer, feature_layers_to_extract, features, cut_network]


class BasicBlockReluFirst(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockReluFirst, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

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

        if self.conv1.conv2d.weight.do_binarize:
            x_abs = torch.abs(x)*self.move_average_factor
            # x_no_relu = x
            # x = F.relu(x)

            x_to_shortcut = x_abs
        else:
            x = F.relu(x)
            #x_to_shortcut = torch.abs(x)
            # x = F.relu(x)
            x_to_shortcut = x

        out = self.bn1(self.conv1(x))

        if not self.conv2.conv2d.weight.do_binarize:
            out = F.relu(out)

        if self.conv2.conv2d.weight.do_binarize:
            # out_mid = torch.abs(out)/2
            # out_mid = self.shortcut(x_to_shortcut)/2
            out_mid = out

        i_layer += 1
        if cut_network:
            if cut_network == i_layer:
                return [out, i_layer, feature_layers_to_extract, features, cut_network]

        out = self.bn2(self.conv2(out))

        if self.conv2.conv2d.weight.do_binarize:
            # res_shortcut = (out_mid + self.shortcut(x_to_shortcut) / 2)
            res_shortcut = self.shortcut(x_to_shortcut)
        else:
            res_shortcut = self.shortcut(x_to_shortcut)
            #res_shortcut = (out_mid + self.shortcut(x_to_shortcut) / 2)


        if self.conv2.conv2d.weight.do_binarize:
            res_shortcut = res_shortcut * self.move_average_factor

        # res_shortcut_abs = out_mid_abs + self.shortcut(x_abs)/2
        # res_shortcut_no_relu = out_mid + self.shortcut(x_no_relu)
        i_layer += 1

        #if i_layer == 19:
        #    fig, ax = plt.subplots()
        #    ax.hist(res_shortcut.view(-1), 50, alpha=0.4, histtype='stepfilled', density=True, label='shortcut')
        #    plt.show()
        #     plot = False
        # else:
        #     plot = False
        # if plot:
        #     fig, (ax_shortcut, ax_out, ax_combined) = plt.subplots(1, 3, figsize=(11, 3))
        #     ax_shortcut.set_title('Shortcut')
        #     ax_out.set_title('Output before addition')
        #     ax_combined.set_title('Output after addition')
        #
        #     alpha = 0.25
        #     color = 'black'
        #     color_abs = 'green'
        #     color_no_relu = 'blue'
        #     ax_out.hist(out.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color, label='relu')
        #     ax_shortcut.hist(res_shortcut_abs.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_abs, label='abs*0.6')
        #     ax_shortcut.hist(res_shortcut.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color, label='relu')
        #     ax_shortcut.hist(res_shortcut_no_relu.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_no_relu, label='none')
        #     ax_shortcut.legend(frameon=False)

        # out_abs = out + res_shortcut_abs
        # out_no_relu = out + res_shortcut_no_relu
        #out_abs[out_abs > 0] = out_abs[out_abs > 0]

        if self.conv2.conv2d.weight.do_binarize:
            # plt.hist(out.view(-1), 50, alpha=0.4, histtype='stepfilled', density=True, label='before', color='green')
            # plt.hist(((out/2 + out_mid/2)*2).view(-1), 50, alpha=0.4, histtype='stepfilled', density=True, label='before', color='blue')
            # plt.hist(out_mid.view(-1), 50, alpha=0.4, histtype='stepfilled', density=True, label='before', color='red')


            # out = res_shortcut + out + out_mid
            out = res_shortcut + out

        else:
            out += res_shortcut



        #if self.conv2.conv2d.weight.do_binarize:  # divide all values less than 0 by 2 to be similar to relu-addition
        #    out[out < 0] = out[out < 0]*0.5

        # if plot:
        #     ax_combined.hist(out.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color, label='relu')
        #     ax_combined.hist(out_abs.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_abs, label='abs*0.6')
        #     ax_combined.hist(out_no_relu.view(-1), 50, alpha=alpha, histtype='stepfilled', density=True, color=color_no_relu, label='none')
        #     ax_combined.legend(frameon=False)
        #     plt.show()

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

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockNaive, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

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

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockWithRelu, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
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

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockAbs, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

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

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision'):
        super(BasicBlockAbsDoubleShortcut, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.move_average_factor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

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


class BasicBlockReluDoubleShortcut(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, input_size, stride=1, option='cifar10', net_type='full_precision', factorized_gamma=False):
        super(BasicBlockReluDoubleShortcut, self).__init__()
        self.conv1 = my_conv3x3(in_planes, planes, input_size, net_type=net_type, stride=stride, bias=False, factorized_gamma=factorized_gamma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = my_conv3x3(planes, planes, input_size, net_type=net_type, bias=False, factorized_gamma=factorized_gamma)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_size = planes

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'cifar10':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            else:
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     # myConv2d(in_planes, self.expansion*planes, input_size, kernel_size=1, stride=stride, bias=False, padding=0, net_type='full_precision'),
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


class ResNet1layer(nn.Module):
    def __init__(self, block, layers, net_type='full_precision', dataset="cifar10", num_classes=10, in_planes=None):
        super(ResNet1layer, self).__init__()
        self.dataset = dataset
        self.net_type = net_type

        if in_planes:
            self.in_planes = in_planes
        elif "cifar" in dataset:
            self.in_planes = 16
        else:
            self.in_planes = 64

        if dataset == "cifar10":
            num_classes = 10
            input_size = [32]
        elif dataset == "imagenet":
            num_classes = 1000
            input_size = [224]

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        ip = self.in_planes
        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=3, stride=1, padding=1, net_type=net_type,
                                  bias=False)
            self.layer4 = None
        else:
            self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)
            self.conv1 = nn.Conv2d(3, ip, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer4 = self._make_layer(block, ip * 8, layers[3], input_size, stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)

        self.layer1 = self._make_layer(block, ip, input_size, layers[0], stride=1, net_type=net_type)
        self.layer2 = self._make_layer(block, ip * 2, input_size, layers[1], stride=2, net_type=net_type)
        self.layer3 = self._make_layer(block, ip * 4, input_size, layers[2], stride=2, net_type=net_type)

        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)

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
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, input_size, stride, self.dataset, net_type))
            if i == 0: self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, feature_layers_to_extract=None, cut_network=None):

        features = OrderedDict()

        out = self.relu(self.bn1(self.conv1(x)))
        i_layer = 1

        output = self.layer1([out, i_layer, feature_layers_to_extract, features, cut_network])
        return output[0]


class ResNetReluFirst(nn.Module):
    def __init__(self, block, layers, net_type='full_precision', dataset="cifar10", num_classes=10, in_planes=None, factorized_gamma=False):
        super(ResNetReluFirst, self).__init__()
        self.dataset = dataset
        self.net_type = net_type
        self.factorized_gamma = factorized_gamma

        if in_planes:
            self.in_planes = in_planes
        elif "cifar" in dataset:
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
        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=3, stride=1, padding=1, net_type='full_precision', bias=False, factorized_gamma=factorized_gamma)
            self.layer4 = None
        else:
            self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)
            print('input size: ' + str(input_size))
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=7, stride=2, padding=3, net_type='full_precision', bias=False, factorized_gamma=factorized_gamma)
            print('input size: ' + str(input_size))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1 = self._make_layer(block, ip, input_size, layers[0], stride=1, net_type=net_type)
        self.layer2 = self._make_layer(block, ip * 2, input_size, layers[1], stride=2, net_type=net_type)
        self.layer3 = self._make_layer(block, ip * 4, input_size, layers[2], stride=2, net_type=net_type)

        if "ImageNet" in dataset:
            self.layer4 = self._make_layer(block, ip * 8, input_size, layers[3], stride=2, net_type=net_type)

        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)

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
            layers.append(block(self.in_planes, planes, input_size, stride, self.dataset, net_type, factorized_gamma=self.factorized_gamma))
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
            out = self.relu(out)
            #out = F.relu(out)
            out = self.avgpool(out)
        else:
            out = self.relu(out)
            # out = F.relu(out)
            out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        # Fully connected layer to get to the class
        out = self.linear(out)

        if feature_layers_to_extract:
            # soft_output = out.detach()
            return features, out
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, net_type='full_precision', dataset="cifar10", num_classes=10, in_planes=None):
        super(ResNet, self).__init__()
        self.dataset = dataset
        self.net_type = net_type

        if in_planes:
            self.in_planes = in_planes
        elif "cifar" in dataset:
            self.in_planes = 16
        else:
            self.in_planes = 64

        if dataset == "cifar10":
            num_classes = 10
            input_size = [32]
        elif dataset == "imagenet":
            num_classes = 1000
            input_size = [224]

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        ip = self.in_planes
        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)
            self.conv1 = myConv2d(3, ip, input_size, kernel_size=3, stride=1, padding=1, net_type=net_type, bias=False)
            self.layer4 = None
        else:
            input_size_at_start = input_size
            self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)
            self.conv1 = nn.Conv2d(3, ip, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer4 = self._make_layer(block, ip * 8, 4, layers[3], stride=2, net_type=net_type)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            input_size = input_size_at_start

        self.layer1 = self._make_layer(block, ip, input_size, layers[0], stride=1, net_type=net_type)
        self.layer2 = self._make_layer(block, ip * 2, input_size, layers[1], stride=2, net_type=net_type)
        self.layer3 = self._make_layer(block, ip * 4, input_size, layers[2], stride=2, net_type=net_type)

        if ("cifar" in dataset) or ("svhn" in dataset):
            self.linear = nn.Linear(ip * 4 * block.expansion, num_classes)

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
            layers.append(block(self.in_planes, planes, input_size, stride, self.dataset, net_type))
            if i == 0: self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, feature_layers_to_extract=None, cut_network=None):

        features = OrderedDict()

        out = self.relu(self.bn1(self.conv1(x)))

        i_layer = 1

        if cut_network == i_layer:
            return out

        if feature_layers_to_extract:
            if i_layer in feature_layers_to_extract:
                features[i_layer] = out.detach()
        # else:
        #     features = 1

        if self.layer4: out = self.maxpool(out)

        inp = [out, i_layer, feature_layers_to_extract, features, cut_network]
        output = self.layer1([out, i_layer, feature_layers_to_extract, features, cut_network])
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
            out = self.avgpool(out)
        else:
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
    def resnet20(net_type, dataset='cifar10', **kwargs):
        return ResNet(BasicBlock, [3, 3, 3], net_type, dataset, **kwargs)
    @staticmethod
    def resnet20relufirst(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockReluFirst, [3, 3, 3], net_type, dataset=dataset, factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20Naive(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockNaive, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20WithRelu(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockWithRelu, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet20Abs(net_type, dataset='cifar10', factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockAbs, [3, 3, 3], net_type, dataset=dataset,factorized_gamma=factorized_gamma,  **kwargs)
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
    def resnet18ReluDoubleShortcut(net_type, dataset, factorized_gamma=False, **kwargs):
        return ResNetReluFirst(BasicBlockReluDoubleShortcut, [2, 2, 2, 2], net_type, dataset=dataset,
                               factorized_gamma=factorized_gamma, **kwargs)
    @staticmethod
    def resnet18ForTeacher(net_type, dataset, **kwargs):
        return ResNetReluFirst(BasicBlockForTeacher, [2, 2, 2, 2], net_type, dataset=dataset, **kwargs)

    @staticmethod
    def resnet32(net_type, **kwargs):
        return ResNet(BasicBlock, [5, 5, 5], net_type, **kwargs)
    @staticmethod
    def resnet44(net_type, **kwargs):
        return ResNet(BasicBlock, [7, 7, 7], net_type, **kwargs)
    @staticmethod
    def resnet56(net_type, **kwargs):
        return ResNet(BasicBlock, [9, 9, 9], net_type, **kwargs)
    @staticmethod
    def resnet110(net_type, **kwargs):
        return ResNet(BasicBlock, [18, 18, 18], net_type, **kwargs)
    @staticmethod
    def resnet1202(net_type, **kwargs):
        return ResNet(BasicBlock, [200, 200, 200], net_type,  **kwargs)


resnet_models = {
        "resnet20": CifarModel.resnet20,
        "resnet20relufirst": CifarModel.resnet20relufirst,
        "resnet20Naive": CifarModel.resnet20Naive,
        "resnet20WithRelu": CifarModel.resnet20WithRelu,
        "resnet20Abs": CifarModel.resnet20Abs,
        "resnet20AbsDoubleShortcut": CifarModel.resnet20AbsDoubleShortcut,
        "resnet20ReluDoubleShortcut": CifarModel.resnet20ReluDoubleShortcut,
        "resnet20ForTeacher": CifarModel.resnet20ForTeacher,
        "resnet18ReluDoubleShortcut": CifarModel.resnet18ReluDoubleShortcut,
        "resnet18ForTeacher": CifarModel.resnet18ForTeacher,
        "resnet32": CifarModel.resnet32,
        "resnet44": CifarModel.resnet44,
        "resnet56": CifarModel.resnet56,
        "resnet110": CifarModel.resnet110,
        "resnet1202": CifarModel.resnet1202
}


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))
