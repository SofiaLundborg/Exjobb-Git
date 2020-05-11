import torch
import torch.nn as nn
from extraUtils import calculate_output_size
from extraUtils import get_device_id


def binarize_weights(net):
    """ Binarizes all parameters with attribute do_binarize == True """
    for p in list(net.parameters()):
        if hasattr(p, 'do_binarize'):
            if p.do_binarize:
                if ((torch.sum(p.data == 1) + torch.sum(p.data == -1)).item() == torch.numel(p.data)):
                    print('Error: Binarizing already binary weights, keeping old weights')
                    p.data.sign_()
                else:
                    p.real_weights = p.data.clone()
                    p.data.sign_()


def set_layers_to_binarize(net, layers):
    """ set which convolutional layers to binarize """

    for p in list(net.parameters()):
        if hasattr(p, 'do_binarize'):
            p.do_binarize = False

    for layer_str in layers:
        layer = getattr(net, layer_str)
        for p in list(layer.parameters()):
            if hasattr(p, 'do_binarize'):
                p.do_binarize = True


def set_layers_to_update(net, layers):
    """ set which layers to apply weight update """

    net.eval()
    for p in list(net.parameters()):
        p.requires_grad = False

    for layer_str in layers:
        layer = getattr(net, layer_str)
        layer.train()
        for p in list(layer.parameters()):
            p.requires_grad = True


def set_layers_to_train_mode(net, layers):
    for layer_str in layers:
        layer = getattr(net, layer_str)
        layer.train()


def make_weights_real(net):
    """ Set all the weigths that have been binarized to their real value version """
    for p in list(net.parameters()):
        if hasattr(p, 'real_weights'):
            p.data.copy_(p.real_weights.clamp_(-1, 1))  # Also clip them


def delete_real_weights(net):
    for p in list(net.parameters()):
        if hasattr(p, 'real_weights'):
            delattr(p, 'real_weights')
            p.data = (p.data+1).bool()


class BinaryActivation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sign()
        result[input == 0] = -1
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # derivative is 1 in [-1, 1] and 0 otherwise
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input >= 1] = 0
        grad_input[input <= -1] = 0
        return grad_input


binarize = BinaryActivation.apply


class myConv2d(nn.Module):
    """ Depending on which net-type, the 2d-conv calculation is done differently. For all net types, the layer updates
    the input size to the output size after every call of myConv2d. """

    def __init__(self, input_channels, output_channels, input_size=None, kernel_size=-1, stride=-1, padding=-1,
                 net_type='full_precision', dropout=0, bias=False, factorized_gamma=False):
        super(myConv2d, self).__init__(),

        self.net_type = net_type
        self.input_size = input_size
        self.output_channels = output_channels
        self.layer_type = self.net_type + '_Conv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.output_channels = output_channels
        self.factorized_gamma = factorized_gamma

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        self.conv2d = nn.Conv2d(input_channels, output_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.conv2d.weight.do_binarize = False

        if input_size is not None:
            new_input_size = calculate_output_size(input_size[0], kernel_size, stride, padding)
            input_size[0] = new_input_size

        if net_type == 'Xnor++':
            if input_size is not None:
                scaling_factor = 1
                output_size = input_size[0]

                if factorized_gamma:
                    self.alpha = torch.nn.Parameter(scaling_factor * torch.ones(output_channels, 1, 1), requires_grad=True)
                    self.beta = torch.nn.Parameter(scaling_factor * torch.ones(1, output_size, 1), requires_grad=True)
                    self.gamma = torch.nn.Parameter(scaling_factor * torch.ones(1, 1, output_size), requires_grad=True)
                    self.gamma_large = torch.mul(torch.mul(self.alpha, self.beta), self.gamma)
                    # gamma_large = torch.einsum('i, j, k -> ijk', self.alpha, self.beta, self.gamma)
                else:
                    self.gamma_large = torch.nn.Parameter(
                        scaling_factor * torch.ones(output_channels, output_size, output_size), requires_grad=True)
            else:
                print('Add input size for layer')

    def forward(self, x):

        if self.dropout_ratio != 0:
            x = self.dropout(x)

        if self.net_type == 'binary':
            if self.conv2d.weight.do_binarize:
                x = binarize(x)
            x = self.conv2d(x)

        if self.net_type == 'Xnor':
            if self.conv2d.weight.do_binarize:

                mean_across_channels = torch.mean(x.abs(), 1, keepdim=True)
                kConv2d = nn.Conv2d(1, self.output_channels,
                                         kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
                kConv2d.weight.data = kConv2d.weight.data.zero_().add(1 / (self.kernel_size * self.kernel_size))
                kConv2d.bias.data = kConv2d.bias.data.zero_()
                kConv2d.weight.requires_grad = False
                kConv2d.bias.requires_grad = False

                if torch.cuda.is_available():
                    kConv2d = kConv2d.cuda(device=get_device_id())

                k = kConv2d(mean_across_channels)

                w = self.conv2d.weight.real_weights.detach()
                alpha_values = torch.mean(w.abs(), [1, 2, 3], keepdim=True).flatten()

                for i in range(self.output_channels):
                    k[:, i, :, :].mul_(alpha_values[i])

                x = binarize(x)
                x = self.conv2d(x)
                x = x*k
            else:
                x = self.conv2d(x)

        if self.net_type == 'binary_with_alpha':
            if self.conv2d.weight.do_binarize:
                x = binarize(x)
                x = self.conv2d(x)
                w = self.conv2d.weight.real_weights
                alpha_values = torch.mean(w.abs(), [1, 2, 3], keepdim=True).flatten()
                alpha_matrix = torch.ones(size=x.size())
                for i in range(self.output_channels):
                    alpha_matrix[:, i, :, :] = alpha_matrix[:, i, :, :]*alpha_values[i]
                x = x * alpha_matrix
            else:
                x = self.conv2d(x)

        if self.net_type == 'Xnor++':
            if self.conv2d.weight.do_binarize:
                x = binarize(x)
                x = self.conv2d(x)
                if self.factorized_gamma:
                    gamma_large = torch.mul(torch.mul(self.alpha, self.beta), self.gamma)
                    x = x*gamma_large
                else:
                    x = x*self.gamma_large
            else:
                x = self.conv2d(x)

        if self.net_type == 'full_precision':
            x = self.conv2d(x)

        return x


class myMaxPool2d(nn.Module):
    """ Same as regular maxPool2d but updates the variable input_size """
    def __init__(self, kernel_size, stride, padding=0, input_size=None):
        super(myMaxPool2d, self).__init__(),

        self.maxPool2d = nn.MaxPool2d(kernel_size, stride, padding)

        if input_size:
            new_input_size = calculate_output_size(input_size[0], kernel_size, stride, padding)
            input_size[0] = new_input_size

    def forward(self, x):
        x = self.maxPool2d(x)
        return x


class myAvgPool2d(nn.Module):
    """ Same as regular avgPool2d but updates the variable input_size """
    def __init__(self, kernel_size, stride, padding=0, input_size=None):
        super(myAvgPool2d, self).__init__(),

        self.avgPool2d = nn.AvgPool2d(kernel_size, stride, padding)

        if input_size:
            new_input_size = calculate_output_size(input_size[0], kernel_size, stride, padding)
            input_size[0] = new_input_size

    def forward(self, x):
        x = self.avgPool2d(x)
        return x
