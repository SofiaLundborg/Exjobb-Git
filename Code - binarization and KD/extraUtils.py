from collections import OrderedDict
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime


def get_device():
    if torch.cuda.is_available():
        return 'cuda2'
    else:
        return 'cpu'


def calculate_output_size(input_size, kernel_size, stride, padding):
    output_size = int((input_size - kernel_size + 2 * padding) / stride + 1)
    return output_size


def change_loaded_checkpoint(checkpoint, student_net):
    student_dict = student_net.state_dict()

    new_checkpoint = OrderedDict()
    for key in checkpoint:
        str_key = key
        str_key = str_key.replace('conv1.', 'conv1.conv2d.')
        str_key = str_key.replace('conv2.', 'conv2.conv2d.')
        str_key = str_key.replace('downsample', 'shortcut')
        str_key = str_key.replace('fc', 'linear')
        #str_key = str_key.replace('.0.shortcut.0.weight', '.0.shortcut.0.conv2d.weight')

        new_checkpoint[str_key] = checkpoint[key]

    for key_student in student_dict:
        if key_student not in new_checkpoint:
            new_checkpoint[key_student] = student_dict[key_student]

    return new_checkpoint


def calculate_accuracy(data_loader, net, topk=(1,5)):
    net.eval()
    n_correct = 0
    n_total = 0
    accuracy1 = AverageMeter()
    accuracy5 = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

            images, targets = data

            if torch.cuda.is_available():
                images = images.to('cuda')
                targets = targets.to('cuda')

            outputs = net(images)
            prec1, prec5 = accuracy(outputs, targets, topk=topk)
            accuracy1.update(prec1[0], images.size(0))
            accuracy5.update(prec5[0], images.size(0))
    if len(topk)>1:
        return accuracy1.avg.item(), accuracy5.avg.item() #100 * n_correct / n_total
    else:
        return accuracy1.avg.item()


def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_results(ax, fig, train_results, validation_results, max_epochs, filename=None, title=None, eps=False):
    ax.clear()
    ax.plot(np.arange(max_epochs+1) + 1, train_results[:max_epochs+1], label='train')
    if validation_results is not None:
        ax.plot(np.arange(max_epochs+1) + 1, validation_results[:max_epochs+1], label='validation')
    ax.legend()

    if title:
        ax.set_title(title)

    if eps:
        if not filename:
            f_name = 'latest_plot.eps'
        else:
            f_name = './Figures/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.eps'
        fig.savefig(f_name, format='eps')
    else:
        if not filename:
            f_name = 'latest_plot.png'
        else:
            f_name = './Figures/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.png'
        fig.savefig(f_name)
