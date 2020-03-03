import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from binaryUtils import *
from extraUtils import copy_parameters, change_loaded_checkpoint
from models import originalResnet, resNet
import distillation_loss
from datetime import datetime
from matplotlib.lines import Line2D


import dataloaders

def load_data():
    # Load data
    normalizing_mean = [0.485, 0.456, 0.406]
    normalizing_std = [0.229, 0.224, 0.225]

    if torch.cuda.is_available():
        batch_size_training = 64
        batch_size_validation = 256
    else:
        batch_size_training = 4
        batch_size_validation = 16

    #normalizing_mean = [0.4914, 0.4822, 0.4465]
    #normalizing_std = [0.2470, 0.2435, 0.2616]

    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalizing_mean, std=normalizing_std)])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalizing_mean, std=normalizing_std)])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

    # divide into train and validation data (80% train)
    train_size = int(0.8 * len(train_set))
    validation_size = len(train_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, validation_size])

    # train_set, ndjkfnskj = torch.utils.data.random_split(train_set, [800, len(train_set)-800])
    # validation_set, ndjkfnskj = torch.utils.data.random_split(validation_set, [500, len(validation_set)-500])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_training,
                                               shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_validation,
                                                    shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_validation,
                                              shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader


def get_data_loaders():
    return dataloaders.CIFAR10DataLoaders.train_loader(batch_size=32), dataloaders.CIFAR10DataLoaders.val_loader()


def get_one_sample(data_loader):
    image, targets = next(iter(data_loader))
    return image


def calculate_accuracy(data_loader, net):
    net.eval()
    n_correct = 0
    n_total = 0
    accuracy1 = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, targets = data
            outputs = net(images)
            prec1 = accuracy(outputs, targets)
            accuracy1.update(prec1[0], images.size(0))

            #if i % 10 == 9:
            #    print('mean accuarcy: ' + str(accuracy1.avg))

            # _, predicted = torch.max(outputs.data, 1)
            # n_total += targets.size(0)
            # n_correct += (predicted == targets).sum().item()

    return accuracy1.avg.item()        #100 * n_correct / n_total


def accuracy(output, target, topk=(1,)):
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


def train_first_layers(start_layer, end_layer, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type):
    set_layers_to_binarize(student_net, start_layer, end_layer)
    set_layers_to_update(student_net, start_layer, end_layer)
    cut_network = end_layer

    criterion = distillation_loss.Loss(0, 0, 0)

    filename = str(start_layer) + '_to_' + str(end_layer) + 'layers_bin_' + str(net_type)
    title = 'loss, ' + str(start_layer) + ' to ' + str(end_layer) + ' layers binarized, ' + str(net_type)
    train_results, validation_results = train_one_block(student_net, train_loader, validation_loader, max_epochs,
                                                        criterion, teacher_net, cut_network=cut_network,
                                                        filename=filename, title=title, accuracy_calc=False)
    min_loss = min(train_results)

    return min_loss


def train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=None,
                    intermediate_layers=None, cut_network=None, filename=None, title=None, accuracy_calc=True):

    optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9)

    train_results = np.empty(max_epochs)
    if accuracy_calc:
        validation_results = np.empty(max_epochs)
    else:
        validation_results = None
    best_validation_accuracy = 0
    lowest_loss = np.inf
    best_epoch = 0

    fig, ax = plt.subplots()

    for epoch in range(max_epochs):  # loop over the data set multiple times
        student_net.train()
        if teacher_net:
            teacher_net.eval()
        running_loss = 0.0
        running_loss_minibatch = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data

            # Cuda
            if torch.cuda.is_available():
                device = 'cuda'
                criterion = criterion.cuda()
            else:
                device = 'cpu'
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            binarize_weights(student_net)

            loss = criterion(inputs, targets, student_net, teacher_net, intermediate_layers, cut_network)
            loss.backward(retain_graph=True)  # calculate loss

            #plot_grad_flow(student_net.named_parameters())

            make_weights_real(student_net)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_minibatch += loss.item()
            #if i % 200 == 199:  # print every 200 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch, i + 1, running_loss_minibatch / 200))
            #    running_loss_minibatch = 0.0

        loss_for_epoch = running_loss / len(train_loader)

        binarize_weights(student_net)

        if accuracy_calc:
            student_net.eval()
            accuracy_train = calculate_accuracy(train_loader, student_net)
            accuracy_validation = calculate_accuracy(validation_loader, student_net)
            train_results[epoch] = accuracy_train
            validation_results[epoch] = accuracy_validation
            print('Accuracy of the network on the train images: %d %%' % accuracy_train)
            print('Accuracy of the network on the validation images: %d %%' % accuracy_validation)
            if accuracy_validation > best_validation_accuracy:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_accuracy = accuracy_validation
                best_epoch = epoch
        else:
            train_results[epoch] = loss_for_epoch
            if lowest_loss > loss_for_epoch:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                lowest_loss = loss_for_epoch
                best_epoch = epoch

        make_weights_real(student_net)

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Epoch loss: ' + str(loss_for_epoch))
        print('Best loss: ' + str(lowest_loss))
        if accuracy_calc:
            print('Best validation accuracy: ' + str(best_validation_accuracy))

        plot_results(ax, fig, train_results, validation_results, epoch+1, filename, title)

    print('Finished Training')

    return train_results, validation_results


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def plot_results(ax, fig, train_results, validation_results, max_epochs, filename=None, title=None):
    # ax.plot(np.arange(max_epochs) + 1, train_results[:max_epochs], label='train')
    ax.plot(np.arange(max_epochs) + 1, train_results[:max_epochs])
    if validation_results:
        ax.plot(np.arange(max_epochs) + 1, validation_results[:max_epochs], label='validation')
    # ax.legend()

    if title:
        ax.set_title(title)

    if not filename:
        f_name = 'latest_plot.eps'
    else:
        f_name = './Figures/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.eps'
    fig.savefig(f_name, format='eps')


def main():
    net_name = 'resnet20'           # 'leNet', 'ninNet', 'resnetX' where X = 20, 32, 44, 56, 110, 1202
    net_type = 'Xnor++'             # 'full_precision', 'binary_with_alpha', 'Xnor' or 'Xnor++'
    max_epochs = 2000
    scaling_factor_total = 0.75     # LIT: 0.75
    scaling_factor_kd_loss = 0.95   # LIT: 0.95
    temperature_kd_loss = 6.0       # LIT: 6.0

    train_loader, validation_loader, test_loader = load_data()

    # initailize_networks
    teacher_net = resNet.resnet_models["cifar"][net_name]('full_precision')
    student_net = resNet.resnet_models["cifar"][net_name](net_type)

    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_net)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    teacher_net.load_state_dict(new_checkpoint_teacher)
    student_net.load_state_dict(new_checkpoint_student)

    if torch.cuda.is_available():
        teacher_net = teacher_net.cuda()
        student_net = student_net.cuda()

    trained_student_checkpoint = torch.load('Trained_Models/1_to_7layers_bin_Xnor++_20200302.pth', map_location='cpu')
    trained_student_net = resNet.resnet_models["cifar"][net_name]('Xnor++')
    trained_student_net.load_state_dict(trained_student_checkpoint)

    criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd_loss, temperature_kd_loss)

    # train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=teacher_net,
    #                intermediate_layers=intermediate_layers, cut_network=None, filename='hejhej', title=None)

    start_layer = 1
    end_layer = 7
    train_first_layers(start_layer, end_layer, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type)


if __name__ == '__main__':
    main()
