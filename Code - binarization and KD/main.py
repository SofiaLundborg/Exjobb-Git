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

import dataloaders

def load_data():
    # Load data
    normalizing_mean = [0.485, 0.456, 0.406]
    normalizing_std = [0.229, 0.224, 0.225]

    #normalizing_mean = [0.4914, 0.4822, 0.4465]
    #normalizing_std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64,
                                                    shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256,
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


def train_first_layers(n_layers, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type):
    set_layers_to_binarize(student_net, [1, n_layers])
    set_layers_to_update(student_net, [1, n_layers])
    cut_network = n_layers

    criterion = distillation_loss.Loss(0, 0, 0)

    filename = str(n_layers) + 'layers_bin_' + str(net_type)
    title = 'loss, ' + str(n_layers) + ' layers binarized, ' + str(net_type)
    train_results, validation_results = train_one_block(student_net, train_loader, validation_loader, max_epochs,
                                                        criterion, teacher_net, cut_network=cut_network,
                                                        filename=filename, title=title, accuracy_calc=False)
    min_loss = min(train_results)

    return min_loss


def train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=None,
                    intermediate_layers=None, cut_network=None, filename=None, title=None, accuracy_calc=True):

    # criterion = nn.CrossEntropyLoss()
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
        running_loss = 0.0
        running_loss_minibatch = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
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
            loss.backward()  # calculate loss

            make_weights_real(student_net)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_minibatch += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 200))
                running_loss_minibatch = 0.0

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
                PATH = './Trained Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_accuracy = accuracy_validation
                best_epoch = epoch
        else:
            train_results[epoch] = loss_for_epoch
            if lowest_loss > loss_for_epoch:
                # save network
                PATH = './Trained Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                lowest_loss = loss_for_epoch
                best_epoch = epoch

        make_weights_real(student_net)

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Best loss: ' + str(lowest_loss))
        if accuracy_calc:
            print('Best validation accuracy: ' + str(best_validation_accuracy))

        plot_results(ax, fig, train_results, validation_results, epoch+1, filename, title)

    print('Finished Training')

    return train_results, validation_results


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
    net_type = 'Xnor'               # 'full_precision', 'binary_with_alpha', 'Xnor' or 'Xnor++'
    max_epochs = 200
    scaling_factor_total = 0.75     # LIT: 0.75
    scaling_factor_kd_loss = 0.95   # LIT: 0.95
    temperature_kd_loss = 6.0       # LIT: 6.0

    # train_loader, validation_loader = get_data_loaders()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    train_loader, validation_loader, test_loader = load_data()
    # train_loader, validation_loader = get_data_loaders()
    # test_loader = validation_loader

    # load pre trained teacher network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'

    teacher_net_org = originalResnet.resnet_models["cifar"][net_name]()
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    teacher_net_org.load_state_dict(teacher_checkpoint)
    teacher_net = resNet.resnet_models["cifar"][net_name]('full_precision')

    new_teacher_checkpoint = change_loaded_checkpoint(teacher_checkpoint)

    teacher_net.load_state_dict(new_teacher_checkpoint)
    #teacher_net = originalResnet.resnet_models["cifar"][net_name]()

    # initialize student network as the teacher network
    # student_net = originalResnet.resnet_models["cifar"][net_name]()
    student_net = resNet.resnet_models["cifar"][net_name](net_type)
    student_net.load_state_dict(new_teacher_checkpoint)
    #copy_parameters(student_net, teacher_net)

    sample_batch = get_one_sample(test_loader)
    sample_batch = sample_batch.to(device)

    if torch.cuda.is_available():
        teacher_net_org = teacher_net_org.cuda()
        teacher_net = teacher_net.cuda()
        student_net = student_net.cuda()

    teacher_net_org.eval()
    teacher_net.eval()
    student_net.eval()

    out_org = teacher_net_org(sample_batch)
    out_teach = teacher_net(sample_batch)
    out_stud = student_net(sample_batch)



    #set_layers_to_binarize(student_net, [1, 1])
    #set_layers_to_update(student_net, [1, 1])
    intermediate_layers = [1]
    cut_network = 2
    out = student_net(sample_batch, feature_layers_to_extract=None, cut_network=cut_network)
    # out = student_net(sample_batch, feature_layers_to_extract=None)

    teacher_net_org
    teacher_net
    student_net

    criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd_loss, temperature_kd_loss)

    #train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=teacher_net,
    #                intermediate_layers=intermediate_layers, cut_network=None, filename='hejhej', title=None)

    n_layers = 3

    train_first_layers(n_layers, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type)


if __name__ == '__main__':
    main()
