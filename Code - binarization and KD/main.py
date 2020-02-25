import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from binaryUtils import *
from extraUtils import copy_parameters
from models import originalResnet, resNet
import distillation_loss


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
    validation_set, ndjkfnskj = torch.utils.data.random_split(validation_set, [2000, len(validation_set)-2000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4,
                                                    shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader


def get_one_sample(data_loader):
    image, targets = next(iter(data_loader))
    return image

def calculate_accuracy(data_loader, net):
    n_correct = 0
    n_total = 0
    accuracy1 = AverageMeter()
    with torch.no_grad():
        for data in data_loader:
            images, targets = data
            outputs = net(images)
            prec1 = accuracy(outputs, targets)
            accuracy1.update(prec1[0], images.size(0))

            # _, predicted = torch.max(outputs.data, 1)
            # n_total += targets.size(0)
            # n_correct += (predicted == targets).sum().item()

    return accuracy1.avg        #100 * n_correct / n_total


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


def train_first_layers(n_layers, student_net, teacher_net, train_loader, validation_loader, max_epochs):
    set_layers_to_binarize(student_net, [1, n_layers])
    set_layers_to_update(student_net, [1, n_layers])
    cut_network = n_layers

    criterion = distillation_loss.Loss(0, 0, 0)

    train_results, validation_results = train_one_block(student_net, train_loader, validation_loader, max_epochs, )



def train_one_block(student_net, train_loader, validation_loader, max_epochs, net_name, net_type, criterion,
                    intermediate_layers=None, teacher_net=None, cut_network=None, filename=None):

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9)

    train_results = np.empty(max_epochs)
    validation_results = np.empty(max_epochs)
    best_validation_accuracy = 0
    best_epoch = 0

    for epoch in range(max_epochs):  # loop over the data set multiple times
        student_net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data

            # Cuda

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
            if i % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        binarize_weights(student_net)

        student_net.eval()
        accuracy_train = calculate_accuracy(train_loader, student_net)
        accuracy_validation = calculate_accuracy(validation_loader, student_net)
        train_results[epoch] = accuracy_train
        validation_results[epoch] = accuracy_validation
        print('Epoch: ' + str(epoch))
        print('Accuracy of the network on the train images: %d %%' % accuracy_train)
        print('Accuracy of the network on the validation images: %d %%' % accuracy_validation)

        make_weights_real(student_net)

        if accuracy_validation > best_validation_accuracy:
            # save network
            PATH = './cifar10_' + net_type + '_' + net_name + '.pth'
            torch.save(student_net.state_dict(), PATH)

            best_validation_accuracy = accuracy_validation
            best_epoch = epoch

        print('Best epoch: ' + str(best_epoch + 1))

        plot_results(train_results, validation_results, epoch+1, net_name, net_type)

    print('Finished Training')

    return train_results, validation_results


def loss_KT(output_features_fp, output_features_bin, output_classifier_bin, targets, scaling_factor):
    criterion_class = nn.CrossEntropyLoss()
    criterion_feature = nn.MSELoss()
    classification_loss = criterion_class.forward(output_classifier_bin, targets)
    feature_loss = criterion_feature.forward(output_features_bin, output_features_fp)

    total_loss = classification_loss + scaling_factor*feature_loss

    return total_loss


def train_with_KT(fp_net, net, train_loader, validation_loader, max_epochs, scaling_factor_loss, net_name, net_type):
    feature_net_fp = nn.Sequential(*list(fp_net.children())[:-1])
    feature_net_bin = nn.Sequential(*list(net.children())[:-1])
    classifier_net_bin = nn.Sequential(*list(net.children())[-1])

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_results = np.empty(max_epochs)
    validation_results = np.empty(max_epochs)
    best_validation_accuracy = 0
    best_epoch = 0

    for epoch in range(max_epochs):  # loop over the dataset multiple times

        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            binarize_weights(net)

            output_features_fp = feature_net_fp(inputs)
            output_features_bin = feature_net_bin(inputs)
            input_classifier_bin = output_features_bin.view(-1, list(classifier_net_bin.modules())[0][0].in_features)
            output_classifier_bin = classifier_net_bin(input_classifier_bin)
            loss = loss_KT(output_features_fp, output_features_bin, output_classifier_bin, labels, scaling_factor_loss)
            loss.backward()  # calculate loss

            make_weights_real(net)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        binarize_weights(net)

        net.eval()
        accuracy_train = calculate_accuracy(train_loader, net)
        accuracy_validation = calculate_accuracy(validation_loader, net)
        train_results[epoch] = accuracy_train
        validation_results[epoch] = accuracy_validation
        print('Epoch: ' + str(epoch+1))
        print('Accuracy of the network on the train images: %d %%' % accuracy_train)
        print('Accuracy of the network on the validation images: %d %%' % accuracy_validation)

        make_weights_real(net)

        if accuracy_validation > best_validation_accuracy:
            # save network
            PATH = './cifar10_' + net_type + '_' + net_name + '_KT_true' + '.pth'
            torch.save(net.state_dict(), PATH)

            best_validation_accuracy = accuracy_validation
            best_epoch = epoch

        print('Best epoch: ' + str(best_epoch+1))

        plot_results(train_results, validation_results, epoch+1, net_name, net_type, True)

    print('Finished Training')
    return train_results, validation_results


def plot_results(train_results, validation_results, max_epochs, net_name, net_type, filename=None, title=None):
    fig, ax = plt.subplots()
    ax.plot(np.arange(max_epochs) + 1, train_results[:max_epochs], label='train accuracy')
    ax.plot(np.arange(max_epochs) + 1, validation_results[:max_epochs], label='validation accuracy')
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title('accuracy for ' + net_name + ' ' + net_type)

    if not filename:
        if net_type == 'Xnor++':
            n_type = 'XnorPp'
        else:
            n_type = net_type
        filename = 'accuracy_' + net_name + '_' + n_type + '.eps'
    fig.savefig(filename, format='eps')


def main():
    net_name = 'resnet110'           # 'leNet', 'ninNet', 'resnetX' where X = 20, 32, 44, 56, 110, 1202
    net_type = 'Xnor'               # 'full_precision', 'binary_with_alpha', 'Xnor' or 'Xnor++'
    max_epochs = 150
    scaling_factor_total = 0.75     # LIT: 0.75
    scaling_factor_kd_loss = 0.95   # LIT: 0.95
    temperature_kd_loss = 6.0       # LIT: 6.0

    # train_loader, validation_loader = get_data_loaders()

    train_loader, validation_loader, test_loader = load_data()

    # load pre trained teacher network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'
    teacher_pth = './pretrained_resnet_cifar10_models/teacher/' + net_name + '.pth'

    teacher_net_org = originalResnet.resnet_models["cifar"][net_name]()
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    teacher_net_org.load_state_dict(teacher_checkpoint)
    teacher_net = resNet.resnet_models["cifar"][net_name]('full_precision')
    copy_parameters(teacher_net, teacher_net_org)

    # initialize student network as the teacher network
    student_net = resNet.resnet_models["cifar"][net_name](net_type)
    copy_parameters(student_net, teacher_net)


    sample_batch = get_one_sample(test_loader)
    set_layers_to_binarize(student_net, [1, 1])
    set_layers_to_update(student_net, [1, 1])
    intermediate_layers = [1]
    cut_network = 2
    # out = student_net(sample_batch, feature_layers_to_extract=None, cut_network=cut_network)
    out = student_net(sample_batch, feature_layers_to_extract=None)


    acc2 = calculate_accuracy(test_loader, teacher_net_org)
    print(acc2)

    acc2 = calculate_accuracy(validation_loader, teacher_net_org)
    print(acc2)

    acc1 = calculate_accuracy(validation_loader, teacher_net)
    print(acc1)

    acc3 = calculate_accuracy(validation_loader, student_net)
    print(acc3)





    criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd_loss, temperature_kd_loss)

    #train_one_block(student_net, train_loader, validation_loader, max_epochs, net_name, net_type, criterion,
    #                intermediate_layers, teacher_net)


if __name__ == '__main__':
    main()
