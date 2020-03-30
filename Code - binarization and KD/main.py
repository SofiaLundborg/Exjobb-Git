import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from binaryUtils import *
from extraUtils import change_loaded_checkpoint
from models import resNet
import distillation_loss
from datetime import datetime
from matplotlib.lines import Line2D
import time

import dataloaders
import warnings


def load_data():
    # Load data
    normalizing_mean = [0.485, 0.456, 0.406]
    normalizing_std = [0.229, 0.224, 0.225]

    if torch.cuda.is_available():
        batch_size_training = 512
        batch_size_validation = 1024
    else:
        batch_size_training = 4
        batch_size_validation = 4

    # normalizing_mean = [0.4914, 0.4822, 0.4465]
    # normalizing_std = [0.2470, 0.2435, 0.2616]

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

    # train_set, ndjkfnskj = torch.utils.data.random_split(train_set, [50, len(train_set) - 50])
    # validation_set, ndjkfnskj = torch.utils.data.random_split(validation_set, [50, len(validation_set)-50])

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

            if torch.cuda.is_available():
                images = images.to('cuda')
                targets = targets.to('cuda')

            outputs = net(images)
            prec1 = accuracy(outputs, targets)
            accuracy1.update(prec1[0], images.size(0))

            # if i % 10 == 9:
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
    #set_layers_to_binarize(student_net, start_layer, end_layer)
    #set_layers_to_update(student_net, start_layer, end_layer)

    layers = ['layer1']

    set_layers_to_binarize(student_net, layers)

    cut_network = end_layer
    # cut_network = None

    criterion = distillation_loss.Loss(1, 0.95, 6.0)

    filename = str(start_layer) + '_to_' + str(end_layer) + 'layers_bin_' + str(net_type)
    title = 'loss, ' + str(start_layer) + ' to ' + str(end_layer) + ' layers binarized, ' + str(net_type)
    train_results, validation_results = train_one_block(student_net, train_loader, validation_loader, max_epochs,
                                                        criterion, teacher_net, layers_to_train=layers, cut_network=cut_network,
                                                        filename=filename, title=title, accuracy_calc=False)
    min_loss = min(train_results)

    return min_loss


def training_a(student_net, teacher_net, train_loader, validation_loader):

    title_loss = 'method a) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method a) - accuracy, ' + str(student_net.net_type)
    filename = 'method_a_one_shortcut_distribution_scaling_' + str(student_net.net_type)

    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    layers = ['layer1', 'layer2', 'layer3', 'all']
    max_epoch_layer = 30
    max_epochs = max_epoch_layer * 5

    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    train_loss = np.empty(max_epochs)
    validation_loss = np.empty(max_epochs)
    train_accuracy = np.empty(max_epoch_layer)
    validation_accuracy = np.empty(max_epoch_layer)

    for layer_idx, layer in enumerate(layers):
        if layer == 'all':
            set_layers_to_binarize(student_net, ['layer1', 'layer2', 'layer3'])
            max_epoch_layer = 60
        else:
            set_layers_to_binarize(student_net, layers[:layer_idx+1])
        cut_network = 1 + 6 * (layer_idx+1)

        lr = 0.01
        weight_decay = 0  # 0.00001
        optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

        best_validation_loss = np.inf
        best_epoch = 0

        for epoch in range(max_epoch_layer):

            total_epoch = epoch + max_epoch_layer*layer_idx

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                set_layers_to_update(student_net, layers[:layer_idx+1])

            learning_rate_change = [15, 20, 25]
            if layer == 'all':
                learning_rate_change = [30, 40, 50]

            if epoch in learning_rate_change:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            running_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data

                # cpu / gpu
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                binarize_weights(student_net)

                output_student = student_net(inputs, cut_network=cut_network)
                with torch.no_grad():
                    output_teacher = teacher_net(inputs, cut_network=cut_network)

                total_loss = criterion(output_student, output_teacher)

                total_loss.backward(retain_graph=True)  # calculate loss
                running_loss += total_loss.item()

                make_weights_real(student_net)
                optimizer.step()

            # time.sleep(5)

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[total_epoch] = training_loss_for_epoch

            running_validation_loss = 0
            binarize_weights(student_net)
            for i, data in enumerate(validation_loader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    output_student = student_net(inputs, cut_network=cut_network)
                    output_teacher = teacher_net(inputs, cut_network=cut_network)
                    running_validation_loss += criterion(output_student, output_teacher).item()

            validation_loss_for_epoch = running_validation_loss / len(validation_loader)
            validation_loss[total_epoch] = validation_loss_for_epoch

            if layer == 'all':
                accuracy_train_epoch = calculate_accuracy(train_loader, student_net)
                accuracy_validation_epoch = calculate_accuracy(validation_loader, student_net)
                train_accuracy[epoch] = accuracy_train_epoch
                validation_accuracy[epoch] = accuracy_validation_epoch
                plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=filename,
                             title=title_accuracy)
                torch.save(validation_accuracy[:total_epoch + 1],
                           './Results/validation_accuracy_' + filename + '_' + datetime.today().strftime(
                               '%Y%m%d') + '.pt')
                torch.save(train_accuracy[:total_epoch + 1],
                           './Results/train_accuracy_' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pt')

            make_weights_real(student_net)

            plot_results(ax_loss, fig, train_loss, validation_loss, total_epoch, filename=filename, title=title_loss)

            torch.save(validation_loss[:total_epoch+1], './Results/validation_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') +  '.pt')
            torch.save(train_loss[:total_epoch+1], './Results/train_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')

            if validation_loss_for_epoch < best_validation_loss:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_loss = validation_loss_for_epoch
                best_epoch = total_epoch

            print('Epoch: ' + str(total_epoch))
            print('Best epoch: ' + str(best_epoch))
            print('Loss on train images: ' + str(training_loss_for_epoch))
            print('Loss on validation images: ' + str(validation_loss_for_epoch))
            if layer == 'all':
                print('Accuracy on train images: %d %%' % accuracy_train_epoch)
                print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

            time.sleep(5)


def finetuning(net, train_loader, validation_loader, max_epochs):

    student_dict = torch.load('./Trained_Models/method_a_one_shortcut_distribution_scaling_Xnor++_20200330.pth',
                              map_location=get_device())
    net.load_state_dict(student_dict)

    layers_to_train = ['layer1', 'layer2', 'layer3']
    set_layers_to_binarize(net, layers_to_train)

    title_loss = 'method a - loss, ' + str(net.net_type)
    title_accuracy = 'method a - accuracy, ' + str(net.net_type)
    filename = 'method_a_one_shortcut_distribution_scaling_finetuning' + str(net.net_type)

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    lr = 0.01
    weight_decay = 0  # 0.00001
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

    train_loss = np.empty(max_epochs)
    validation_loss = np.empty(max_epochs)
    train_accuracy = np.empty(max_epochs)
    validation_accuracy = np.empty(max_epochs)
    best_validation_loss = np.inf
    best_epoch = 0

    for epoch in range(max_epochs):

        net.train()
        for p in list(net.parameters()):
            p.requires_grad = True

        learning_rate_change = [30, 40, 50, 60]
        if epoch in learning_rate_change:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data

            # cpu / gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            binarize_weights(net)

            total_loss = criterion(net(inputs), targets)

            total_loss.backward(retain_graph=True)  # calculate loss
            running_loss += total_loss.item()

            make_weights_real(net)
            optimizer.step()

        time.sleep(5)

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(net)
        for i, data in enumerate(validation_loader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            running_validation_loss += criterion(net(inputs), targets).item()

        validation_loss_for_epoch = running_validation_loss / len(validation_loader)
        validation_loss[epoch] = validation_loss_for_epoch

        accuracy_train_epoch = calculate_accuracy(train_loader, net)
        accuracy_validation_epoch = calculate_accuracy(validation_loader, net)
        train_accuracy[epoch] = accuracy_train_epoch
        validation_accuracy[epoch] = accuracy_validation_epoch
        make_weights_real(net)

        plot_results(ax_loss, fig, train_loss, validation_loss, epoch, filename=filename, title=title_loss)
        plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=filename, title=title_accuracy)

        torch.save(validation_loss[:epoch + 1], './Results/validation_loss_' + filename + '.pt')
        torch.save(train_loss[:epoch + 1], './Results/train_loss_' + filename + '.pt')
        torch.save(validation_accuracy[:epoch + 1], './Results/validation_accuracy_' + filename + '.pt')
        torch.save(train_accuracy[:epoch + 1], './Results/train_accuracy_' + filename + '.pt')

        if validation_loss_for_epoch < best_validation_loss:
            # save network
            PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
            torch.save(net.state_dict(), PATH)
            best_validation_loss = validation_loss_for_epoch
            best_epoch = epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Loss on train images: ' + str(training_loss_for_epoch))
        print('Loss on validation images: ' + str(validation_loss_for_epoch))
        print('Accuracy on train images: %d %%' % accuracy_train_epoch)
        print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

        time.sleep(5)


def training_c(student_net, teacher_net, train_loader, validation_loader, max_epochs=200):
    title_loss = 'method c) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method c) - accuracy, ' + str(student_net.net_type)
    filename = 'method_c_one_shortcut_distribution_scaling' + str(student_net.net_type)

    scaling_factor_total = 0.5

    layers = ['layer1', 'layer2', 'layer3', 'all']
    max_epoch_layer = 40

    criterion = distillation_loss.Loss_c(scaling_factor_total)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    for layer_idx, layer in enumerate(layers):
        if layer == 'all':
            set_layers_to_binarize(student_net, layers)
        else:
            set_layers_to_binarize(student_net, layers[:layer_idx+1])
        cut_network = 1 + 6 * (layer_idx+1)

        lr = 0.01
        weight_decay = 0  # 0.00001
        optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

        train_loss = np.empty(max_epochs)
        validation_loss = np.empty(max_epochs)
        train_accuracy = np.empty(max_epochs)
        validation_accuracy = np.empty(max_epochs)
        best_validation_loss = np.inf
        best_epoch = 0

        for epoch in range(max_epoch_layer):

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                set_layers_to_update(student_net, layers[:layer_idx+1])

            learning_rate_change = [20, 25, 30]
            if epoch in learning_rate_change:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            running_loss = 0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data

                # cpu / gpu
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                binarize_weights(student_net)

                total_loss = criterion(inputs, targets, student_net, teacher_net, cut_network=cut_network)

                total_loss.backward(retain_graph=True)  # calculate loss
                running_loss += total_loss.item()

                make_weights_real(student_net)
                optimizer.step()

            time.sleep(5)

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[epoch] = training_loss_for_epoch

            running_validation_loss = 0
            binarize_weights(student_net)
            for i, data in enumerate(validation_loader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                running_validation_loss += criterion(inputs, targets, student_net, teacher_net, cut_network=cut_network, training=False).item()

            validation_loss_for_epoch = running_validation_loss / len(validation_loader)
            validation_loss[epoch] = validation_loss_for_epoch

            accuracy_train_epoch = calculate_accuracy(train_loader, student_net)
            accuracy_validation_epoch = calculate_accuracy(validation_loader, student_net)
            train_accuracy[epoch] = accuracy_train_epoch
            validation_accuracy[epoch] = accuracy_validation_epoch
            make_weights_real(student_net)

            plot_results(ax_loss, fig, train_loss, validation_loss, epoch, filename=filename, title=title_loss)
            plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=filename, title=title_accuracy)

            torch.save(validation_loss[:epoch+1], './Results/validation_loss_' + filename + '.pt')
            torch.save(train_loss[:epoch+1], './Results/train_loss_' + filename + '.pt')
            torch.save(validation_accuracy[:epoch+1], './Results/validation_accuracy_' + filename + '.pt')
            torch.save(train_accuracy[:epoch+1], './Results/train_accuracy_' + filename + '.pt')

            if validation_loss_for_epoch < best_validation_loss:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_loss = validation_loss_for_epoch
                best_epoch = epoch

            print('Epoch: ' + str(epoch))
            print('Best epoch: ' + str(best_epoch))
            print('Loss on train images: ' + str(training_loss_for_epoch))
            print('Loss on validation images: ' + str(validation_loss_for_epoch))
            print('Accuracy on train images: %d %%' % accuracy_train_epoch)
            print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

            time.sleep(5)




def lit_training(student_net, train_loader, validation_loader, max_epochs=200, teacher_net=None):

    student_dict = torch.load('./Trained_Models/' + 'LIT_with_double_shortcut_Xnor++_20200325.pth',
                              map_location=get_device())
    #student_net.load_state_dict(student_dict)

    temperature_kd = 6
    scaling_factor_kd = 0.95        # LIT 0.95
    scaling_factor_total = 0     # LIT 0.75

    title_loss = 'Loss Lit, ' + str(student_net.net_type)
    filename = 'lit_' + str(student_net.net_type)
    title_accuracy = 'Accuracy Lit, ' + str(student_net.net_type)

    title_loss = 'CE after LIT with double shortcut - loss, ' + str(student_net.net_type)
    title_accuracy = 'CE after LIT with double shortcut - accuracy, ' + str(student_net.net_type)
    filename = 'CE_after_LIT_with_double_shortcut_' + str(student_net.net_type)

    title_loss = 'LIT with two shortcut - loss, ' + str(student_net.net_type)
    title_accuracy = 'LIT with two shortcut - accuracy, ' + str(student_net.net_type)
    filename = 'beta0_LIT_with_two_shortcut_distribution_scaling' + str(student_net.net_type)

    criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd, temperature_kd)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    layers_to_train = ['layer1', 'layer2', 'layer3']
    intermediate_layers = [1, 7, 13, 19]
    set_layers_to_binarize(student_net, layers_to_train)
    set_layers_to_update(student_net, layers_to_train)
    lit = True
    input_from_teacher = True

    if teacher_net:
        teacher_net.eval()

    lr = 0.01
    weight_decay = 0  # 0.00001
    optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

    train_loss = np.empty(max_epochs)
    validation_loss = np.empty(max_epochs)
    train_accuracy = np.empty(max_epochs)
    validation_accuracy = np.empty(max_epochs)
    best_validation_loss = np.inf
    best_epoch = 0

    for epoch in range(max_epochs):
        running_loss = 0
        if lit and (epoch <= 1000):
            # for p in list(student_net.parameters()):
            #     p.requires_grad = True
            set_layers_to_update(student_net, layers_to_train)
        else:
            student_net.train()
            for p in list(student_net.parameters()):
                p.requires_grad = True

        learning_rate_change = [30, 50, 80, 90]
        if epoch in learning_rate_change:
            lr = lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch == 1000:
            student_dict = torch.load('./Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth', map_location=device)
            student_net.load_state_dict(student_dict)
            teacher_net = None
            intermediate_layers = None
            input_from_teacher = False
            # lit = False
            lr = 0.01
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch == 600:
            lit = False
            lr = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data

            # cpu / gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            binarize_weights(student_net)

            if lit:
                total_loss = criterion(inputs, targets, student_net, teacher_net, intermediate_layers, lit_training=lit, input_from_teacher=input_from_teacher)
            else:
                total_loss = criterion(inputs, targets, student_net)

            total_loss.backward(retain_graph=True)  # calculate loss
            running_loss += total_loss.item()

            make_weights_real(student_net)
            optimizer.step()

        time.sleep(5)

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(student_net)
        for i, data in enumerate(validation_loader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            if lit:
                running_validation_loss += criterion(inputs, targets, student_net, teacher_net, intermediate_layers, lit_training=True, training=False, input_from_teacher=input_from_teacher).item()
            else:
                running_validation_loss += criterion(inputs, targets, student_net).item()

        validation_loss_for_epoch = running_validation_loss / len(validation_loader)
        validation_loss[epoch] = validation_loss_for_epoch

        accuracy_train_epoch = calculate_accuracy(train_loader, student_net)
        accuracy_validation_epoch = calculate_accuracy(validation_loader, student_net)
        train_accuracy[epoch] = accuracy_train_epoch
        validation_accuracy[epoch] = accuracy_validation_epoch
        make_weights_real(student_net)

        plot_results(ax_loss, fig, train_loss, validation_loss, epoch, filename=filename, title=title_loss)
        plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=filename, title=title_accuracy)

        torch.save(validation_loss[:epoch+1], './Results/validation_loss_' + filename + '.pt')
        torch.save(train_loss[:epoch+1], './Results/train_loss_' + filename + '.pt')
        torch.save(validation_accuracy[:epoch+1], './Results/validation_accuracy_' + filename + '.pt')
        torch.save(train_accuracy[:epoch+1], './Results/train_accuracy_' + filename + '.pt')

        if validation_loss_for_epoch < best_validation_loss:
            # save network
            PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
            torch.save(student_net.state_dict(), PATH)
            best_validation_loss = validation_loss_for_epoch
            best_epoch = epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Loss on train images: ' + str(training_loss_for_epoch))
        print('Loss on validation images: ' + str(validation_loss_for_epoch))
        print('Accuracy on train images: %d %%' % accuracy_train_epoch)
        print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

        time.sleep(5)


def train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=None, layers_to_train=None,
                    intermediate_layers=None, cut_network=None, filename=None, title=None, accuracy_calc=True):

    lr = 0.001
    weight_decay = 0     # 0.00001

    # optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.1)     # Original momentum = 0.9
    optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

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

        if epoch % 25 == 24:
            lr = lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        set_layers_to_update(student_net, layers_to_train)
        if teacher_net:
            teacher_net.eval()

        running_loss = 0.0
        running_loss_minibatch = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data

            # cpu / gpu
            if torch.cuda.is_available():
                criterion = criterion.cuda()
            device = get_device()
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            binarize_weights(student_net)

            loss = criterion(inputs, targets, student_net, teacher_net, intermediate_layers, cut_network)
            loss.backward(retain_graph=True)  # calculate loss
            running_loss += loss.item()
            running_loss_minibatch += loss.item()

            # plot_grad_flow(student_net.named_parameters())

            make_weights_real(student_net)
            optimizer.step()

            # print statistics

            # if i % 200 == 199:  # print every 200 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch, i + 1, running_loss_minibatch / 200))
            #    running_loss_minibatch = 0.0

        loss_for_epoch = running_loss / len(train_loader)

        if accuracy_calc:
            binarize_weights(student_net)
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
            make_weights_real(student_net)

        else:
            train_results[epoch] = loss_for_epoch
            if lowest_loss > loss_for_epoch:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_epoch = epoch

        if lowest_loss > loss_for_epoch:
            lowest_loss = loss_for_epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Epoch loss: ' + str(loss_for_epoch))
        print('Best loss: ' + str(lowest_loss))
        if accuracy_calc:
            print('Training accuracy: ' + str(accuracy_train))
            print('Best validation accuracy: ' + str(best_validation_accuracy))

        plot_results(ax, fig, train_results, validation_results, epoch+1, filename, title)

    print('Finished Training')

    return train_results, validation_results


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


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


def main():
    net_name = 'resnet20'           # 'leNet', 'ninNet', 'resnetX' where X = 20, 32, 44, 56, 110, 1202
    net_type = 'Xnor++'             # 'full_precision', 'binary', 'binary_with_alpha', 'Xnor' or 'Xnor++'
    max_epochs = 200
    scaling_factor_total = 0.75     # LIT: 0.75
    scaling_factor_kd_loss = 0.95   # LIT: 0.95
    temperature_kd_loss = 6.0       # LIT: 6.0

    train_loader, validation_loader, test_loader = load_data()

    # initailize_networks
    teacher_net = resNet.resnet_models["cifar"][net_name + 'relufirst']('full_precision')

    student_net = resNet.resnet_models["cifar"][net_name + 'relufirst'](net_type)

    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_net)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    teacher_net.load_state_dict(new_checkpoint_teacher)
    student_net.load_state_dict(new_checkpoint_student)

    checkpoint = torch.load('./Trained_Models/lit_finetuning_binary_20200316.pth', map_location='cpu')

    if torch.cuda.is_available():
        teacher_net = teacher_net.cuda()
        student_net = student_net.cuda()

    trained_student_checkpoint = torch.load('Trained_Models/1_to_7layers_bin_Xnor++_20200302.pth', map_location='cpu')
    trained_student_net = resNet.resnet_models["cifar"][net_name]('Xnor++')
    trained_student_net.load_state_dict(trained_student_checkpoint)

    device = get_device()
    sample = get_one_sample(train_loader).to(device)

    layers_to_train = ['layer1', 'layer2', 'layer3']
    #set_layers_to_binarize(student_net, layers_to_train)
    set_layers_to_update(student_net, layers_to_train)
    # binarize_weights(student_net)

    # with torch.no_grad():
    #     teacher_res = teacher_net(sample, cut_network=7)
    #     student_res = student_net(sample, cut_network=7)
    #
    #     fig, ax = plt.subplots(1, 3)
    #     ax[0].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher')
    #     ax[0].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student')
    #
    #     teacher_res = teacher_net(sample, cut_network=13)
    #     student_res = student_net(sample, cut_network=13)
    #     ax[1].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher')
    #     ax[1].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student')
    #
    #     teacher_res = teacher_net(sample, cut_network=19)
    #     student_res = student_net(sample, cut_network=19)
    #     ax[2].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher')
    #     ax[2].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student')
    #
    #     #plt.show()


    print('student net')

    acc_teacher = calculate_accuracy(train_loader, teacher_net)

    # set_layers_to_binarize(trained_student_net, 1, 7)
    # out = trained_student_net(sample)

    # criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd_loss, temperature_kd_loss)

    # train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=teacher_net,
    #                intermediate_layers=intermediate_layers, cut_network=None, filename='hejhej', title=None)

    # train_first_layers(start_layer, end_layer, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type)
    # lit_training(student_net, train_loader, validation_loader, max_epochs, teacher_net)

    finetuning(student_net, train_loader, validation_loader, 60)

    #training_c(student_net, teacher_net, train_loader, validation_loader, max_epochs=200)

    # training_a(student_net, teacher_net, train_loader, validation_loader)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
