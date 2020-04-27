import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from binaryUtils import *
from extraUtils import change_loaded_checkpoint
from models import resNet
from models import originalResnet
import distillation_loss
from datetime import datetime
from matplotlib.lines import Line2D
import time
import torchvision.models as models
import torch.distributed as dist
from tqdm import tqdm

import dataloaders
import warnings


def load_imageNet(subsets=None):
    normalizing_mean = [0.485, 0.456, 0.406]
    normalizing_std = [0.229, 0.224, 0.225]

    if torch.cuda.is_available():
        batch_size_training = 64    #64
        batch_size_validation = 64  #64
    else:
        batch_size_training = 4
        batch_size_validation = 4

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalizing_mean, std=normalizing_std)])

    preprocessing_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalizing_mean, std=normalizing_std),
    ])

    preprocessing_valid = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalizing_mean, std=normalizing_std),
    ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalizing_mean, std=normalizing_std)])

    train_set = torchvision.datasets.ImageNet(root='./data', split='train', transform=preprocessing_train)
    train_set_not_disturbed = torchvision.datasets.ImageNet(root='./data', split='train', transform=preprocessing_valid)
    print('train set is loaded')
    validation_set = torchvision.datasets.ImageNet(root='./data', split='val', transform=preprocessing_valid)
    print('validation set is loaded')
    if subsets:
        train_set, ndjkfnskj = torch.utils.data.random_split(train_set, [10000, len(train_set) - 10000])
        validation_set, ndjkfnskj = torch.utils.data.random_split(validation_set, [10000, len(validation_set)-10000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_training,
                                               shuffle=True, num_workers=8, pin_memory=True)
    train_loader_not_disturbed = torch.utils.data.DataLoader(train_set_not_disturbed, batch_size=batch_size_training,
                                                             shuffle=False, num_workers=8, pin_memory=True)
    print('train_loader finished')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_validation,
                                                    shuffle=False, num_workers=8, pin_memory=True)
    print('validation_loader finished')

    return train_loader, validation_loader, train_loader_not_disturbed


def load_data(dataset):
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

    if dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
    elif dataset == 'ImageNet':
        train_set = torchvision.datasets.ImageNet(root='./data', train=True, transform=transform_train)
        #test_set = torchvision.datasets.ImageNet(root='./data', train=False,
                                                  #transform=transform_test)

    # divide into train and validation data (80% train)
    train_size = int(0.8 * len(train_set))
    validation_size = len(train_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, validation_size])

    #train_set, ndjkfnskj = torch.utils.data.random_split(train_set, [200, len(train_set) - 200])
    #validation_set, ndjkfnskj = torch.utils.data.random_split(validation_set, [50, len(validation_set)-50])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_training,
                                               shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_validation,
                                                    shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_validation,
                                              shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader


def save_training(epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_idx, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'train_accuracy': train_accuracy,
        'validation_accuracy': validation_accuracy,
        'layer_index': layer_idx
    }, PATH)


def load_training(model, optimizer, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    validation_loss = checkpoint['validation_loss']
    train_accuracy = checkpoint['train_accuracy']
    validation_accuracy = checkpoint['validation_accuracy']
    layer_index = checkpoint['layer_index']

    return epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index

def load_model_from_saved_training(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_data_loaders():
    return dataloaders.CIFAR10DataLoaders.train_loader(batch_size=32), dataloaders.CIFAR10DataLoaders.val_loader()


def get_one_sample(data_loader):
    image, targets = next(iter(data_loader))
    return image


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


def training_a(student_net, teacher_net, train_loader, validation_loader, filename=None, saved_training=None):

    if not filename:
        filename = 'method_a_' + str(student_net.net_type)

    title_loss = 'method a) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method a) - accuracy, ' + str(student_net.net_type)

    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    optimizer = optim.Adam(student_net.parameters(), lr=0.01, weight_decay=0)

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'all']
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    max_epoch_layer = 30
    max_epochs = max_epoch_layer * 3 + 60

    if saved_training:
        total_epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_idx = load_training(student_net, optimizer, saved_training)
        lr = 0.01
        epoch = total_epoch % max_epoch_layer
    else:
        train_loss = np.empty(max_epochs)
        validation_loss = np.empty(max_epochs)
        train_accuracy = np.empty(110)
        validation_accuracy = np.empty(110)
        layer_idx = 0
        total_epoch = -1
        epoch = -1

    PATH = None

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        device = get_device()
    teacher_net.eval()

    changed_layer = False

    while layer_idx < len(layers):
        layer = layers[layer_idx]
    #for layer_idx, layer in enumerate(layers):
        n_not_improved = 0
        if layer == 'all':
            set_layers_to_binarize(student_net, ['layer1', 'layer2', 'layer3', 'layer4'])
            max_epoch_layer = 60
            criterion = torch.nn.CrossEntropyLoss()
        else:
            max_epoch_layer = 4
            set_layers_to_binarize(student_net, layers[:layer_idx+1])
        if student_net.dataset == 'ImageNet':
            cut_network = 1 + 4 * (layer_idx+1)
        else:
            cut_network = 1 + 6 * (layer_idx+1)

        if changed_layer or (not saved_training):
            lr = 0.01
            if layer == 'all':
                lr = 0.01
            weight_decay = 0  # 0.00001
            optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=0)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

        best_validation_loss = np.inf
        best_epoch = 0
        n_change_learning_rate = 0

        #for epoch in range(max_epoch_layer):

        print(layer + " is training")
        # epoch = -1
        #for epoch in range(max_epoch_layer):

        while (epoch < max_epoch_layer-1):
            epoch += 1
            start_time_epoch = time.time()

            #total_epoch = epoch + 5*layer_idx
            total_epoch += 1

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                set_layers_to_update(student_net, layers[:layer_idx+1])

            learning_rate_change = [2, 3, 4]
            #learning_rate_change = [30, 45, 50, 55]
            if layer == 'all':
                learning_rate_change = [50, 70, 90, 100]
                learning_rate_change = [30, 40, 50]

            # if n_not_improved >= 2:
            #     if layer == 'all':
            #         if n_change_learning_rate >= 4:
            #             lr = lr * 0.1
            #             n_change_learning_rate = 0
            #         else:
            #             n_change_learning_rate += 1
            #     else:
            #         lr = lr * 0.1
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #         n_not_improved = 0

            if epoch in learning_rate_change:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print('learning rate decreased to: ' + str(lr))

            running_loss = 0
            start_training_time = time.time()
            print('Training of epoch ' + str(total_epoch) + " has started")
            for i, data in enumerate(tqdm(train_loader)):
                inputs, targets = data

                # cpu / gpu
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                binarize_weights(student_net)

                if layer == 'all':
                    total_loss = criterion(student_net(inputs), targets)
                else:
                    output_student = student_net(inputs, cut_network=cut_network)
                    with torch.no_grad():
                        output_teacher = teacher_net(inputs, cut_network=cut_network)
                    total_loss = criterion(output_student, output_teacher)

                total_loss.backward(retain_graph=True)  # calculate loss
                running_loss += total_loss.item()

                make_weights_real(student_net)
                optimizer.step()

            end_training_time = time.time()
            log = open("timelog.txt", "a+")
            log.write(
                "Training time for epoch " + str(total_epoch) + ": " + str(end_training_time - start_training_time) + "seconds \n\r")
            log.close()

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[total_epoch] = training_loss_for_epoch

            print("validation loss calculation has started")
            start_validation_loss = time.time()
            running_validation_loss = 0
            binarize_weights(student_net)
            for i, data in enumerate(tqdm(validation_loader)):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    if layer == 'all':
                        running_validation_loss += criterion(student_net(inputs), targets)
                    else:
                        output_student = student_net(inputs, cut_network=cut_network)
                        output_teacher = teacher_net(inputs, cut_network=cut_network)
                        running_validation_loss += criterion(output_student, output_teacher).item()
            validation_loss_for_epoch = running_validation_loss / len(validation_loader)
            validation_loss[total_epoch] = validation_loss_for_epoch
            end_validation_loss = time.time()

            log = open("timelog.txt", "a+")
            log.write(
                "Validation loss calculation time for epoch " + str(total_epoch) + ": " + str(
                    end_validation_loss - start_validation_loss) + " seconds \n\r")
            log.close()

            if student_net.dataset == 'ImageNet':
                folder = 'ImageNet/'
            else:
                folder = 'cifar10/'

            print("accuracy calculation has started")
            start_accuracy_time = time.time()
            if layer == 'all':
                accuracy_train_epoch, accuracy_train_epoch_top5 = calculate_accuracy(train_loader, student_net, topk=(1,5))
                accuracy_validation_epoch, accuracy_validation_epoch_top5 = calculate_accuracy(validation_loader, student_net, topk=(1,5))
                train_accuracy[epoch] = accuracy_train_epoch
                validation_accuracy[epoch] = accuracy_validation_epoch
                plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=folder + filename,
                             title=title_accuracy)

                torch.save(validation_accuracy[:total_epoch + 1],
                           './Results/' + folder + 'validation_accuracy_' + filename + '_' + datetime.today().strftime(
                               '%Y%m%d') + '.pt')
                torch.save(train_accuracy[:total_epoch + 1],
                           './Results/' + folder + 'train_accuracy_' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pt')
            end_accuracy_time = time.time()

            log = open("timelog.txt", "a+")
            log.write(
                "Accuracy calculation time for epoch " + str(total_epoch) + ": " + str(
                    end_accuracy_time - start_accuracy_time) + " seconds\n\r")
            log.close()

            make_weights_real(student_net)

            plot_results(ax_loss, fig, train_loss, validation_loss, total_epoch, filename= folder + filename, title=title_loss)

            if student_net.dataset == 'ImageNet':
                folder = 'ImageNet/'
            else:
                folder = 'cifar10/'
            torch.save(validation_loss[:total_epoch+1], './Results/' + folder + 'validation_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') +  '.pt')
            torch.save(train_loss[:total_epoch+1], './Results/' + folder + 'train_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')

            if validation_loss_for_epoch < best_validation_loss:
                # save network
                if layer=='all':
                    PATH = './Trained_Models/' + folder + filename + '_finetuning' + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                else:
                    PATH = './Trained_Models/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_loss = validation_loss_for_epoch
                best_epoch = total_epoch

            else:
                n_not_improved += 1

            save_training(total_epoch, student_net, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_idx,
                          'saved_training/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d'))

            print('Epoch: ' + str(total_epoch))
            print('Best epoch: ' + str(best_epoch))
            print('Loss on train images: ' + str(training_loss_for_epoch))
            print('Loss on validation images: ' + str(validation_loss_for_epoch))
            if layer == 'all':
                print('Accuracy on train images: %d %%' % accuracy_train_epoch)
                print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

            end_time_epoch = time.time()

            log = open("timelog.txt", "a+")
            log.write("TOTAL TIME for epoch " + str(total_epoch) + ": " + str(end_time_epoch-start_time_epoch) + " seconds\n\n\r" )
            log.close()

        layer_idx += 1
        changed_layer = True

    return PATH


def finetuning(net, train_loader, validation_loader, train_loader_for_accuracy, max_epochs, path=None, filename=None, learning_rate_change=None, saved_training = None, saved_model=None):

    if net.dataset == 'ImageNet':
        layers_to_train = ['layer1', 'layer2', 'layer3', 'layer4']
        print(layers_to_train)
    else:
        layers_to_train = ['layer1', 'layer2', 'layer3']
    set_layers_to_binarize(net, layers_to_train)

    title_loss = 'loss, ' + str(net.net_type)
    title_accuracy = 'accuracy, ' + str(net.net_type)
    title_accuracy_top5 = 'top 5 accuracy, ' + str(net.net_type)
    if not filename:
        filename = 'finetuning_' + str(net.net_type)

    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    lr = 0.001
    lr = 0.01
    weight_decay = 0  # 0.00001
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if saved_model:
        epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index = load_training(
            net, optimizer, saved_model)

    # lr = 0.01
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

    train_loss = np.empty(max_epochs+1)
    validation_loss = np.empty(max_epochs+1)
    train_accuracy = np.empty(max_epochs+1)
    train_accuracy_top5 = np.empty(max_epochs+1)
    validation_accuracy = np.empty(max_epochs+1)
    validation_accuracy_top5 = np.empty(max_epochs+1)
    best_validation_loss = np.inf
    best_epoch = 0

    if not learning_rate_change:
        learning_rate_change = [0, 10, 20, 25]

    fig, (ax_loss, ax_acc, ax_acc5) = plt.subplots(1, 3, figsize=(15, 5))

    if saved_training:
        epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index = load_training(net, optimizer, saved_training)
    else:
        epoch = -1
    while epoch < max_epochs:
        epoch += 1
        print('training for epoch ' + str(epoch) + 'has started')
    #for epoch in range(max_epochs):
        net.train()
        for p in list(net.parameters()):
            p.requires_grad = True

        if epoch in learning_rate_change:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
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

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(net)
        print('Validation loss calculation has started')
        for i, data in enumerate(tqdm(validation_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            running_validation_loss += criterion(net(inputs), targets).item()

        validation_loss_for_epoch = running_validation_loss / len(validation_loader)
        validation_loss[epoch] = validation_loss_for_epoch

        print('Accuracy of train set has started')
        accuracy_train_epoch, accuracy_train_epoch_top5 = calculate_accuracy(train_loader_for_accuracy, net, topk=(1,5))
        print('Accuracy of validation set has started')
        accuracy_validation_epoch, accuracy_validation_epoch_top5 = calculate_accuracy(validation_loader, net, topk=(1,5))

        train_accuracy[epoch] = accuracy_train_epoch
        train_accuracy_top5[epoch] = accuracy_train_epoch_top5
        validation_accuracy[epoch] = accuracy_validation_epoch
        validation_accuracy_top5[epoch] = accuracy_validation_epoch_top5

        make_weights_real(net)

        if net.dataset == 'ImageNet':
            folder = 'ImageNet/'
        else:
            folder = 'cifar10/'

        plot_results(ax_loss, fig, train_loss, validation_loss, epoch, filename=folder+filename, title=title_loss)
        plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=folder+filename, title=title_accuracy)
        plot_results(ax_acc5, fig, train_accuracy_top5, validation_accuracy_top5, epoch, filename=folder+filename, title=title_accuracy_top5)

        torch.save(validation_loss[:epoch + 1], './Results/' + folder + 'validation_loss_' + filename + '.pt')
        torch.save(train_loss[:epoch + 1], './Results/' + folder + 'train_loss_' + filename + '.pt')
        torch.save(validation_accuracy[:epoch + 1], './Results/' + folder + 'validation_accuracy_top1_' + filename + '.pt')
        torch.save(train_accuracy[:epoch + 1], './Results/' + folder + 'train_accuracy_top1_' + filename + '.pt')
        torch.save(train_accuracy_top5[:epoch + 1], './Results/' + folder + 'validation_accuracy_top5_' + filename + '.pt')
        torch.save(validation_accuracy_top5[:epoch + 1], './Results/' + folder + 'train_accuracy_top5_' + filename + '.pt')

        if validation_loss_for_epoch < best_validation_loss:
            # save network
            PATH = './Trained_Models/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
            torch.save(net.state_dict(), PATH)
            best_validation_loss = validation_loss_for_epoch
            best_epoch = epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Loss on train images: ' + str(training_loss_for_epoch))
        print('Loss on validation images: ' + str(validation_loss_for_epoch))
        print('Accuracy top 1 on train images: %d %%' % accuracy_train_epoch)
        print('Accuracy top 1 on validation images: %d %%' % accuracy_validation_epoch)
        print('Accuracy top 5 on train images: %d %%' % accuracy_train_epoch_top5)
        print('Accuracy top 5 on validation images: %d %%' % accuracy_validation_epoch_top5)

        save_training(epoch, net, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy,
                      None, 'saved_training/' + folder + filename + '_' + 'lr' + str(lr) + '_' + datetime.today().strftime('%Y%m%d'))


def training_c(student_net, teacher_net, train_loader, validation_loader, filename=None, max_epochs=200, scaling_factor_total=0.5):
    title_loss = 'method c) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method c) - accuracy, ' + str(student_net.net_type)
    if not filename:
        filename = 'method_c_' + str(student_net.net_type)

    filename = filename + '_lambda_' + str(scaling_factor_total)

    layers = ['layer1', 'layer2', 'layer3', 'all']
    max_epoch_layer = 40
    max_epochs = max_epoch_layer * 6
    train_loss = np.empty(max_epochs)
    validation_loss = np.empty(max_epochs)
    train_accuracy = np.empty(max_epochs)
    validation_accuracy = np.empty(max_epochs)

    criterion = distillation_loss.Loss_c(scaling_factor_total)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    teacher_net.eval()

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

            total_epoch = epoch + 40*layer_idx

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                set_layers_to_update(student_net, layers[:layer_idx+1])

            learning_rate_change = [20, 30, 35]
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

                if layer == 'all':
                    total_loss = criterion(student_net(inputs), targets)
                else:
                    total_loss = criterion(inputs, targets, student_net, teacher_net, cut_network=cut_network)

                total_loss.backward(retain_graph=True)  # calculate loss
                running_loss += total_loss.item()

                make_weights_real(student_net)
                optimizer.step()

            time.sleep(5)

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[total_epoch] = training_loss_for_epoch

            running_validation_loss = 0
            binarize_weights(student_net)
            for i, data in enumerate(validation_loader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                if layer == 'all':
                    with torch.no_grad():
                        running_validation_loss += criterion(student_net(inputs), targets)
                else:
                    running_validation_loss += criterion(inputs, targets, student_net, teacher_net, cut_network=cut_network, training=False).item()

            validation_loss_for_epoch = running_validation_loss / len(validation_loader)
            validation_loss[total_epoch] = validation_loss_for_epoch

            accuracy_train_epoch = calculate_accuracy(train_loader, student_net)
            accuracy_validation_epoch = calculate_accuracy(validation_loader, student_net)
            train_accuracy[total_epoch] = accuracy_train_epoch
            validation_accuracy[total_epoch] = accuracy_validation_epoch
            make_weights_real(student_net)

            plot_results(ax_loss, fig, train_loss, validation_loss, total_epoch, filename=filename, title=title_loss)
            plot_results(ax_acc, fig, train_accuracy, validation_accuracy, total_epoch, filename=filename, title=title_accuracy)

            torch.save(validation_loss[:total_epoch+1], './Results/validation_loss_' + filename + '.pt')
            torch.save(train_loss[:total_epoch+1], './Results/train_loss_' + filename + '.pt')
            torch.save(validation_accuracy[:total_epoch+1], './Results/validation_accuracy_' + filename + '.pt')
            torch.save(train_accuracy[:total_epoch+1], './Results/train_accuracy_' + filename + '.pt')

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
            print('Accuracy on train images: %d %%' % accuracy_train_epoch)
            print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

            time.sleep(5)


def test_heatmap(student_net, teacher_net, train_loader):

    student_dict = torch.load('./Trained_Models/' + 'method_a_one_shortcut_distribution_scaling_Xnor++_20200331.pth',
                              map_location=get_device())
    student_net.load_state_dict(student_dict)

    cut_network = 19

    student_net.eval()
    teacher_net.eval()

    layers_to_train = ['layer1', 'layer2', 'layer3']

    set_layers_to_binarize(student_net, layers_to_train)
    set_layers_to_update(student_net, layers_to_train)
    binarize_weights(student_net)

    loss_mse = torch.nn.MSELoss()

    for i, data in enumerate(train_loader, 0):
        inputs, targets = data

        with torch.no_grad():
            student_output = student_net(inputs, cut_network=cut_network)
            teacher_output = teacher_net(inputs, cut_network=cut_network)

        student_feature_map = student_output.numpy()
        teacher_feature_map = teacher_output.numpy()

        diff = student_feature_map - teacher_feature_map

        fig, ax = plt.subplots(3, 5)

        ax[0, 0].imshow(student_feature_map[0, 0, :, :])
        ax[1, 0].imshow(teacher_feature_map[0, 0, :, :])
        ax[2, 0].imshow(diff[0, 0, :, :])
        ax[0, 1].imshow(student_feature_map[0, 1, :, :])
        ax[1, 1].imshow(teacher_feature_map[0, 1, :, :])
        ax[2, 1].imshow(diff[0, 1, :, :])
        ax[0, 2].imshow(student_feature_map[0, 2, :, :])
        ax[1, 2].imshow(teacher_feature_map[0, 2, :, :])
        ax[2, 2].imshow(diff[0, 2, :, :])

        ax[0, 0].set_ylabel('student')
        ax[1, 0].set_ylabel('teacher')
        ax[2, 0].set_ylabel('diff')

        im03 = ax[0, 3].imshow(student_feature_map[0, 3, :, :])
        im13 = ax[1, 3].imshow(teacher_feature_map[0, 3, :, :])
        im23 = ax[2, 3].imshow(diff[0, 3, :, :])

        fig.colorbar(im03, ax=ax[0, :])
        fig.colorbar(im13, ax=ax[1, :])
        fig.colorbar(im23, ax=ax[2, :])

        layers_to_train = ['layer1', 'layer2', 'layer3']
        set_layers_to_binarize(student_net, layers_to_train)
        set_layers_to_update(student_net, layers_to_train)

        with torch.no_grad():
            teacher_res = teacher_net(inputs, cut_network=19)
            student_res = student_net(inputs, cut_network=19)
            ax[0, 4].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher', color='blue')
            ax[1, 4].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student', color='green')
            ax[2, 4].hist((student_res - teacher_res).view(-1), bins=50, alpha=0.3, density=True, label='Student', color='red')

        loss = loss_mse(student_output, teacher_output)
        print(loss)

        plt.show()


def lit_training(student_net, train_loader, validation_loader, max_epochs=200, teacher_net=None, filename=None, scaling_factor_total=0.75, scaling_factor_kd=0.95):

    student_dict = torch.load('./Trained_Models/' + 'LIT_with_double_shortcut_Xnor++_20200325.pth',
                              map_location=get_device())
    # student_net.load_state_dict(student_dict)

    temperature_kd = 6
    #scaling_factor_kd = 0.95        # LIT 0.95
    #scaling_factor_total = 0.75     # LIT 0.75

    title_loss = 'Loss method b), ' + str(student_net.net_type)
    title_accuracy = 'Accuracy method b), ' + str(student_net.net_type)

    if not filename:
        filename = 'method_b_' + str(student_net.net_type)

    filename = filename + 'scaling_tot_' + str(scaling_factor_total) + '_scaling_kd_' + str(scaling_factor_kd)

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

    PATH = None

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

        learning_rate_change = [30, 50, 70, 90]
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

        accuracy_train_epoch = calculate_accuracy(train_loader, student_net, topk=(1,))
        accuracy_validation_epoch = calculate_accuracy(validation_loader, student_net, topk=(1,0))
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

    return PATH


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
    dataset = 'cifar10'
    max_epochs = 200
    scaling_factor_total = 0.75     # LIT: 0.75
    scaling_factor_kd_loss = 0.95   # LIT: 0.95
    temperature_kd_loss = 6.0       # LIT: 6.0

    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), './pretrained_resnet_models_imagenet/resnet18.pth')
    original_teacher_dict = torch.load('./pretrained_resnet_models_imagenet/resnet18.pth')
    print('pretrained model loaded')
    teacher_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)

    train_loader, validation_loader, test_loader = load_data(dataset)
    train_loader, validation_loader, train_loader_not_disturbed = load_imageNet()
    print('ImageNet loaded')


    teacher_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)
    checkpoint_teacher = change_loaded_checkpoint(original_teacher_dict, teacher_ResNet18)
    teacher_ResNet18.load_state_dict(checkpoint_teacher)

    # # Initialize distributed training
    # if torch.cuda.device_count() > 1:
    #     teacher_ResNet18 = torch.nn.DataParallel(teacher_ResNet18)
    #     resnet18 = torch.nn.DataParallel(resnet18)

    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
        teacher_ResNet18 = teacher_ResNet18.cuda()
    device = get_device()
    sample = get_one_sample(train_loader).to(device)

    train_loader_subset, validation_loader_subset, train_loader_not_disturbed_subset= load_imageNet(subsets=True)
    #torch.save(train_loader_subset, 'train_loader_subset.pth')
    #torch.save(validation_loader_subset, 'validation_loader_subset.pth')


    filename = 'method_a_double_shortcut_with_finetuning_' + str(net_type)
    student_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)
    checkpoint_student = change_loaded_checkpoint(original_teacher_dict, student_ResNet18)
    student_ResNet18.load_state_dict(checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet18 = student_ResNet18.cuda()

    filename = 'finetuning_after_method_a_complete_set_initial_lr_0.001_' + str(net_type)
    saved_training = './saved_training/ImageNet/finetuning_after_method_a_complete_set_initial_lr_0.001_Xnor++_lr0.001_20200426'
    student_ResNet18 = load_model_from_saved_training(student_ResNet18, PATH='./saved_training/ImageNet/finetuning_after_method_a_double_shortcut_complete_setXnor++_20200424')


    # lr = 0.001
    # weight_decay = 0  # 0.00001
    # optimizer = optim.Adam(student_ResNet18.parameters(), lr=lr, weight_decay=weight_decay)
    #
    # epoch, student_ResNet18, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index = load_training(
    #     student_ResNet18, optimizer, './saved_training/ImageNet/finetuning_after_method_a_double_shortcut_complete_setXnor++_20200424')

    finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_disturbed, 30, filename=filename, saved_model='./saved_training/ImageNet/method_a_double_shortcut_with_relu_long_Xnor++_20200421', saved_training=saved_training)

    filename = 'method_a_correct_shortcut_factorized_Xnor++_'
    #path = training_a(student_ResNet18, teacher_ResNet18, train_loader, validation_loader, filename, saved_training='./saved_training/ImageNet/method_a_correct_shortcut_factorized_Xnor++__20200427')

    print('finished training')

    #
    # accuracy_org = calculate_accuracy(validation_loader, resnet18)
    # print('accuracy org: ' + str(accuracy_org))
    # accuracy_teacher = calculate_accuracy(validation_loader, teacher_ResNet18)
    # print('accuracy teacher: ' + str(accuracy_teacher))
    #
    # # initailize_networks
    # teacher_net = resNet.resnet_models['resnet20ForTeacher']('full_precision', dataset)
    # student_net = resNet.resnet_models[net_name + 'relufirst'](net_type)
    #
    # # load pretrained network into student and techer network
    # teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'
    # teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    # new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_net)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # teacher_net.load_state_dict(new_checkpoint_teacher)
    # student_net.load_state_dict(new_checkpoint_student)
    #
    # checkpoint = torch.load('./Trained_Models/lit_finetuning_binary_20200316.pth', map_location='cpu')
    #
    # teacher_net_org = originalResnet.resnet_models["cifar"][net_name]()
    # teacher_pth = './pretrained_resnet_cifar10_models/student/' + net_name + '.pth'
    # teacher_checkpoint_org = torch.load(teacher_pth, map_location='cpu')
    # teacher_net_org.load_state_dict(teacher_checkpoint_org)
    #
    # if torch.cuda.is_available():
    #     teacher_net = teacher_net.cuda()
    #     student_net = student_net.cuda()
    #     teacher_net_org = teacher_net_org.cuda()
    #
    # trained_student_checkpoint = torch.load('Trained_Models/1_to_7layers_bin_Xnor++_20200302.pth', map_location='cpu')
    # trained_student_net = resNet.resnet_models[net_name]('Xnor++')
    # trained_student_net.load_state_dict(trained_student_checkpoint)
    #
    #
    #
    # #teacher_net_pretrained = models.resnet18(pretrained=True)
    #
    # teacher_net.eval()

    #acc_teacher = calculate_accuracy(train_loader, teacher_net)
    #print(acc_teacher)

    #acc_teacher_org = calculate_accuracy(train_loader, teacher_net_org)
    #print(acc_teacher_org)

    # res_teacher = teacher_net(sample)
    # res_org = teacher_net_org(sample)

    # print(str(res_teacher == res_org))

    # print('hej')

    # layers_to_train = ['layer1', 'layer2', 'layer3']
    # set_layers_to_binarize(student_net, layers_to_train)
    # set_layers_to_update(student_net, layers_to_train)
    # binarize_weights(student_net)
    #
    # with torch.no_grad():
    #     fig, ax = plt.subplots(1, 3)
    #     # fig, ax = plt.subplots()
    #
    #
    #     teacher_res = teacher_net(sample, cut_network=7)
    #     student_res = student_net(sample, cut_network=7)
    #     ax[0].hist(teacher_res.view(-1), bins=100, alpha=0.5, density=True, label='Teacher', color='grey')
    #     ax[0].plot([0,0], [0, 0.30], linestyle='dashed', color='black')
    #     ax[0].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student', color='green')
    #
    #     teacher_res = teacher_net(sample, cut_network=13)
    #     student_res = student_net(sample, cut_network=13)
    #     ax[1].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher', color='grey')
    #     ax[1].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student', color='green')
    #
    #     teacher_res = teacher_net(sample, cut_network=19)
    #     student_res = student_net(sample, cut_network=19)
    #     ax[2].hist(teacher_res.view(-1), bins=50, alpha=0.3, density=True, label='Teacher', color='grey')
    #     ax[2].hist(student_res.view(-1), bins=50, alpha=0.3, density=True, label='Student', color='green')
    #
    #     plt.show()

    # set_layers_to_binarize(trained_student_net, 1, 7)
    # out = trained_student_net(sample)

    # criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd_loss, temperature_kd_loss)

    # train_one_block(student_net, train_loader, validation_loader, max_epochs, criterion, teacher_net=teacher_net,
    #                intermediate_layers=intermediate_layers, cut_network=None, filename='hejhej', title=None)

    # train_first_layers(start_layer, end_layer, student_net, teacher_net, train_loader, validation_loader, max_epochs, net_type)
    # lit_training(student_net, train_loader, validation_loader, max_epochs, teacher_net)

    #finetuning(student_net, train_loader, validation_loader, 60)

    #training_c(student_net, teacher_net, train_loader, validation_loader, max_epochs=200)

    # training_a(student_net, teacher_net, train_loader, validation_loader)

    # training_c(student_net, teacher_net, train_loader, validation_loader)
    # test_heatmap(student_net, teacher_net, train_loader)

    #
    # filename = 'method_a_with_relu_block' + str(net_type)
    # student_net = resNet.resnet_models['resnet20WithRelu'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_with_relu_block_finetuning_' + str(net_type)
    # finetuning(student_net, train_loader, validation_loader, 60, path, filename)
    #
    # filename = 'method_a_with_abs_block' + str(net_type)
    # student_net = resNet.resnet_models['resnet20Abs'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_with_abs_block_finetuning_' + str(net_type)
    # finetuning(student_net, train_loader, validation_loader, 60, path, filename)
    #
    # filename = 'method_a_with_double_shortcut_block' + str(net_type)
    # student_net = resNet.resnet_models['resnet20AbsDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_with_double_shortcut_block_finetuning_' + str(net_type)
    # finetuning(student_net, train_loader, validation_loader, 60, path, filename)
    #

    learning_rate_change = [50, 70, 90, 100]
    #


    #
    # net_type = 'Xnor++'
    # factorized_gamma = True
    # filename = 'no_method_double_shortcut_with_relu_factorized_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type, factorized_gamma=factorized_gamma)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # finetuning(student_net, train_loader, validation_loader, 110, filename=filename,
    #            learning_rate_change=learning_rate_change)
    # filename = 'method_a_double_shortcut_with_relu_factorized_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type, factorized_gamma=factorized_gamma)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    #
    #
    # net_type = 'binary'
    # filename = 'no_method_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # finetuning(student_net, train_loader, validation_loader, 110, filename=filename,
    #            learning_rate_change=learning_rate_change)
    # filename = 'method_a_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_double_shortcut_with_relu_finetuning_' + str(net_type)
    # #finetuning(student_net, train_loader, validation_loader, 110, path, filename,
    # #           learning_rate_change=learning_rate_change)
    #
    # net_type = 'Xnor'
    # filename = 'no_method_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # finetuning(student_net, train_loader, validation_loader, 110, filename=filename,
    #            learning_rate_change=learning_rate_change)
    # filename = 'method_a_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_double_shortcut_with_relu_finetuning_' + str(net_type)
    # #finetuning(student_net, train_loader, validation_loader, 110, path, filename,
    # #           learning_rate_change=learning_rate_change)
    #
    # net_type = 'Xnor++'
    # filename = 'no_method_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # finetuning(student_net, train_loader, validation_loader, 110, filename=filename,
    #            learning_rate_change=learning_rate_change)

    # filename = 'method_a_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # path = training_a(student_net, teacher_net, train_loader, validation_loader, filename)
    # filename = 'method_a_double_shortcut_with_relu_finetuning_' + str(net_type)
    #finetuning(student_net, train_loader, validation_loader, 110, path, filename,
    #           learning_rate_change=learning_rate_change)


    scaling_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # for scaling_factor in scaling_factors:
    #
    #     filename = 'method_c_double_shortcut_with_relu_' + str(net_type)
    #     student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    #     new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    #     student_net.load_state_dict(new_checkpoint_student)
    #     if torch.cuda.is_available():
    #         student_net = student_net.cuda()
    #     training_c(student_net, teacher_net, train_loader, validation_loader, filename=filename, scaling_factor_total=scaling_factor)

    # for scaling_factor in scaling_factors:
    #     filename = 'method_b_double_shortcut_with_relu_' + str(net_type)
    #     student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    #     new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    #     student_net.load_state_dict(new_checkpoint_student)
    #     if torch.cuda.is_available():
    #         student_net = student_net.cuda()
    #     path = lit_training(student_net, train_loader, validation_loader, max_epochs=100, teacher_net=teacher_net, filename=filename, scaling_factor_total=scaling_factor)
    #     filename = 'method_b_finetuning_double_shortcut_with_relu_tot_scaling_' + str(scaling_factor) + '_' + str(net_type)
    #     finetuning(student_net, train_loader, validation_loader, 60, path, filename)


    # learning_rate_change = [50, 70, 90, 100]
    # filename = 'no_method_double_shortcut_with_relu_' + str(net_type)
    # student_net = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_net)
    # student_net.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_net = student_net.cuda()
    # finetuning(student_net, train_loader, validation_loader, 110, filename=filename, learning_rate_change=learning_rate_change)



if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
