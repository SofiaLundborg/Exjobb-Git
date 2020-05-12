import matplotlib.pyplot as plt
import torch.optim as optim
import distillation_loss
from datetime import datetime
import time
from tqdm import tqdm
from binaryUtils import *
from extraUtils import calculate_accuracy, get_device, plot_results, get_device_id
from loadUtils import save_training, load_training
import numpy as np


def finetuning(net, train_loader, validation_loader, train_loader_for_accuracy, max_epochs, learning_rate_change, path=None, filename=None, saved_training = None, saved_model=None, initial_learning_rate=0.01):

    if net.n_layers == 18:
        layers_to_train = ['layer1', 'layer2', 'layer3', 'layer4']
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

    lr = initial_learning_rate
    weight_decay = 0
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = np.empty(max_epochs+1)
    validation_loss = np.empty(max_epochs+1)
    train_accuracy = np.empty(max_epochs+1)
    train_accuracy_top5 = np.empty(max_epochs+1)
    validation_accuracy = np.empty(max_epochs+1)
    validation_accuracy_top5 = np.empty(max_epochs+1)
    best_validation_accuracy = 0
    best_epoch = 0
    fig, (ax_loss, ax_acc, ax_acc5) = plt.subplots(1, 3, figsize=(15, 5))

    if saved_training:
        epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy,train_accuracy_top5, \
        validation_accuracy_top5, layer_index = load_training(net, optimizer, saved_training)
    else:
        epoch = -1
    while epoch < max_epochs:
        epoch += 1
        print('training for epoch ' + str(epoch) + 'has started')
        net.train()
        for p in list(net.parameters()):
            p.requires_grad = True

        if epoch in learning_rate_change:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('changed learning rate to ' + str(lr))

        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, targets = data

            # cpu / gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            binarize_weights(net)

            total_loss = criterion(net(inputs), targets)

            total_loss.backward()
            running_loss += total_loss.item()

            make_weights_real(net)
            optimizer.step()

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(net)
        print('Validation loss calculation has started')
        for i, data in enumerate(tqdm(validation_loader)):
            net.eval()
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
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

        torch.save(validation_loss[:epoch + 1], './Results/' + folder + 'validation_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_loss[:epoch + 1], './Results/' + folder + 'train_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(validation_accuracy[:epoch + 1], './Results/' + folder + 'validation_accuracy_top1_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_accuracy[:epoch + 1], './Results/' + folder + 'train_accuracy_top1_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_accuracy_top5[:epoch + 1], './Results/' + folder + 'train_accuracy_top5_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(validation_accuracy_top5[:epoch + 1], './Results/' + folder + 'validation_accuracy_top5_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')


        if accuracy_validation_epoch > best_validation_accuracy:
            # save network
            PATH = './Trained_Models/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
            torch.save(net.state_dict(), PATH)
            best_validation_accuracy = accuracy_validation_epoch
            best_epoch = epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Loss on train images: ' + str(training_loss_for_epoch))
        print('Loss on validation images: ' + str(validation_loss_for_epoch))
        print('Accuracy top 1 on train images: ' + str(accuracy_train_epoch))
        print('Accuracy top 1 on validation images: ' + str(accuracy_validation_epoch))
        print('Accuracy top 5 on train images: %d %%' + str(accuracy_train_epoch_top5))
        print('Accuracy top 5 on validation images: %d %%' + str(accuracy_validation_epoch_top5))

        save_training(epoch, net, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, train_accuracy_top5, validation_accuracy_top5,
                      None, 'saved_training/' + folder + filename + '_' + 'lr' + str(lr) + '_' + datetime.today().strftime('%Y%m%d'))


def training_a(student_net, teacher_net, train_loader, validation_loader, train_loader_not_augmented, filename=None, saved_training=None, modified=False, max_epoch_layer=40, max_epoch_finetuning=120, learning_rate_change=None):

    if not filename:
        filename = 'method_a_' + str(student_net.net_type)

    title_loss = 'method a) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method a) - accuracy, ' + str(student_net.net_type)

    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    optimizer = optim.Adam(student_net.parameters(), lr=0.01, weight_decay=0)

    if student_net.dataset == 'ImageNet':
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
    else:
        layers = ['layer1', 'layer2', 'layer3', 'all']
    #max_epoch_layer = 40
    max_epochs = max_epoch_layer * 4 + max_epoch_finetuning

    if saved_training:
        total_epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, accuracy_train_epoch_top5, accuracy_validation_epoch_top5, layer_idx = load_training(student_net, optimizer, saved_training)
        lr = 0.01
        epoch = total_epoch % max_epoch_layer

        train_loss_temp = np.empty(max_epochs)
        validation_loss_temp = np.empty(max_epochs)
        train_loss_temp[:len(train_loss)] = train_loss
        validation_loss_temp[:len(validation_loss)] = validation_loss
        train_loss = train_loss_temp
        validation_loss = validation_loss_temp

    else:
        train_loss = np.empty(max_epochs)
        validation_loss = np.empty(max_epochs)
        train_accuracy = np.empty(max_epoch_finetuning)
        validation_accuracy = np.empty(max_epoch_finetuning)
        layer_idx = 0
        total_epoch = -1
        epoch = -1

    PATH = None

    if torch.cuda.is_available():
        criterion = criterion.cuda(device=get_device_id())
        device = get_device()
    teacher_net.eval()

    changed_layer = False

    while layer_idx < len(layers):
        layer = layers[layer_idx]
        n_not_improved = 0
        if layer == 'all':
            if student_net.dataset == 'ImageNet':
                set_layers_to_binarize(student_net, ['layer1', 'layer2', 'layer3', 'layer4'])
            else:
                set_layers_to_binarize(student_net, ['layer1', 'layer2', 'layer3'])
            max_epoch_layer = max_epoch_finetuning
            criterion = torch.nn.CrossEntropyLoss()
        else:
            max_epoch_layer = max_epoch_layer
            set_layers_to_binarize(student_net, layers[:layer_idx+1])
        if student_net.dataset == 'ImageNet':
            cut_network = 1 + 4 * (layer_idx+1)
        else:
            cut_network = 1 + 6 * (layer_idx+1)

        if changed_layer or (not saved_training):
            lr = 0.01
            epoch = -1
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

        print(layer + " is training")

        while (epoch < max_epoch_layer-1):
            epoch += 1
            start_time_epoch = time.time()

            total_epoch += 1

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                if modified:
                    set_layers_to_update(student_net, layers[:layer_idx+1])
                else:
                    set_layers_to_update(student_net, [layer])

            #learning_rate_change = [25, 30, 35, 39]
            if layer == 'all':
                learning_rate_change = [50, 70, 80, 90]
                learning_rate_change = [70, 90, 100, 110]

            if epoch in learning_rate_change:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print('learning rate decreased to: ' + str(lr))

            running_loss = 0
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

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[total_epoch] = training_loss_for_epoch

            print("validation loss calculation has started")
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

            if student_net.dataset == 'ImageNet':
                folder = 'ImageNet/'
            else:
                folder = 'cifar10/'

            print("accuracy calculation has started")
            if layer == 'all':
                accuracy_train_epoch, accuracy_train_epoch_top5 = calculate_accuracy(train_loader_not_augmented, student_net, topk=(1,5))
                accuracy_validation_epoch, accuracy_validation_epoch_top5 = calculate_accuracy(validation_loader, student_net, topk=(1,5))
                train_accuracy[epoch] = accuracy_train_epoch
                validation_accuracy[epoch] = accuracy_validation_epoch
                plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=folder + filename,
                             title=title_accuracy)

                torch.save(validation_accuracy[:total_epoch + 1],
                           './Results/' + folder + datetime.today().strftime(
                               '%Y%m%d') + '_validation_accuracy_' + filename  + '.pt')
                torch.save(train_accuracy[:total_epoch + 1],
                           './Results/' + folder +  datetime.today().strftime(
                               '%Y%m%d') + '_train_accuracy_' + filename + '.pt')

            make_weights_real(student_net)

            plot_results(ax_loss, fig, train_loss, validation_loss, total_epoch, filename= folder + filename, title=title_loss)

            if student_net.dataset == 'ImageNet':
                folder = 'ImageNet/'
            else:
                folder = 'cifar10/'
            torch.save(validation_loss[:total_epoch+1], './Results/' + folder +  datetime.today().strftime('%Y%m%d') + 'validation_loss_' + filename +   '.pt')
            torch.save(train_loss[:total_epoch+1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'train_loss_' + filename + '.pt')

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

            save_training(total_epoch, student_net, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, None, None, layer_idx,
                          'saved_training/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d'))

            print('Epoch: ' + str(total_epoch))
            print('Best epoch: ' + str(best_epoch))
            print('Loss on train images: ' + str(training_loss_for_epoch))
            print('Loss on validation images: ' + str(validation_loss_for_epoch))
            if layer == 'all':
                print('Accuracy on train images: ' + str(accuracy_train_epoch))
                print('Accuracy on validation images: ' + str(accuracy_validation_epoch))

            if get_device() == 'cuda:1':
                time.sleep(0.1)

        layer_idx += 1
        changed_layer = True

    return PATH


def training_kd(studet_net, teacher_net, train_loader, validation_loader, train_loader_for_accuracy, temperature=6, scaling_factor_kd_loss=0.95, max_epochs=110, path=None, filename=None, learning_rate_change=None, saved_training = None, saved_model=None):


    title_loss = 'loss, ' + str(studet_net.net_type)
    title_accuracy = 'accuracy, ' + str(studet_net.net_type)
    title_accuracy_top5 = 'top 5 accuracy, ' + str(studet_net.net_type)
    if not filename:
        filename = 'kd_' + str(studet_net.net_type)

    criterion = distillation_loss.KdLoss(temperature=temperature, alpha=scaling_factor_kd_loss)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    device = get_device()

    lr = 1e-2
    weight_decay = 0  # 0.00001
    optimizer = optim.Adam(studet_net.parameters(), lr=lr, weight_decay=weight_decay)

    if saved_model:
        epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index = load_training(
            studet_net, optimizer, saved_model)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print('current learning rate is ' + str(lr))

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

    if studet_net.dataset == 'ImageNet':
        layers_to_train = ['layer1', 'layer2', 'layer3', 'layer4']
        print(layers_to_train)
    else:
        layers_to_train = ['layer1', 'layer2', 'layer3']
    set_layers_to_binarize(studet_net, layers_to_train)
    teacher_net.eval()

    if not learning_rate_change:
        learning_rate_change = [50, 200, 250]

    fig, (ax_loss, ax_acc, ax_acc5) = plt.subplots(1, 3, figsize=(15, 5))

    if saved_training:
        epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, layer_index = load_training(studet_net, optimizer, saved_training)
    else:
        epoch = -1
    while epoch < max_epochs:
        epoch += 1
        print('training for epoch ' + str(epoch) + 'has started')
        studet_net.train()
        for p in list(studet_net.parameters()):
            p.requires_grad = True

        if epoch in learning_rate_change:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('changed learning rate to ' + str(lr))

        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, targets = data

            # cpu / gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            binarize_weights(studet_net)

            output_student = studet_net(inputs)
            with torch.no_grad():
                output_teacher = teacher_net(inputs)

            total_loss = criterion(output_student, output_teacher, targets)

            total_loss.backward(retain_graph=True)  # calculate loss
            running_loss += total_loss.item()

            make_weights_real(studet_net)
            optimizer.step()

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(studet_net)
        print('Validation loss calculation has started')
        for i, data in enumerate(tqdm(validation_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                output_student = studet_net(inputs)
                output_teacher = teacher_net(inputs)
            running_validation_loss += criterion(output_student, output_teacher, targets)

        validation_loss_for_epoch = running_validation_loss / len(validation_loader)
        validation_loss[epoch] = validation_loss_for_epoch

        print('Accuracy of train set has started')
        accuracy_train_epoch, accuracy_train_epoch_top5 = calculate_accuracy(train_loader_for_accuracy, studet_net, topk=(1,5))
        print('Accuracy of validation set has started')
        accuracy_validation_epoch, accuracy_validation_epoch_top5 = calculate_accuracy(validation_loader, studet_net, topk=(1,5))

        train_accuracy[epoch] = accuracy_train_epoch
        train_accuracy_top5[epoch] = accuracy_train_epoch_top5
        validation_accuracy[epoch] = accuracy_validation_epoch
        validation_accuracy_top5[epoch] = accuracy_validation_epoch_top5

        make_weights_real(studet_net)

        if studet_net.dataset == 'ImageNet':
            folder = 'ImageNet/'
        else:
            folder = 'cifar10/'

        plot_results(ax_loss, fig, train_loss, validation_loss, epoch, filename=folder+filename, title=title_loss)
        plot_results(ax_acc, fig, train_accuracy, validation_accuracy, epoch, filename=folder+filename, title=title_accuracy)
        plot_results(ax_acc5, fig, train_accuracy_top5, validation_accuracy_top5, epoch, filename=folder+filename, title=title_accuracy_top5)

        torch.save(validation_loss[:epoch + 1], './Results/' + folder + 'validation_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_loss[:epoch + 1], './Results/' + folder + 'train_loss_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(validation_accuracy[:epoch + 1], './Results/' + folder + 'validation_accuracy_top1_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_accuracy[:epoch + 1], './Results/' + folder + 'train_accuracy_top1_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(train_accuracy_top5[:epoch + 1], './Results/' + folder + 'train_accuracy_top5_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')
        torch.save(validation_accuracy_top5[:epoch + 1], './Results/' + folder + 'validation_accuracy_top5_' + filename+ '_' + datetime.today().strftime('%Y%m%d') + '.pt')

        if validation_loss_for_epoch < best_validation_loss:
            # save network
            PATH = './Trained_Models/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
            torch.save(studet_net.state_dict(), PATH)
            best_validation_loss = validation_loss_for_epoch
            best_epoch = epoch

        print('Epoch: ' + str(epoch))
        print('Best epoch: ' + str(best_epoch))
        print('Loss on train images: ' + str(training_loss_for_epoch))
        print('Loss on validation images: ' + str(validation_loss_for_epoch))
        print('Accuracy top 1 on train images: ' + str(accuracy_train_epoch))
        print('Accuracy top 1 on validation images: ' + str(accuracy_validation_epoch))
        print('Accuracy top 5 on train images: %d %%' + str(accuracy_train_epoch_top5))
        print('Accuracy top 5 on validation images: %d %%' + str(accuracy_validation_epoch_top5))

        save_training(epoch, studet_net, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, train_accuracy_top5, validation_accuracy_top5,
                      None, 'saved_training/' + folder + filename + '_' + 'lr' + str(lr) + '_' + datetime.today().strftime('%Y%m%d'))


def training_c(student_net, teacher_net, train_loader, validation_loader, train_loader_non_augmented, filename=None, max_epochs=200, scaling_factor_total=0.5, max_epoch_layer=40, learning_rate_change=None, max_epoch_finetuning=25, saved_training=None):
    title_loss = 'method c) - loss, ' + str(student_net.net_type)
    title_accuracy = 'method c) - accuracy, ' + str(student_net.net_type)
    if not filename:
        filename = 'method_c_' + str(student_net.net_type)

    filename = filename

    if student_net.n_layers==20:
        layers = ['layer1', 'layer2', 'layer3']
    else:
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
    #max_epoch_layer = 40
    max_epochs = max_epoch_layer * 6
    train_loss = np.empty(max_epochs)
    validation_loss = np.empty(max_epochs)
    train_accuracy = np.empty(max_epochs)
    validation_accuracy = np.empty(max_epochs)

    criterion = distillation_loss.Loss_c(scaling_factor_total)
    lr = 0.01
    weight_decay = 0  # 0.00001
    optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

    if saved_training:
        total_epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy,_, _, layer_idx = load_training(
            student_net, optimizer, saved_training)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        epoch = total_epoch % max_epoch_layer - 1
        layer_idx = 2

    else:
        train_loss = np.empty(max_epochs)
        validation_loss = np.empty(max_epochs)
        train_accuracy = np.empty(max_epoch_finetuning)
        validation_accuracy = np.empty(max_epoch_finetuning)
        layer_idx = 0
        total_epoch = -1
        epoch = -1

    if torch.cuda.is_available():
        criterion = criterion.cuda(device=get_device_id())
    device = get_device()

    teacher_net.eval()

    changed_layer = False

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

    while layer_idx < len(layers):
        layer = layers[layer_idx]

        print(str(layer) + ' is training')

        if layer == 'all':
            set_layers_to_binarize(student_net, ['layer1', 'layer2', 'layer3'])
            max_epoch_layer = 60
        else:
            set_layers_to_binarize(student_net, layers[:layer_idx+1])
        cut_network = 1 + 6 * (layer_idx+1)

        best_validation_accuracy = 0
        best_epoch = 0

        #for epoch in range(max_epoch_layer):
        while (epoch < max_epoch_layer):
            epoch += 1

            print('layer-epoch ' + str(epoch) + ' has started')

            total_epoch += 1

            if layer == 'all':
                criterion = torch.nn.CrossEntropyLoss()
                student_net.train()
                for p in list(student_net.parameters()):
                    p.requires_grad = True
            else:
                set_layers_to_update(student_net, layers[:layer_idx+1])

            if student_net.n_layers == 20:
                learning_rate_change = [20, 30, 35]
            if layer == 'all':
                learning_rate_change = [30, 40, 50]

            if changed_layer or (not saved_training):
                lr = 0.01
                epoch = -1
                print(str(epoch))
                if layer == 'all':
                    lr = 0.01
                weight_decay = 0  # 0.00001
                optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                optimizer = optim.Adam(student_net.parameters(), lr=lr, weight_decay=0)

            if epoch in learning_rate_change:
                lr = lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            print('lr: ' + str(lr))

            running_loss = 0
            print('Training of epoch ' + str(total_epoch) + ' has started')
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
                    total_loss = criterion(inputs, targets, student_net, teacher_net, cut_network=cut_network)

                total_loss.backward(retain_graph=True)  # calculate loss
                running_loss += total_loss.item()

                make_weights_real(student_net)
                optimizer.step()

            training_loss_for_epoch = running_loss / len(train_loader)
            train_loss[total_epoch] = training_loss_for_epoch

            running_validation_loss = 0
            binarize_weights(student_net)
            print('Validation loss calculation has started')
            for i, data in enumerate(tqdm(validation_loader)):
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

            print('Accuracy calculation has started')
            accuracy_train_epoch, accuracy_train_epoch_top_5 = calculate_accuracy(train_loader_non_augmented, student_net)
            accuracy_validation_epoch, accuracy_validation_epoch_top_5 = calculate_accuracy(validation_loader, student_net)
            train_accuracy[total_epoch] = accuracy_train_epoch
            validation_accuracy[total_epoch] = accuracy_validation_epoch
            make_weights_real(student_net)

            if student_net.dataset == 'ImageNet':
                folder = 'ImageNet/'
            else:
                folder = 'cifar10/'
            plot_results(ax_loss, fig, train_loss[:total_epoch+1], validation_loss[:total_epoch+1], total_epoch, filename=folder+filename, title=title_loss)
            plot_results(ax_acc, fig, train_accuracy[:total_epoch+1], validation_accuracy[:total_epoch+1], total_epoch, filename=folder+filename, title=title_accuracy)

            torch.save(validation_loss[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'validation_loss_' + filename + '.pt')
            torch.save(train_loss[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'train_loss_' + filename + '.pt')
            torch.save(validation_accuracy[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'validation_accuracy_' + filename + '.pt')
            torch.save(train_accuracy[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'train_accuracy_' + filename + '.pt')

            if accuracy_validation_epoch > best_validation_accuracy:
                # save network
                PATH = './Trained_Models/' + filename + '_' + datetime.today().strftime('%Y%m%d') + '.pth'
                torch.save(student_net.state_dict(), PATH)
                best_validation_accuracy = accuracy_validation_epoch
                best_epoch = total_epoch

            print('Epoch: ' + str(total_epoch))
            print('Best epoch: ' + str(best_epoch))
            print('Loss on train images: ' + str(training_loss_for_epoch))
            print('Loss on validation images: ' + str(validation_loss_for_epoch))
            print('Accuracy on train images: %d %%' % accuracy_train_epoch)
            print('Accuracy on validation images: %d %%' % accuracy_validation_epoch)

            save_training(total_epoch, student_net, optimizer, train_loss, validation_loss, train_accuracy,
                          validation_accuracy, accuracy_train_epoch_top_5, accuracy_validation_epoch_top_5, layer_idx,
                          'saved_training/' + folder + filename + '_' + datetime.today().strftime('%Y%m%d'))

        layer_idx += 1
        changed_layer = True

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


def lit_training(student_net, train_loader, validation_loader, train_loader_non_augmented, max_epochs=200, teacher_net=None, filename=None, scaling_factor_total=0.75, scaling_factor_kd=0.95):

    temperature_kd = 6

    title_loss = 'Loss method b), ' + str(student_net.net_type)
    title_accuracy = 'Accuracy method b), ' + str(student_net.net_type)

    if not filename:
        filename = 'method_b_' + str(student_net.net_type)

    filename = filename + 'scaling_tot_' + str(scaling_factor_total) + '_scaling_kd_' + str(scaling_factor_kd)

    criterion = distillation_loss.Loss(scaling_factor_total, scaling_factor_kd, temperature_kd)
    if torch.cuda.is_available():
        criterion = criterion.cuda(device=get_device_id())
    device = get_device()

    if student_net.n_layers == 20:
        layers_to_train = ['layer1', 'layer2', 'layer3']
        intermediate_layers = [1, 7, 13, 19]

    else:
        layers_to_train = ['layer1', 'layer2', 'layer3', 'layer4']
        intermediate_layers = [1, ]
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

        learning_rate_change = [30, 40, 50, 55]
        #learning_rate_change = [70, 90, 100, 110]
        if epoch in learning_rate_change:
            lr = lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print('Training of epoch ' + str(epoch) + ' has started')
        for i, data in enumerate(tqdm(train_loader)):
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

        training_loss_for_epoch = running_loss / len(train_loader)
        train_loss[epoch] = training_loss_for_epoch

        running_validation_loss = 0
        binarize_weights(student_net)

        print('Validation loss calculation has started')
        for i, data in enumerate(tqdm(validation_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            if lit:
                running_validation_loss += criterion(inputs, targets, student_net, teacher_net, intermediate_layers, lit_training=True, training=False, input_from_teacher=input_from_teacher).item()
            else:
                running_validation_loss += criterion(inputs, targets, student_net).item()

        validation_loss_for_epoch = running_validation_loss / len(validation_loader)
        validation_loss[epoch] = validation_loss_for_epoch

        print('Accuracy calculation has started')
        accuracy_train_epoch, accuracy_train_epoch_top5 = calculate_accuracy(train_loader_non_augmented, student_net, topk=(1,5))
        accuracy_validation_epoch, accuracy_validation_epoch_top5 = calculate_accuracy(validation_loader, student_net, topk=(1,5))
        train_accuracy[epoch] = accuracy_train_epoch
        validation_accuracy[epoch] = accuracy_validation_epoch
        make_weights_real(student_net)

        if student_net.dataset == 'ImageNet':
            folder = 'ImageNet/'
        else:
            folder = 'cifar10/'

        plot_results(ax_loss, fig, train_loss[:epoch+1], validation_loss[:epoch+1], epoch, filename=folder+filename, title=title_loss)
        plot_results(ax_acc, fig, train_accuracy[:epoch+1], validation_accuracy[:epoch+1], epoch, filename=folder+filename, title=title_accuracy)

        torch.save(validation_loss[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'validation_loss_' + filename + '.pt')
        torch.save(train_loss[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'train_loss_' + filename + '.pt')
        torch.save(validation_accuracy[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'validation_accuracy_' + filename + '.pt')
        torch.save(train_accuracy[:epoch + 1], './Results/' + folder + datetime.today().strftime('%Y%m%d') + 'train_accuracy_' + filename + '.pt')

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
