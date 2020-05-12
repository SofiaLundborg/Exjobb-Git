from models import resNet, originalResnet
import torchvision.models as models
import warnings
from binaryUtils import *
from loadUtils import load_cifar10, load_imageNet
from train import finetuning, training_a, lit_training, training_c, training_kd
from extraUtils import change_loaded_checkpoint, calculate_accuracy, get_device_id
from tqdm import tqdm
import matplotlib.pyplot as plt


def method_a_ImageNet():
    train_loader, validation_loader, train_loader_not_augmented = load_imageNet()

    # ImageNet
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), './pretrained_resnet_models_imagenet/resnet18.pth')
    original_teacher_dict = torch.load('./pretrained_resnet_models_imagenet/resnet18.pth')
    print('pretrained model loaded')
    teacher_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut']('full_precision', 'ImageNet', factorized_gamma=True)
    teacher_checkpoint = change_loaded_checkpoint(original_teacher_dict, teacher_ResNet18)
    teacher_ResNet18.load_state_dict(teacher_checkpoint)

    net_type = 'Xnor++'
    student_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)
    student_checkpoint = change_loaded_checkpoint(original_teacher_dict, student_ResNet18)
    student_ResNet18.load_state_dict(student_checkpoint)

    if torch.cuda.is_available():
        teacher_ResNet18 = teacher_ResNet18.cuda(device=get_device_id())
        student_ResNet18 = student_ResNet18.cuda(device=get_device_id())

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet18)))

    learning_rate_change = [2, 4, 5, 6]

    filename = 'resnet18_method_a_training'
    training_a(student_ResNet18, teacher_ResNet18, train_loader, validation_loader, train_loader_not_augmented, filename=filename,
               modified=True, max_epoch_layer=6, max_epoch_finetuning=0, learning_rate_change=learning_rate_change, saved_training=None)


def imagenet_without_pre_training():

    train_loader, validation_loader, train_loader_not_augmented = load_imageNet()

    # ImageNet
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), './pretrained_resnet_models_imagenet/resnet18.pth')
    original_teacher_dict = torch.load('./pretrained_resnet_models_imagenet/resnet18.pth')
    print('pretrained model loaded')

    net_type = 'Xnor++'
    student_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)
    student_checkpoint = change_loaded_checkpoint(original_teacher_dict, student_ResNet18)
    student_ResNet18.load_state_dict(student_checkpoint)

    if torch.cuda.is_available():
        student_ResNet18 = student_ResNet18.cuda(device=get_device_id())

    lr = 1e-3
    learning_rate_change = [15, 20, 23]

    filename = 'resnet18_finetuning_no_pretraining'

    finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_augmented, 25, learning_rate_change,
                   path=None, filename=filename, saved_model=None, initial_learning_rate=lr, saved_training='./saved_training/ImageNet/resnet18_finetuning_no_pretraining_lr0.001_20200512')



def finetuning_no_method():

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))


    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=True)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda(device=get_device_id())

    filename = 'resnet20_xnor++_factorized_double_shortcut_finetuning_no_method'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)


def method_b_training():

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    scaling_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for scaling_factor in scaling_factors:
        net_type = 'Xnor++'
        student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                 factorized_gamma=True)
        new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
        student_ResNet20.load_state_dict(new_checkpoint_student)
        if torch.cuda.is_available():
            student_ResNet20 = student_ResNet20.cuda(device=get_device_id())

        filename = 'resnet20_xnor++_factorized_double_shortcut_training_b_'
        lit_training(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, max_epochs=60,
                     teacher_net=teacher_ResNet20, filename=filename, scaling_factor_total=scaling_factor, scaling_factor_kd=0.95)

        filename = 'resnet20_xnor++_factorized_double_shortcut_training_b_finetuning_scaling_factor_' + str(scaling_factor)
        finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
                   learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2)


def method_c_training():
    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    scaling_factors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    scaling_factors = [0.8, 1]

    for scaling_factor in scaling_factors:
        net_type = 'Xnor++'
        student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                 factorized_gamma=True)
        new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
        student_ResNet20.load_state_dict(new_checkpoint_student)
        if torch.cuda.is_available():
            student_ResNet20 = student_ResNet20.cuda(device=get_device_id())

        if scaling_factor == 0.8:
            filename = 'resnet20_xnor++_factorized_double_shortcut_training_c_finetuning_scaling_factor_' + str(
                scaling_factor)
            finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
                       learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2, saved_training='./saved_training/cifar10/resnet20_xnor++_factorized_double_shortcut_training_c_finetuning_scaling_factor_0.8_lr0.01_20200510')
        else:
            filename = 'resnet20_xnor++_factorized_double_shortcut_training_c_scaling_factor_' + str(scaling_factor)
            training_c(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented, filename=filename, max_epochs=120,
                           scaling_factor_total=scaling_factor)

            filename = 'resnet20_xnor++_factorized_double_shortcut_training_c_finetuning_scaling_factor_' + str(scaling_factor)
            finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
                       learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2)


def training_network_architecture_method_a():
    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
    #                                                          factorized_gamma=True)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    # filename = 'resnet20_xnor++_factorized_naive_training_a'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=True)
    #
    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20WithRelu'](net_type=net_type, dataset='cifar10',
    #                                                             factorized_gamma=True)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    # filename = 'resnet20_xnor++_factorized_with_relu_single_training_a'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=True)
    #
    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20Abs'](net_type=net_type, dataset='cifar10',
    #                                                        factorized_gamma=True)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    # filename = 'resnet20_xnor++_factorized_abs_training_a'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=True)

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                          factorized_gamma=True)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    filename = 'resnet20_xnor++_factorized_relu_double_training_a'
    training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
               filename=filename, modified=True, saved_training='./saved_training/cifar10/resnet20_xnor++_factorized_relu_double_training_a_20200509')


def training_a_double_shortcut_and_double_no_method():
    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
    #                                                                       factorized_gamma=True)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    # filename = 'resnet20_xnor++_factorized_relu_double_training_a_2'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=True, learning_rate_change=[25, 30, 35, 39])
    #
    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
    #                                                                       factorized_gamma=True)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    # filename = 'resnet20_xnor++_factorized_relu_double_training_a_not_modified'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=False, learning_rate_change=[25, 30, 35, 39])


    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                          factorized_gamma=True)
    trained_student_checkpoint = torch.load('./Trained_Models/cifar10/resnet20_xnor++_factorized_double_shortcut_finetuning_no_method_20200510.pth')
    new_checkpoint_student = change_loaded_checkpoint(trained_student_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(trained_student_checkpoint)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    filename = 'resnet20_xnor++_factorized_relu_double_no_method_times2'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
               max_epochs=120, learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=0.01)


def different_architectures_method_c():
    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    scaling_factor = 0.4

    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
    #                                                                       factorized_gamma=False)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    #
    # filename = 'resnet20_xnor++_non_factorized_double_shortcut_training_c_scaling_factor_' + str(scaling_factor)
    # training_c(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, max_epochs=120,
    #            scaling_factor_total=scaling_factor)
    #
    # filename = 'resnet20_xnor++_non_factorized_double_shortcut_training_c_finetuning_scaling_factor_' + str(
    #     scaling_factor)
    # finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2)
    #
    #
    #
    # net_type = 'Xnor'
    # student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
    #                                                                       factorized_gamma=False)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda(device=get_device_id())
    #
    # filename = 'resnet20_xnor_double_shortcut_training_c_scaling_factor_' + str(scaling_factor)
    # training_c(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, max_epochs=120,
    #            scaling_factor_total=scaling_factor)
    #
    # filename = 'resnet20_xnor_double_shortcut_training_c_finetuning_scaling_factor_' + str(
    #     scaling_factor)
    # finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2)
    #
    net_type = 'binary'
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                        factorized_gamma=False)
    #new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    #student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda(device=get_device_id())

    # filename = 'resnet20_binary_double_shortcut_training_c_scaling_factor_' + str(scaling_factor)
    # training_c(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, max_epochs=120,
    #            scaling_factor_total=scaling_factor)

    checkp = torch.load('./Trained_Models/cifar10/resnet20_xnor_double_shortcut_training_c_finetuning_scaling_factor_0.4_20200511.pth', map_location='cpu')
    student_ResNet20.load_state_dict(checkp)
    filename = 'resnet20_binary_double_shortcut_training_c_finetuning_scaling_factor_' + str(
        scaling_factor)
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename, initial_learning_rate=1e-2)


def get_mean_and_std_at_layer():

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True,subsets=True)


    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda(device=get_device_id())
    teacher_ResNet20.eval()

    mean_7 = 0
    std_7 = 0
    mean_13 = 0
    std_13 = 0
    mean_19 = 0
    std_19 = 0

    i = 0
    for i, (images, _) in enumerate(tqdm(train_loader)):

        batch_samples = images.size(0)
        i +=1

        with torch.no_grad():
            res_layer_7 = teacher_ResNet20(images, cut_network=7).view(-1)
            res_layer_13 = teacher_ResNet20(images, cut_network=13).view(-1)
            res_layer_19 = teacher_ResNet20(images, cut_network=19).view(-1)

        mean_7 += res_layer_7.mean().sum(0)
        std_7 += res_layer_7.std().sum(0)
        mean_13 += res_layer_13.mean().sum(0)
        std_13 += res_layer_13.std().sum(0)
        mean_19 += res_layer_19.mean().sum(0)
        std_19 += res_layer_19.std().sum(0)

    mean_7 /= len(train_loader)
    mean_13 /= len(train_loader)
    mean_19 /= len(train_loader)

    std_7 /= len(train_loader)
    std_13 /= len(train_loader)
    std_19 /= len(train_loader)

    print('mean 7: ' + str(mean_7))
    print('mean 13: ' + str(mean_13))
    print('mean 19: ' + str(mean_19))
    print('std 7: ' + str(std_7))
    print('std 13: ' + str(std_13))
    print('std 19: ' + str(std_19))


def training_c_imagenet():

    train_loader, validation_loader, train_loader_not_augmented = load_imageNet()

    # ImageNet
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), './pretrained_resnet_models_imagenet/resnet18.pth')
    original_teacher_dict = torch.load('./pretrained_resnet_models_imagenet/resnet18.pth')
    print('pretrained model loaded')
    teacher_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut']('full_precision', 'ImageNet',
                                                                          factorized_gamma=True)
    teacher_checkpoint = change_loaded_checkpoint(original_teacher_dict, teacher_ResNet18)
    teacher_ResNet18.load_state_dict(teacher_checkpoint)

    net_type = 'Xnor++'
    student_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)
    student_checkpoint = change_loaded_checkpoint(original_teacher_dict, student_ResNet18)
    student_ResNet18.load_state_dict(student_checkpoint)

    if torch.cuda.is_available():
        teacher_ResNet18 = teacher_ResNet18.cuda(device=get_device_id())
        student_ResNet18 = student_ResNet18.cuda(device=get_device_id())

    #print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet18)))

    learning_rate_change = [2, 4, 5]

    filename = 'resnet18_method_c_training'
    training_c(student_ResNet18, teacher_ResNet18, train_loader, validation_loader, train_loader_not_augmented, filename=filename,
               max_epochs=20, scaling_factor_total=0.4, max_epoch_layer=5, learning_rate_change=learning_rate_change, saved_training='./saved_training/ImageNet/resnet18_method_c_training_20200512')


def main():
    #training_network_architecture_method_a()

    #method_c_training()
    #method_b_training()

    #finetuning_no_method()
    #method_a_ImageNet()
    #imagenet_without_pre_training()
    #training_a_double_shortcut_and_double_no_method()
    #different_architectures_method_c()

    #get_mean_and_std_at_layer()

    plt.close('all')
    training_c_imagenet()





if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
