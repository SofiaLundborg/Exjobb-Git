from models import resNet, originalResnet
import torchvision.models as models
import warnings
from binaryUtils import *
from loadUtils import load_cifar10
from train import finetuning, training_a, lit_training, training_c, training_kd
from extraUtils import change_loaded_checkpoint, calculate_accuracy


def main():
    net_type = 'Xnor'             # 'full_precision', 'binary', 'binary_with_alpha', 'Xnor' or 'Xnor++'

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)


    # student_ResNet18 = resNet.resnet_models['resnet18Naive'](net_type=net_type, dataset='cifar10',
    #                                                          factorized_gamma=True)
    # if torch.cuda.is_available():
    #     student_ResNet18 = student_ResNet18.cuda()
    # filename = 'resnet18_xnor_naive_finetuning'
    # finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename)


    # net_type = 'binary'
    # student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
    #                                                          factorized_gamma=False)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda()
    # filename = 'resnet20_binary_naive_finetuning'
    # finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename)

    # net_type = 'Xnor'
    # student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
    #                                                          factorized_gamma=True)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda()
    # filename = 'resnet20_xnor_naive_finetuning'
    # finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename)
    #

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda()
    teacher_ResNet20.eval()

    print('accuracy_teacher: ' + str(calculate_accuracy(validation_loader, teacher_ResNet20)))

    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
    #                                                             factorized_gamma=False)
    # new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    # student_ResNet20.load_state_dict(new_checkpoint_student)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda()
    # filename = 'resnet20_xnor++_factorized_naive_training_a'
    # training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
    #            filename=filename, modified=True, saved_training='./saved_training/cifar10/resnet20_xnor++_factorized_naive_training_a_20200509')

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20WithRelu'](net_type=net_type, dataset='cifar10', factorized_gamma=False)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda()
    filename = 'resnet20_xnor++_factorized_with_relu_single_training_a'
    training_a(student_ResNet20,teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
               filename=filename, modified=True)

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20Abs'](net_type=net_type, dataset='cifar10',
                                                                factorized_gamma=False)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda()
    filename = 'resnet20_xnor++_factorized_abs_training_a'
    training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
               filename=filename, modified=True)

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                           factorized_gamma=False)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet20.cuda()
    filename = 'resnet20_xnor++_factorized_relu_double_training_a'
    training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader_not_augmented,
               filename=filename, modified=True)



    # net_type = 'Xnor++'
    # student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
    #                                                          factorized_gamma=False)
    # if torch.cuda.is_available():
    #     student_ResNet20 = student_ResNet20.cuda()
    # filename = 'resnet20_xnor++_non_factorized_naive_finetuning'
    # finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
    #            learning_rate_change=[70, 90, 100, 110], filename=filename)
    #




if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
