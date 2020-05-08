from models import resNet
import torchvision.models as models
import warnings
from binaryUtils import *
from loadUtils import load_cifar10
from train import finetuning, training_a, lit_training, training_c, training_kd


def main():
    net_type = 'Xnor'             # 'full_precision', 'binary', 'binary_with_alpha', 'Xnor' or 'Xnor++'

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)


    student_ResNet18 = resNet.resnet_models['resnet18Naive'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=True)
    if torch.cuda.is_available():
        student_ResNet18 = student_ResNet18.cuda()
    filename = 'resnet18_xnor_naive_finetuning'
    finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)

    net_type = 'Xnor'
    student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=True)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet18.cuda()
    filename = 'resnet20_xnor_naive_finetuning'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=True)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet18.cuda()
    filename = 'resnet20_xnor++_factorized_naive_finetuning'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)

    net_type = 'Xnor++'
    student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=False)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet18.cuda()
    filename = 'resnet20_xnor++_non_factorized_naive_finetuning'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)

    net_type = 'binary'
    student_ResNet20 = resNet.resnet_models['resnet20Naive'](net_type=net_type, dataset='cifar10',
                                                             factorized_gamma=False)
    if torch.cuda.is_available():
        student_ResNet20 = student_ResNet18.cuda()
    filename = 'resnet20_binary_naive_finetuning'
    finetuning(student_ResNet20, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 90, 100, 110], filename=filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
