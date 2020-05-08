from models import resNet
import torchvision.models as models
import warnings
from binaryUtils import *
from loadUtils import load_cifar10
from train import finetuning, training_a, lit_training, training_c, training_kd


def main():
    net_type = 'Xnor'             # 'full_precision', 'binary', 'binary_with_alpha', 'Xnor' or 'Xnor++'

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    student_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type=net_type, dataset='cifar10',
                                                                          factorized_gamma=True)
    if torch.cuda.is_available():
        student_ResNet18 = student_ResNet18.cuda()

    filename = 'xnor_double_shortcut_not_binarized_first_layer_bin_conv_in_shortcut'
    finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_augmented, 100,
               learning_rate_change=[50, 70, 90], filename=filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
