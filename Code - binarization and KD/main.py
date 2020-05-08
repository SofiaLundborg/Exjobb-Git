from models import resNet
import torchvision.models as models
import warnings
from binaryUtils import *
from extraUtils import change_loaded_checkpoint, calculate_accuracy
from loadUtils import load_cifar10
from train import finetuning, training_a, lit_training, training_c, training_kd


def main():
    net_type = 'Xnor++'             # 'full_precision', 'binary', 'binary_with_alpha', 'Xnor' or 'Xnor++'

    train_loader, validation_loader, test_loader, train_loader_not_augmented = load_cifar10(test_as_validation=True)

    student_ResNet18 = resNet.resnet_models['resnet18Naive'](net_type=net_type, dataset='cifar10', factorized_gamma=True)
    if torch.cuda.is_available():
        student_ResNet18 = student_ResNet18.cuda()

    filename = 'xnor++_with_bias_naive'
    finetuning(student_ResNet18, train_loader, validation_loader, train_loader_not_augmented, 120,
               learning_rate_change=[70, 100, 110], filename=filename)

    teacher_ResNet20 = resNet.resnet_models['resnet20ForTeacher'](net_type='full_precision', dataset='cifar10')
    student_ResNet20 = resNet.resnet_models['resnet20ReluDoubleShortcut'](net_type=net_type, dataset='cifar10', factorized_gamma=False)
    # load pretrained network into student and techer network
    teacher_pth = './pretrained_resnet_cifar10_models/student/' + 'resnet20' + '.pth'
    teacher_checkpoint = torch.load(teacher_pth, map_location='cpu')
    new_checkpoint_teacher = change_loaded_checkpoint(teacher_checkpoint, teacher_ResNet20)
    new_checkpoint_student = change_loaded_checkpoint(teacher_checkpoint, student_ResNet20)
    teacher_ResNet20.load_state_dict(new_checkpoint_teacher)
    student_ResNet20.load_state_dict(new_checkpoint_student)
    if torch.cuda.is_available():
        teacher_ResNet20 = teacher_ResNet20.cuda()
        student_ResNet20 = student_ResNet20.cuda()



    # training_kd(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, train_loader, filename=filename, saved_training=None, max_epochs=110)

    #training_a(student_ResNet20, teacher_ResNet20, train_loader, validation_loader, filename=filename, saved_training=None,
    #           modified=False)


    # ImageNet
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18.state_dict(), './pretrained_resnet_models_imagenet/resnet18.pth')
    original_teacher_dict = torch.load('./pretrained_resnet_models_imagenet/resnet18.pth')
    print('pretrained model loaded')
    teacher_ResNet18 = resNet.resnet_models['resnet18ReluDoubleShortcut'](net_type, 'ImageNet', factorized_gamma=True)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency; partially "
                                              "transparent artists will be rendered opaque.")
    main()
