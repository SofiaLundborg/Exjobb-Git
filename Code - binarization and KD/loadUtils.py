import torchvision
import torchvision.transforms as transforms
import torch
from extraUtils import get_device_id

def load_imageNet(subsets=False):
    normalizing_mean = [0.485, 0.456, 0.406]
    normalizing_std = [0.229, 0.224, 0.225]

    if torch.cuda.is_available():
        batch_size_training = 64    #64
        batch_size_validation = 64  #64
        if get_device_id() == 0:
            batch_size_validation = 64
        else:
            batch_size_validation = 64
    else:
        batch_size_training = 4
        batch_size_validation = 4

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

    train_set = torchvision.datasets.ImageNet(root='./data', split='train', transform=preprocessing_train)
    train_set_not_disturbed = torchvision.datasets.ImageNet(root='./data', split='train', transform=preprocessing_valid)
    print('train set is loaded')
    validation_set = torchvision.datasets.ImageNet(root='./data', split='val', transform=preprocessing_valid)
    print('validation set is loaded')
    if subsets:
        train_set, _ = torch.utils.data.random_split(train_set, [10000, len(train_set) - 10000])
        validation_set, _ = torch.utils.data.random_split(validation_set, [10000, len(validation_set)-10000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_training,
                                               shuffle=True, num_workers=8, pin_memory=True)
    train_loader_not_disturbed = torch.utils.data.DataLoader(train_set_not_disturbed, batch_size=batch_size_validation,
                                                             shuffle=False, num_workers=8, pin_memory=True)
    print('train_loader finished')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_validation,
                                                    shuffle=False, num_workers=8, pin_memory=True)
    print('validation_loader finished')

    return train_loader, validation_loader, train_loader_not_disturbed


def load_cifar10(subsets=False, test_as_validation=False):
    # Load data
    normalizing_mean = [0.485, 0.456, 0.406]        # for ImageNet, used in some models
    normalizing_std = [0.229, 0.224, 0.225]         # for ImageNet, used in some models

    if torch.cuda.is_available():
        batch_size_training = 512
        batch_size_validation = 1024
    else:
        batch_size_training = 4
        batch_size_validation = 4

    normalizing_mean = [0.4914, 0.4822, 0.4465]
    normalizing_std = [0.2470, 0.2435, 0.2616]

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
    train_set_not_augmented = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                           download=True, transform=transform_test)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

    # divide into train and validation data (80% train)
    if not test_as_validation:
        train_size = int(0.8 * len(train_set))
        validation_size = len(train_set) - train_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, validation_size])
    else:
        validation_set = test_set

    # make a small subset of the data
    if subsets:
        train_set, _ = torch.utils.data.random_split(train_set, [200, len(train_set) - 200])
        validation_set, _ = torch.utils.data.random_split(validation_set, [50, len(validation_set)-50])

    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_training,
                                               shuffle=True, num_workers=8, pin_memory=pin_memory)
    train_loader_not_disturbed = torch.utils.data.DataLoader(train_set_not_augmented, batch_size=batch_size_validation,
                                               shuffle=False, num_workers=8, pin_memory=pin_memory)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_validation,
                                                    shuffle=False, num_workers=8, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_validation,
                                              shuffle=False, num_workers=8, pin_memory=pin_memory)

    return train_loader, validation_loader, test_loader, train_loader_not_disturbed


def save_training(epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy, top5_accuracy_train, top5_accuracy_validation, layer_idx, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'train_accuracy': train_accuracy,
        'validation_accuracy': validation_accuracy,
        'top5_train_accuracy': top5_accuracy_train,
        'top5_validation_accuracy': top5_accuracy_validation,
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
    train_accuracy_top5 = checkpoint['top5_train_accuracy']
    validation_accuracy_top5 = checkpoint['top5_validation_accuracy']
    layer_index = checkpoint['layer_index']

    return epoch, model, optimizer, train_loss, validation_loss, train_accuracy, validation_accuracy,train_accuracy_top5,validation_accuracy_top5, layer_index


def load_model_from_saved_training(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_one_sample(data_loader):
    image, targets = next(iter(data_loader))
    return image

