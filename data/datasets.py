import torch
from torchvision import datasets, transforms
from random import randrange

# You need to download the imagenet dataset yourself and put it in the root directory.
def load_imagenet(batch_size: int, test: bool = False, num_workers: int = 2, train_ratio: float = 0.8):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download the datasets
    trainset = None
    if not test:
        trainset = datasets.ImageNet(root='./imagenet',
                                     transform=transform,
                                     train=True)

    testset = datasets.ImageNet(root='./imagenet',
                                split='val',
                                transform=transform)

    return _get_dataloders(trainset, testset, batch_size, num_workers, train_ratio), testset.classes

def load_dpo_cifar10(batch_size: int, test: bool = False, num_workers: int = 2, train_ratio: float = 0.8, data_augment: bool = True):
    base_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )

    target_transform = lambda x: (x, randrange(10))

    trainset = None
    if not test:
        train_transform = base_transform
        if data_augment:
            train_transform = transforms.Compose([data_augment_transform, base_transform])
        # Download the datasets
        trainset = datasets.CIFAR10(root='.',
                                    train=True,
                                    download=True,
                                    transform=train_transform,
                                    target_transform=target_transform)
    testset = datasets.CIFAR10(root='.',
                               train=False,
                               download=True,
                               transform=base_transform,
                               target_transform=target_transform)

    return _get_dataloders(trainset, testset, batch_size, num_workers, train_ratio), testset.classes

def load_cifar10(batch_size: int, test: bool = False, num_workers: int = 2, train_ratio: float = 0.8, data_augment: bool = True):
    base_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )

    trainset = None
    if not test:
        train_transform = base_transform
        if data_augment:
            train_transform = transforms.Compose([data_augment_transform, base_transform])
        # Download the datasets
        trainset = datasets.CIFAR10(root='.',
                                    train=True,
                                    download=True,
                                    transform=train_transform)
    testset = datasets.CIFAR10(root='.',
                               train=False,
                               download=True,
                               transform=base_transform)

    return _get_dataloders(trainset, testset, batch_size, num_workers, train_ratio), testset.classes

def _get_dataloders(
        trainset: datasets.VisionDataset,
        testset: datasets.VisionDataset,
        batch_size: int,
        num_workers: int,
        train_ratio: float=0.8):

    # Data loaders (should pin_memory if not on Mac)
    trainloader = None
    validloader = None
    if trainset is not None:
        # Split the training set into training and validation
        train_size = int(train_ratio * len(trainset))
        valid_size = len(trainset) - train_size
        trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        validloader = torch.utils.data.DataLoader(validset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)


    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return trainloader, validloader, testloader