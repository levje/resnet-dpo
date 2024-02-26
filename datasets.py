import torch
from torchvision import datasets, transforms

def load_imagenet(batch_size: int, test: bool = False, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download the datasets
    trainset = None
    if not test:
        trainset = datasets.ImageNet(root='./data/imagenet',
                                     transform=transform,
                                     train=True)

    testset = datasets.ImageNet(root='./data/imagenet',
                                split='val',
                                transform=transform)

    return _get_dataloders(trainset, testset, batch_size, num_workers), testset.classes

def load_cifar10(batch_size: int, test: bool = False, num_workers: int = 2):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = None
    if not test:
        # Download the datasets
        trainset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform)
    testset = datasets.CIFAR10(root='./data',
                               train=False,
                               download=True,
                               transform=transform)

    return _get_dataloders(trainset, testset, batch_size, num_workers), testset.classes

def _get_dataloders(
        trainset: datasets.VisionDataset,
        testset: datasets.VisionDataset,
        batch_size: int,
        num_workers: int):

    # Data loaders (should pin_memory if not on Mac)
    trainloader = None
    if trainset is not None:
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader