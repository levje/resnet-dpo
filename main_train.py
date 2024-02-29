import argparse
from trainer import Trainer
from resnet_cifar import ResnetCifar
from datasets import load_cifar10
import torch

from utils.torch_utils import get_default_device

def main(args):
    batch_size = 16
    num_workers = 2
    num_epochs = 10
    log_interval = 2

    (trainloader, validloader, testloader), classes = load_cifar10(batch_size, num_workers=num_workers, train_ratio=0.8)
    print("CIFAR {} classes: {}".format(len(classes), classes))
    print("Train size: {}, Valid size: {}, Test size: {}".format(len(trainloader), len(validloader), len(testloader)))

    model = ResnetCifar(n_classes=len(classes))
    device = get_default_device()
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, trainloader, validloader, testloader, loss_func, lr=0.001, optimizer='adam')
    model, learn_hists, best_epoch = trainer.train_model(num_epochs=1)


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Train the ResNet model on CIFAR-10 dataset')
    main(parser.parse_args())