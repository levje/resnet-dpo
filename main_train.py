import argparse
from trainer import Trainer
from resnet_cifar import ResnetCifar
from datasets import load_cifar10
import torch

from utils.torch_utils import get_default_device

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the ResNet model on CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--log_interval', type=int, default=2, help='Number of batches to wait before logging')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    return parser

def main(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    log_interval = args.log_interval
    lr = args.lr

    (trainloader, validloader, testloader), classes = load_cifar10(batch_size, num_workers=num_workers, train_ratio=0.8)
    print("CIFAR {} classes: {}".format(len(classes), classes))
    print("Train size: {}, Valid size: {}, Test size: {}".format(len(trainloader), len(validloader), len(testloader)))

    model = ResnetCifar(n_classes=len(classes))
    device = get_default_device()
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, trainloader, validloader, testloader, loss_func, lr=lr, optimizer='adam')
    model, learn_hists, best_epoch = trainer.train_model(num_epochs=num_epochs)

    trainer.test_model()


if '__main__' == __name__:
    parser = build_parser()
    main(parser.parse_args())