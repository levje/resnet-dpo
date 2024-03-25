import argparse
from trainers.preference_trainer import PreferenceTrainer
from resnet_cifar import ResnetCifar, ResnetCifarDropout
from datasets import load_dpo_cifar10
from loss import DPOLoss
import torch
from utils.logger import Logger
import os

from utils.torch_utils import get_default_device
from utils.utils import save_learn_hists, visualize_learn_hists, load_learn_hists

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the ResNet model on CIFAR-10 dataset')
    parser.add_argument('exp_name', type=str, help='Name of the experiment')
    parser.add_argument('ref_model_path', type=str, help='Path to the reference model')
    parser.add_argument('policy_model_path', type=str, help='Path to the policy model')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--do_polyak', action='store_true', help='Use Polyak averaging.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save the logs')
    parser.add_argument('--log_interval', type=int, default=2, help='Number of batches to wait before logging')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for the optimizer')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save the results')
    parser.add_argument('--force', action='store_true', help='Force overwrite the results directory')
    return parser

def main(args):
    batch_size = args.batch_size
    do_polyak = args.do_polyak
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    log_interval = args.log_interval
    exp_name = args.exp_name
    lr = args.lr
    ref_model_path = args.ref_model_path
    policy_model_path = args.policy_model_path
    results_dir = os.path.join(args.results_dir, exp_name)
    force = args.force

    if force and os.path.exists(results_dir):
        # remove the directory and its contents
        os.system(f'rm -rf {results_dir}')
    
    os.mkdir(results_dir)

    lr_string = str(lr).replace('.', '')
    model_save_file = f'{results_dir}/dpo_cifar10_{num_epochs}_{lr_string}_{exp_name}.pt'
    log_file = f'{results_dir}/dpo_cifar10_e{num_epochs}_{lr_string}_{exp_name}.log'
    logger = Logger(log_file)

    # Load the CIFAR-10 dataset adapter with win/lose labels
    (trainloader, validloader, testloader), classes = load_dpo_cifar10(batch_size, num_workers=num_workers, train_ratio=0.8)
    logger.log("CIFAR {} classes: {}".format(len(classes), classes))
    logger.log("Train size: {}, Valid size: {}, Test size: {}".format(len(trainloader), len(validloader), len(testloader)))

    device = get_default_device()
    model = ResnetCifarDropout(n_classes=len(classes), model_path=policy_model_path).to(device)
    ref_model = ResnetCifarDropout(n_classes=len(classes), model_path=policy_model_path).to(device)
    loss_func = DPOLoss(beta=0.1)

    trainer = PreferenceTrainer(model, ref_model, trainloader, validloader, testloader, loss_func, logger=logger, lr=lr, optimizer='adam', do_polyak=do_polyak)
    model, learn_hists, best_epoch = trainer.train_model(num_epochs=num_epochs)
    test_acc = trainer.test_model()

    logger.log(f'Best epoch: {best_epoch}/{num_epochs}')
    logger.log(f'Best validation accuracy: {learn_hists["val_acc"][best_epoch]:.4f}')
    logger.log(f'Best training accuracy: {learn_hists["train_acc"][best_epoch]:.4f}')
    logger.log(f'Test accuracy: {test_acc:.4f}')
    logger.log(f'Saving model to {model_save_file}')
    torch.save(model.state_dict(), model_save_file)

    save_learn_hists(learn_hists, f'{results_dir}/dpo_cifar10_{num_epochs}_{lr_string}_{exp_name}.json', 'DPO CIFAR-10')


if '__main__' == __name__:
    parser = build_parser()
    main(parser.parse_args())