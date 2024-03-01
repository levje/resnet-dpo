import numpy as np
import torch
import torchvision
import torch.nn as nn
import time
from tempfile import TemporaryDirectory
import os
from tqdm import tqdm

from utils.torch_utils import get_default_device

class Trainer(object):
    def __init__(self, model, trainloader, validloader, testloader, criterion, lr, optimizer: str = 'adam'):
        self.device = get_default_device()
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.criterion = criterion

        self.trainsize = len(self.trainloader.dataset)
        self.validsize = len(self.validloader.dataset)
        self.testsize = len(self.testloader.dataset)

        self.best_model_params_path = None

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.99, eps=1e-08)
        else:
            raise ValueError(f'Unknown optimizer {optimizer}')

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.best_acc = 0.0
        self.best_epoch = 0.0
        self.learn_hists = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def test_model(self):
        self.model.eval()

        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.float() / self.testsize

        print(f'Test Acc: {epoch_acc:.4f}')

    def train_model(self, num_epochs=25) -> (nn.Module, dict):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                train_loss, train_acc = self._train_loop_epoch()
                print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

                val_loss, val_acc = self._validation_loop()
                print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

                self.learn_hists['train_loss'].append(train_loss)
                self.learn_hists['train_acc'].append(train_acc)
                self.learn_hists['val_loss'].append(val_loss)
                self.learn_hists['val_acc'].append(val_acc)

                # Save the best model found so far. Helpful if training is interrupted or if there's
                # overfitting along the way
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_params_path)

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {self.best_acc:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path))

        return self.model, self.learn_hists, self.best_epoch

    def _train_loop_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_corrects = 0

        with tqdm(range(len(self.trainloader))) as pbar:
            for batch_index, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_corrects += torch.sum(preds == labels.data)
                # self.scheduler.step()

                batch_loss = total_loss / (batch_index + 1)
                batch_acc = total_corrects.float() / ((batch_index + 1) * inputs.size(0))

                pbar.update()
                pbar.set_postfix({'loss': batch_loss, 'acc': batch_acc})

        average_loss = total_loss / len(self.trainloader)
        average_acc = total_corrects.float() / len(self.trainloader.dataset)

        return average_loss, average_acc

    def _validation_loop(self):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.trainsize
        epoch_acc = running_corrects.float() / self.trainsize

        return epoch_loss, epoch_acc
