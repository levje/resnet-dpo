import torch
import torchvision
import torch.nn as nn

class ResnetCifar(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        self.model = torchvision.models.resnet34(pretrained=True)

        # Freeze the original model, since for Cifar it should be decent
        for param in self.model.parameters():
            param.requires_grad = False

        out_size = self.model.fc.in_features
        self.model.fc = nn.Linear(out_size, self.n_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
