import torch
import torchvision
import torch.nn as nn

class ResnetCifar(nn.Module):
    def __init__(self, n_classes: int, model_path: str = None):
        super().__init__()
        self.n_classes = n_classes

        
        if model_path is not None:
            self.model = torchvision.models.resnet50()
        else:
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        
        # Freeze everything
        # for param in self.model.parameters():
        #     param.requires_grad = False

        out_size = self.model.fc.in_features
        self.model.fc = nn.Linear(out_size, self.n_classes)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path)) # TODO: Why are we using strict=False?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
