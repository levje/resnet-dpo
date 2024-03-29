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

        self.model = self._modify_model(self.model, self.n_classes)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

    def _modify_model(self, model: torchvision.models.ResNet, n_classes: int) -> torchvision.models.ResNet:
        out_size = model.fc.in_features
        model.fc = nn.Linear(out_size, n_classes)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class ResnetCifarDropout(ResnetCifar):

    def __init__(self, n_classes: int, model_path: str = None):
        super().__init__(n_classes, model_path)

    def _modify_model(self, model: torchvision.models.ResNet, n_classes: int) -> torchvision.models.ResNet:
        model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(model.fc.in_features, model.fc.in_features//2),
            nn.Dropout(0.1),
            nn.Linear(model.fc.in_features//2, n_classes)
        )
        return model
