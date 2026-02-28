import torch
import torch.nn as nn
from stiefel import retract_qr


DATASET_INFO = {
    "mnist":   {"num_classes": 10, "in_channels": 1, "input_size": 28},
    "fashion": {"num_classes": 10, "in_channels": 1, "input_size": 28},
    "cifar10": {"num_classes": 10, "in_channels": 3, "input_size": 32},
    "stl10":   {"num_classes": 10, "in_channels": 3, "input_size": 96},
}


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, in_channels, input_size, num_classes):
        super().__init__()
        flat_dim = 8 * (input_size // 8) ** 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(flat_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def init_stiefel(model):
    """model.fc.weight を Stiefel 多様体上に射影する。"""
    with torch.no_grad():
        model.fc.weight.copy_(retract_qr(model.fc.weight.T).T)


def make_model(model_cls, dataset, feature, stiefel, spd_dim=None):
    """モデルを生成するファクトリ関数。model_cls に MLP / CNN クラスを直接渡す。"""
    info = DATASET_INFO[dataset]

    if model_cls is CNN:
        model = CNN(
            in_channels=info["in_channels"],
            input_size=info["input_size"],
            num_classes=info["num_classes"],
        )
    else:
        input_dim = spd_dim if feature == "spd" else info["in_channels"] * info["input_size"] ** 2
        model = model_cls(input_dim=input_dim, num_classes=info["num_classes"])

    if stiefel:
        init_stiefel(model)

    return model
