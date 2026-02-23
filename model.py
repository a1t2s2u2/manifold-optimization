import torch
import torch.nn as nn
from stiefel import retract_qr


class MLP_Stiefel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes, bias=False)

        # 初期化後に Stiefel 上へ乗せる（W^T W = I）
        with torch.no_grad():
            self.fc.weight.copy_(retract_qr(self.fc.weight.T).T)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LinearModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNN_Stiefel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * 12 * 12, num_classes, bias=False)

        # 初期化後に Stiefel 上へ乗せる（W^T W = I）
        with torch.no_grad():
            self.fc.weight.copy_(retract_qr(self.fc.weight.T).T)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * 12 * 12, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
