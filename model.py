import torch
import torch.nn as nn
from stiefel import retract_qr


class MLP_Stiefel(nn.Module):
    def __init__(self, num_classes=10, input_dim=28 * 28):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)

        # 初期化後に Stiefel 上へ乗せる（W^T W = I）
        with torch.no_grad():
            self.fc.weight.copy_(retract_qr(self.fc.weight.T).T)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LinearModel(nn.Module):
    def __init__(self, num_classes=10, input_dim=28 * 28):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, num_classes=10, input_dim=28 * 28, hidden_dim=128):
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


class MLP_Stiefel_Hidden(nn.Module):
    def __init__(self, num_classes=10, input_dim=28 * 28, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim, num_classes, bias=False)

        with torch.no_grad():
            self.fc.weight.copy_(retract_qr(self.fc.weight.T).T)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.fc(x)


class CNN_Stiefel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_size=32):
        super().__init__()
        flat_dim = 32 * (input_size // 8) ** 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(flat_dim, num_classes, bias=False)

        # 初期化後に Stiefel 上へ乗せる（W^T W = I）
        with torch.no_grad():
            self.fc.weight.copy_(retract_qr(self.fc.weight.T).T)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_size=32):
        super().__init__()
        flat_dim = 32 * (input_size // 8) ** 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(flat_dim, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
