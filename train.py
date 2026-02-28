import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from model import MLP_Stiefel, MLP_Stiefel_Hidden, LinearModel, MLP, CNN_Stiefel, CNN
from optimizer import StiefelSGD
from spd import spd_log_euclidean_features


def _worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)


def load_data(dataset="mnist", batch_size=256, device="cpu"):
    """データセットをダウンロード・読み込みし DataLoader を返す。"""
    if dataset == "stl10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_ds = datasets.STL10(root="./data", split="train", download=True, transform=transform)
        test_ds  = datasets.STL10(root="./data", split="test", download=True, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds_cls = datasets.FashionMNIST if dataset == "fashion" else datasets.MNIST
        train_ds = ds_cls(root="./data", train=True, download=True, transform=transform)
        test_ds  = ds_cls(root="./data", train=False, download=True, transform=transform)
    pin = device == "cuda"

    generator = torch.Generator()
    generator.manual_seed(0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=pin, worker_init_fn=_worker_init_fn, generator=generator)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def precompute_spd_features(train_loader, test_loader, batch_size=256):
    """全データの SPD 特徴量を事前計算し、新しい DataLoader を返す。"""
    def _extract(loader):
        all_features, all_labels = [], []
        for x, y in loader:
            feat = spd_log_euclidean_features(x)
            all_features.append(feat)
            all_labels.append(y)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    print("SPD 特徴量を事前計算中 (train)...")
    train_feat, train_labels = _extract(train_loader)
    print("SPD 特徴量を事前計算中 (test)...")
    test_feat, test_labels = _extract(test_loader)

    feat_dim = train_feat.shape[1]
    print(f"SPD 特徴量次元: {feat_dim}")

    spd_train_loader = DataLoader(
        TensorDataset(train_feat, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    spd_test_loader = DataLoader(
        TensorDataset(test_feat, test_labels),
        batch_size=1000, shuffle=False,
    )
    return spd_train_loader, spd_test_loader, feat_dim


def train(device="cpu", epochs=1, batch_size=256, lr=0.1, use_stiefel=True, dataset="mnist",
          train_loader=None, test_loader=None, use_spd=False, input_dim=None):
    # データ（外部から渡されなければ内部で読み込む）
    if train_loader is None or test_loader is None:
        train_loader, test_loader = load_data(dataset, batch_size, device)

    # モデル（データセットに応じて自動選択）
    DATASET_INFO = {
        "mnist":   {"num_classes": 10, "in_channels": 1, "input_size": 28, "model_type": "linear"},
        "fashion": {"num_classes": 10, "in_channels": 1, "input_size": 28, "model_type": "linear"},
        "cifar10": {"num_classes": 10, "in_channels": 3, "input_size": 32, "model_type": "cnn"},
        "stl10":   {"num_classes": 10, "in_channels": 3, "input_size": 96, "model_type": "cnn"},
    }
    info = DATASET_INFO[dataset]

    # 全方式で隠れ層付き MLP を使用（公平な比較のため）
    if use_spd:
        if input_dim is None:
            x0, _ = next(iter(train_loader))
            input_dim = x0.shape[1]
    else:
        input_dim = info["in_channels"] * info["input_size"] ** 2

    if use_stiefel:
        model = MLP_Stiefel_Hidden(num_classes=info["num_classes"], input_dim=input_dim).to(device)
        feature_params = [p for p in model.parameters() if p is not model.fc.weight]
        opt = StiefelSGD([
            {'params': feature_params, 'stiefel': False},
            {'params': [model.fc.weight], 'stiefel': True},
        ], lr=lr)
    else:
        model = MLP(num_classes=info["num_classes"], input_dim=input_dim).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)

    # 評価
    def evaluate():
        model.eval()
        correct = torch.tensor(0, device=device)
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum()
                total += y.numel()
        return correct.item() / total

    loss_history = []
    acc_history = []

    # 学習
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate()
        loss_history.append(avg_loss)
        acc_history.append(acc)
        print(f"epoch {ep} | loss {avg_loss:.4f} | test acc {acc:.4f}")

    return model, {"loss": loss_history, "accuracy": acc_history}
