import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP_Stiefel, LinearModel, CNN_Stiefel, CNN
from optimizer import StiefelSGD


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

    def worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % 2**32 + worker_id)

    generator = torch.Generator()
    generator.manual_seed(0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=pin, worker_init_fn=worker_init_fn, generator=generator)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def train(device="cpu", epochs=1, batch_size=256, lr=0.1, use_stiefel=True, dataset="mnist",
          train_loader=None, test_loader=None):
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
    if info["model_type"] == "cnn":
        if use_stiefel:
            model = CNN_Stiefel(num_classes=info["num_classes"], in_channels=info["in_channels"], input_size=info["input_size"]).to(device)
            feature_params = [p for p in model.parameters() if p is not model.fc.weight]
            opt = StiefelSGD([
                {'params': feature_params, 'stiefel': False},
                {'params': [model.fc.weight], 'stiefel': True},
            ], lr=lr)
        else:
            model = CNN(num_classes=info["num_classes"], in_channels=info["in_channels"], input_size=info["input_size"]).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        input_dim = info["in_channels"] * info["input_size"] ** 2
        if use_stiefel:
            model = MLP_Stiefel(num_classes=info["num_classes"], input_dim=input_dim).to(device)
            opt = StiefelSGD([
                {'params': [model.fc.weight], 'stiefel': True},
            ], lr=lr)
        else:
            model = LinearModel(num_classes=info["num_classes"], input_dim=input_dim).to(device)
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
