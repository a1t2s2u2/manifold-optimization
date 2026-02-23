import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from stiefel import retract_qr, project_to_tangent
from model import MLP_Stiefel, LinearModel, CNN_Stiefel, CNN


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
    pin = device != "cpu"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def train(device="cpu", epochs=1, batch_size=256, lr=0.1, use_stiefel=True, dataset="mnist",
          train_loader=None, test_loader=None):
    # データ（外部から渡されなければ内部で読み込む）
    if train_loader is None or test_loader is None:
        train_loader, test_loader = load_data(dataset, batch_size, device)

    # モデル（データセットに応じて自動選択）
    num_classes = 10
    # STL-10: 96x96 → 12x12, CIFAR-10: 32x32 → 4x4 (3回の MaxPool2d(2))
    flat_dims = {"stl10": 128 * 12 * 12, "cifar10": 128 * 4 * 4}
    if dataset in flat_dims:
        flat_dim = flat_dims[dataset]
        if use_stiefel:
            model = CNN_Stiefel(num_classes=num_classes, flat_dim=flat_dim).to(device)
        else:
            model = CNN(num_classes=num_classes, flat_dim=flat_dim).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        if use_stiefel:
            model = MLP_Stiefel(num_classes=num_classes).to(device)
        else:
            model = LinearModel(num_classes=num_classes).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=lr)

    # 評価
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / total

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

            if use_stiefel:
                # Stiefel 上でリーマン勾配により手動更新
                if model.fc.weight.grad is not None:
                    model.fc.weight.grad.zero_()
                loss.backward()

                with torch.no_grad():
                    W = model.fc.weight.T
                    G = model.fc.weight.grad.T
                    rgrad = project_to_tangent(W, G)
                    W_new = W - lr * rgrad
                    W_new = retract_qr(W_new)
                    model.fc.weight.copy_(W_new.T)
            else:
                # 通常の SGD で更新
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
