import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from stiefel import retract_qr, project_to_tangent
from model import MLP_Stiefel, LinearModel


def train(device="cpu", epochs=1, batch_size=256, lr=0.1, use_stiefel=True, dataset="mnist"):
    # データ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds_cls = datasets.FashionMNIST if dataset == "fashion" else datasets.MNIST
    train_ds = ds_cls(root="./data", train=True, download=True, transform=transform)
    test_ds  = ds_cls(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device!="cpu"))
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2, pin_memory=(device!="cpu"))

    # モデル
    if use_stiefel:
        model = MLP_Stiefel(num_classes=10).to(device)
    else:
        model = LinearModel(num_classes=10).to(device)
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

        acc = evaluate()
        print(f"epoch {ep} | loss {total_loss/len(train_loader):.4f} | test acc {acc:.4f}")

    return model
