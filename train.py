import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from optimizer import StiefelSGD
from spd import spd_log_euclidean_features

logger = logging.getLogger(__name__)


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


def precompute_spd_features(train_loader, test_loader, batch_size=256, target_dim=None):
    """全データの SPD 特徴量を事前計算し、新しい DataLoader を返す。

    target_dim が指定された場合、訓練データで PCA を fit して射影する。
    """
    def _extract(loader, desc):
        all_features, all_labels = [], []
        for x, y in tqdm(loader, desc=desc, leave=False):
            feat = spd_log_euclidean_features(x)
            all_features.append(feat)
            all_labels.append(y)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    logger.info("SPD 特徴量を事前計算中 (train)...")
    train_feat, train_labels = _extract(train_loader, "SPD (train)")
    logger.info("SPD 特徴量を事前計算中 (test)...")
    test_feat, test_labels = _extract(test_loader, "SPD (test)")

    # 訓練データの統計量で標準化（mean=0, std=1）
    mean = train_feat.mean(dim=0)
    std = train_feat.std(dim=0).clamp(min=1e-8)
    train_feat = (train_feat - mean) / std
    test_feat = (test_feat - mean) / std

    # PCA で次元を pixel に揃える
    raw_dim = train_feat.shape[1]
    if target_dim is not None and target_dim < raw_dim:
        logger.info(f"PCA: {raw_dim} → {target_dim}")
        U, S, Vt = torch.linalg.svd(train_feat, full_matrices=False)
        proj = Vt[:target_dim]  # (target_dim, raw_dim)
        train_feat = train_feat @ proj.T
        test_feat = test_feat @ proj.T

    feat_dim = train_feat.shape[1]
    logger.info(f"SPD 特徴量次元: {feat_dim}")

    spd_train_loader = DataLoader(
        TensorDataset(train_feat, train_labels),
        batch_size=batch_size, shuffle=True,
    )
    spd_test_loader = DataLoader(
        TensorDataset(test_feat, test_labels),
        batch_size=1000, shuffle=False,
    )
    return spd_train_loader, spd_test_loader, feat_dim


def make_optimizer(model, lr, use_stiefel):
    """StiefelSGD または標準 SGD を返す。"""
    if use_stiefel:
        feature_params = [p for p in model.parameters() if p is not model.fc.weight]
        return StiefelSGD([
            {'params': feature_params, 'stiefel': False},
            {'params': [model.fc.weight], 'stiefel': True},
        ], lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)


def train_one(model, optimizer, train_loader, test_loader, device, epochs):
    """純粋な学習ループ。loss と accuracy の履歴を返す。"""
    model = model.to(device)

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
    n_batches = len(train_loader)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
    for ep in epoch_bar:
        model.train()
        total_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {ep}", leave=False)
        for step, (x, y) in enumerate(batch_bar, 1):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            logger.debug(f"epoch {ep} step {step}/{n_batches} | loss {loss.item():.4f}")

        avg_loss = total_loss / n_batches
        acc = evaluate()
        loss_history.append(avg_loss)
        acc_history.append(acc)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")
        logger.info(f"epoch {ep} | loss {avg_loss:.4f} | test acc {acc:.4f}")

    return {"loss": loss_history, "accuracy": acc_history}
