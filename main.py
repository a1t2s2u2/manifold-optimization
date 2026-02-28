import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="dtype\\(\\).*align")

from save import save_graphs, save_log
from train import load_data, train, precompute_spd_features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATASET_CONFIGS = {
    "mnist":   {"epochs": 20, "lr": 0.1,  "lr_spd": 0.01, "batch_size": 256, "model": "linear"},
    "fashion": {"epochs": 20, "lr": 0.1,  "lr_spd": 0.01, "batch_size": 256, "model": "linear"},
    "cifar10": {"epochs": 20, "lr": 0.05, "lr_spd": 0.005, "batch_size": 256, "model": "cnn"},
    "stl10":   {"epochs": 20, "lr": 0.05, "lr_spd": 0.005, "batch_size": 256, "model": "cnn"},
}

# 実験パターン定義: (label, use_spd, use_stiefel, lr_key)
EXPERIMENT_TYPES = {
    "stiefel_vs_sgd": [
        ("Stiefel",      False, True,  "lr"),
        ("SGD",          False, False, "lr"),
    ],
    "spd_vs_pixel": [
        ("SPD+SGD",      True,  False, "lr_spd"),
        ("Pixel+SGD",    False, False, "lr"),
    ],
    "spd_stiefel_sgd": [
        ("SPD",          True,  False, "lr_spd"),
        ("Stiefel",      False, True,  "lr"),
        ("SGD",          False, False, "lr"),
    ],
    "spd_stiefel_vs_sgd": [
        ("SPD+Stiefel",  True,  True,  "lr_spd"),
        ("SPD+SGD",      True,  False, "lr_spd"),
    ],
    "full": [
        ("SPD+Stiefel",  True,  True,  "lr_spd"),
        ("SPD+SGD",      True,  False, "lr_spd"),
        ("Stiefel",      False, True,  "lr"),
        ("SGD",          False, False, "lr"),
    ],
}

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = "fashion"            # ← データセットを変える
    experiment = "spd_stiefel_sgd" # ← 実験パターンを変える
    cfg = DATASET_CONFIGS[dataset]
    experiments = EXPERIMENT_TYPES[experiment]

    # データ読み込み
    set_seed(42)
    train_loader, test_loader = load_data(dataset, cfg["batch_size"], device)

    # SPD 特徴量が必要なら1回だけ事前計算
    needs_spd = any(exp[1] for exp in experiments)
    spd_train_loader, spd_test_loader, spd_feat_dim = None, None, None
    if needs_spd:
        spd_train_loader, spd_test_loader, spd_feat_dim = precompute_spd_features(
            train_loader, test_loader, cfg["batch_size"]
        )

    # 各実験を実行
    results = {}
    for label, use_spd, use_stiefel, lr_key in experiments:
        set_seed(42)
        print(f"\n=== {label} ===")
        if use_spd:
            _, hist = train(
                device=device, epochs=cfg["epochs"], dataset=dataset,
                batch_size=cfg["batch_size"], lr=cfg[lr_key], use_stiefel=use_stiefel,
                train_loader=spd_train_loader, test_loader=spd_test_loader,
                use_spd=True, input_dim=spd_feat_dim,
            )
        else:
            _, hist = train(
                device=device, epochs=cfg["epochs"], dataset=dataset,
                batch_size=cfg["batch_size"], lr=cfg[lr_key], use_stiefel=use_stiefel,
                train_loader=train_loader, test_loader=test_loader,
            )
        results[label] = hist

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, cfg["epochs"] + 1))
    config = {
        "dataset": dataset,
        "experiment": experiment,
        "model": cfg["model"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "lr_spd": cfg["lr_spd"],
        "device": device,
    }

    save_graphs(save_dir, epochs, results)
    save_log(save_dir, config, results)

    print(f"\n保存完了: {save_dir}/")
