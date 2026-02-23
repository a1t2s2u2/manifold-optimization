import os
import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="dtype\\(\\).*align")

from save import save_graphs, save_log
from train import load_data, train

DATASET_CONFIGS = {
    "mnist":   {"epochs": 20, "lr": 0.1,  "batch_size": 256, "model": "linear"},
    "fashion": {"epochs": 20, "lr": 0.1,  "batch_size": 256, "model": "linear"},
    "cifar10": {"epochs": 20, "lr": 0.05, "batch_size": 256, "model": "cnn"},
    "stl10":   {"epochs": 20, "lr": 0.05, "batch_size": 256, "model": "cnn"},
}

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = "stl10"  # ← ここだけ変える
    cfg = DATASET_CONFIGS[dataset]

    # データ読み込み（ダウンロードはここで完了する）
    train_loader, test_loader = load_data(dataset, cfg["batch_size"], device)

    print("=== Stiefel 制約あり ===")
    _, stiefel_hist = train(device=device, epochs=cfg["epochs"], dataset=dataset,
                            batch_size=cfg["batch_size"], lr=cfg["lr"], use_stiefel=True,
                            train_loader=train_loader, test_loader=test_loader)
    print("=== 制約なし (SGD) ===")
    _, sgd_hist = train(device=device, epochs=cfg["epochs"], dataset=dataset,
                        batch_size=cfg["batch_size"], lr=cfg["lr"], use_stiefel=False,
                        train_loader=train_loader, test_loader=test_loader)

    # 保存先フォルダ（日付時間）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, cfg["epochs"] + 1))
    config = {
        "dataset": dataset,
        "model": cfg["model"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "device": device,
    }

    save_graphs(save_dir, epochs, stiefel_hist, sgd_hist)
    save_log(save_dir, config, stiefel_hist, sgd_hist)

    print(f"\n保存完了: {save_dir}/")
