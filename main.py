import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from train import train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epoch = 50

    print("=== Stiefel 制約あり ===")
    _, stiefel_hist = train(device=device, epochs=num_epoch, dataset="fashion", batch_size=256, lr=0.05, use_stiefel=True)
    print("=== 制約なし (SGD) ===")
    _, sgd_hist = train(device=device, epochs=num_epoch, dataset="fashion", batch_size=256, lr=0.05, use_stiefel=False)

    # 保存先フォルダ（日付時間）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, num_epoch + 1))

    # Loss グラフ
    plt.figure()
    plt.plot(epochs, stiefel_hist["loss"], marker="o", label="Stiefel")
    plt.plot(epochs, sgd_hist["loss"], marker="s", label="SGD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy グラフ
    plt.figure()
    plt.plot(epochs, stiefel_hist["accuracy"], marker="o", label="Stiefel")
    plt.plot(epochs, sgd_hist["accuracy"], marker="s", label="SGD")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # JSON ログ保存
    log = {
        "config": {
            "epochs": num_epoch,
            "dataset": "fashion",
            "batch_size": 256,
            "lr": 0.05,
            "device": device,
        },
        "stiefel": stiefel_hist,
        "sgd": sgd_hist,
    }
    with open(os.path.join(save_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n保存完了: {save_dir}/")
