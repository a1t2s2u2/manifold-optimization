import json
import os

import matplotlib.pyplot as plt


def save_graphs(save_dir, epochs, stiefel_hist, sgd_hist):
    """Loss と Accuracy のグラフを保存する。"""
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


def save_log(save_dir, config, stiefel_hist, sgd_hist):
    """学習ログを JSON として保存する。"""
    log = {
        "config": config,
        "stiefel": stiefel_hist,
        "sgd": sgd_hist,
    }
    with open(os.path.join(save_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
