import json
import os

import matplotlib.pyplot as plt

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "X"]


def save_graphs(save_dir, epochs, results: dict):
    """Loss と Accuracy のグラフを保存する。results は {label: {"loss": [...], "accuracy": [...]}} 形式。"""
    # Loss グラフ
    plt.figure()
    for i, (label, hist) in enumerate(results.items()):
        plt.plot(epochs, hist["loss"], marker=MARKERS[i % len(MARKERS)], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy グラフ
    plt.figure()
    for i, (label, hist) in enumerate(results.items()):
        plt.plot(epochs, hist["accuracy"], marker=MARKERS[i % len(MARKERS)], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_log(save_dir, config, results: dict):
    """学習ログを JSON として保存する。"""
    log = {
        "config": config,
        "results": results,
    }
    with open(os.path.join(save_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
