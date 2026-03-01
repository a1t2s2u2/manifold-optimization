import json
import os

import matplotlib.pyplot as plt

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "X"]


def save_graphs(save_dir, epochs, results: dict):
    """Loss と Accuracy のグラフを保存する。

    results は {label: {"train_loss": [...], "test_loss": [...],
                        "train_acc": [...], "test_acc": [...]}} 形式。
    loss.png: Train Loss (左) | Test Loss (右)
    accuracy.png: Train Accuracy (左) | Test Accuracy (右)
    """
    # Loss グラフ (1×2)
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 5))
    for i, (label, hist) in enumerate(results.items()):
        m = MARKERS[i % len(MARKERS)]
        ax_train.plot(epochs, hist["train_loss"], marker=m, label=label)
        ax_test.plot(epochs, hist["test_loss"], marker=m, label=label)
    ax_train.set(xlabel="Epoch", ylabel="Loss", title="Train Loss")
    ax_test.set(xlabel="Epoch", ylabel="Loss", title="Test Loss")
    for ax in (ax_train, ax_test):
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Accuracy グラフ (1×2)
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 5))
    for i, (label, hist) in enumerate(results.items()):
        m = MARKERS[i % len(MARKERS)]
        ax_train.plot(epochs, hist["train_acc"], marker=m, label=label)
        ax_test.plot(epochs, hist["test_acc"], marker=m, label=label)
    ax_train.set(xlabel="Epoch", ylabel="Accuracy", title="Train Accuracy")
    ax_test.set(xlabel="Epoch", ylabel="Accuracy", title="Test Accuracy")
    for ax in (ax_train, ax_test):
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_log(save_dir, config, results: dict):
    """学習ログを JSON として保存する。"""
    log = {
        "config": config,
        "results": results,
    }
    with open(os.path.join(save_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
