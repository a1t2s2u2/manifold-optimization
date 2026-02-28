import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="dtype\\(\\).*align")

from model import MLP, CNN, make_model
from save import save_graphs, save_log
from train import load_data, precompute_spd_features, make_optimizer, train_one

# ── グローバル設定 ──────────────────────────────────
SEED = 42
EPOCHS = 20
BATCH_SIZE = 256
DATASET = "mnist"

# ── 実験リスト（ここを編集して実験を切り替える）────────
EXPERIMENTS = [
    dict(model=MLP, feature="pixel", stiefel=False, lr=0.1),
    dict(model=MLP, feature="pixel", stiefel=True,  lr=0.1),
    dict(model=MLP, feature="spd", stiefel=True,  lr=0.1)
]
# ────────────────────────────────────────────────────


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_label(exp):
    """実験 dict からラベル文字列を生成する。"""
    feat = exp["feature"].upper()
    model_name = exp["model"].__name__.upper()
    opt = "Stiefel" if exp["stiefel"] else "SGD"
    return f"{feat}+{model_name}+{opt}"


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # データ読み込み（1回）
    set_seed(SEED)
    train_loader, test_loader = load_data(DATASET, BATCH_SIZE, device)

    # SPD 事前計算（必要なら1回）
    needs_spd = any(e["feature"] == "spd" for e in EXPERIMENTS)
    spd_train, spd_test, spd_dim = None, None, None
    if needs_spd:
        spd_train, spd_test, spd_dim = precompute_spd_features(
            train_loader, test_loader, BATCH_SIZE
        )

    # 各実験を実行
    results = {}
    for exp in EXPERIMENTS:
        label = make_label(exp)

        set_seed(SEED)
        print(f"\n=== {label} (lr={exp['lr']}) ===")

        model = make_model(exp["model"], DATASET, exp["feature"], exp["stiefel"], spd_dim)
        optimizer = make_optimizer(model, exp["lr"], exp["stiefel"])

        if exp["feature"] == "spd":
            tl, el = spd_train, spd_test
        else:
            tl, el = train_loader, test_loader

        hist = train_one(model, optimizer, tl, el, device, EPOCHS)
        results[label] = hist

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, EPOCHS + 1))
    config = {
        "dataset": DATASET,
        "experiments": list(results.keys()),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": device,
    }

    save_graphs(save_dir, epochs, results)
    save_log(save_dir, config, results)

    print(f"\n保存完了: {save_dir}/")
