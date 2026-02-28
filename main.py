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

# ── 実験リスト（ここを編集して実験を切り替える）────────
EXPERIMENTS = [
    dict(dataset="mnist", model=MLP, feature="pixel", stiefel=False, lr=0.1),
    dict(dataset="mnist", model=MLP, feature="pixel", stiefel=True,  lr=0.1),
    dict(dataset="mnist", model=CNN, feature="pixel", stiefel=False, lr=0.1),
    dict(dataset="mnist", model=CNN, feature="pixel", stiefel=True,  lr=0.1),
    dict(dataset="mnist", model=MLP, feature="spd",   stiefel=False, lr=0.01),
    dict(dataset="mnist", model=MLP, feature="spd",   stiefel=True,  lr=0.01),
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

    # データをデータセットごとにキャッシュ
    data_cache = {}   # dataset -> (train_loader, test_loader)
    spd_cache = {}    # dataset -> (spd_train, spd_test, spd_dim)

    results = {}
    for exp in EXPERIMENTS:
        ds = exp["dataset"]

        # データ読み込み（データセットごとに1回）
        if ds not in data_cache:
            set_seed(SEED)
            data_cache[ds] = load_data(ds, BATCH_SIZE, device)

        # SPD 事前計算（必要なデータセットごとに1回）
        if exp["feature"] == "spd" and ds not in spd_cache:
            spd_cache[ds] = precompute_spd_features(*data_cache[ds], BATCH_SIZE)

        spd_dim = spd_cache[ds][2] if ds in spd_cache else None
        label = make_label(exp)

        set_seed(SEED)
        print(f"\n=== {label} (lr={exp['lr']}) ===")

        model = make_model(exp["model"], ds, exp["feature"], exp["stiefel"], spd_dim)
        optimizer = make_optimizer(model, exp["lr"], exp["stiefel"])

        if exp["feature"] == "spd":
            tl, el = spd_cache[ds][0], spd_cache[ds][1]
        else:
            tl, el = data_cache[ds]

        hist = train_one(model, optimizer, tl, el, device, EPOCHS)
        results[label] = hist

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, EPOCHS + 1))
    config = {
        "dataset": list({e["dataset"] for e in EXPERIMENTS}),
        "experiments": list(results.keys()),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "device": device,
    }

    save_graphs(save_dir, epochs, results)
    save_log(save_dir, config, results)

    print(f"\n保存完了: {save_dir}/")
