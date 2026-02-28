import argparse
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="dtype\\(\\).*align")

from model import make_model
from save import save_graphs, save_log
from train import load_data, precompute_spd_features, make_optimizer, train_one


# (feature, dataset) → 学習率
LR_DEFAULTS = {
    ("pixel", "mnist"):   0.1,
    ("pixel", "fashion"): 0.1,
    ("pixel", "cifar10"): 0.05,
    ("pixel", "stl10"):   0.05,
    ("spd",   "mnist"):   0.01,
    ("spd",   "fashion"): 0.01,
    ("spd",   "cifar10"): 0.005,
    ("spd",   "stl10"):   0.005,
}

# 全有効組み合わせ (feature, model, optimizer)
ALL_COMBINATIONS = [
    ("pixel", "mlp", "sgd"),
    ("pixel", "mlp", "stiefel"),
    ("pixel", "cnn", "sgd"),
    ("pixel", "cnn", "stiefel"),
    ("spd",   "mlp", "sgd"),
    ("spd",   "mlp", "stiefel"),
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_label(feature, model_type, optimizer):
    return f"{feature.upper()}+{model_type.upper()}+{'Stiefel' if optimizer == 'stiefel' else 'SGD'}"


def validate_combination(feature, model_type):
    if feature == "spd" and model_type == "cnn":
        print("エラー: SPD+CNN は不可（SPD出力はフラットベクトル、CNNは空間入力が必要）", file=sys.stderr)
        sys.exit(1)


def generate_combinations():
    return list(ALL_COMBINATIONS)


def parse_args():
    parser = argparse.ArgumentParser(description="多様体最適化の実験")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "cifar10", "stl10"])
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--feature", default="pixel", choices=["pixel", "spd"])
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "stiefel"])
    parser.add_argument("--lr", type=float, default=None, help="学習率（省略時は自動選択）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="全有効組み合わせ（6パターン）を一括実行")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 実験リストを構築
    if args.all:
        combinations = generate_combinations()
    else:
        validate_combination(args.feature, args.model)
        combinations = [(args.feature, args.model, args.optimizer)]

    # データ読み込み（1回）
    set_seed(args.seed)
    train_loader, test_loader = load_data(args.dataset, args.batch_size, device)

    # SPD 特徴量が必要なら1回だけ事前計算
    needs_spd = any(feat == "spd" for feat, _, _ in combinations)
    spd_train_loader, spd_test_loader, spd_dim = None, None, None
    if needs_spd:
        spd_train_loader, spd_test_loader, spd_dim = precompute_spd_features(
            train_loader, test_loader, args.batch_size
        )

    # 各実験を実行
    results = {}
    for feature, model_type, opt_name in combinations:
        use_stiefel = opt_name == "stiefel"
        label = make_label(feature, model_type, opt_name)
        lr = args.lr if args.lr is not None else LR_DEFAULTS[(feature, args.dataset)]

        set_seed(args.seed)
        print(f"\n=== {label} (lr={lr}) ===")

        model = make_model(args.dataset, model_type, feature, use_stiefel, spd_dim)
        optimizer = make_optimizer(model, lr, use_stiefel)

        if feature == "spd":
            tl, el = spd_train_loader, spd_test_loader
        else:
            tl, el = train_loader, test_loader

        hist = train_one(model, optimizer, tl, el, device, args.epochs)
        results[label] = hist

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, args.epochs + 1))
    labels = list(results.keys())
    config = {
        "dataset": args.dataset,
        "experiments": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
    }

    save_graphs(save_dir, epochs, results)
    save_log(save_dir, config, results)

    print(f"\n保存完了: {save_dir}/")
