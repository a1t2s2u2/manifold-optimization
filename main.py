import os
from datetime import datetime

import torch

from save import save_graphs, save_log
from train import load_data, train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "cifar10"
    num_epoch = 50
    batch_size = 256
    lr = 0.05

    # モデル名の決定（データセットに応じて自動選択）
    model_name = "cnn" if dataset in ("stl10", "cifar10") else "linear"

    # データ読み込み（ダウンロードはここで完了する）
    train_loader, test_loader = load_data(dataset, batch_size, device)

    print("=== Stiefel 制約あり ===")
    _, stiefel_hist = train(device=device, epochs=num_epoch, dataset=dataset,
                            batch_size=batch_size, lr=lr, use_stiefel=True,
                            train_loader=train_loader, test_loader=test_loader)
    print("=== 制約なし (SGD) ===")
    _, sgd_hist = train(device=device, epochs=num_epoch, dataset=dataset,
                        batch_size=batch_size, lr=lr, use_stiefel=False,
                        train_loader=train_loader, test_loader=test_loader)

    # 保存先フォルダ（日付時間）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, num_epoch + 1))
    config = {
        "dataset": dataset,
        "model": model_name,
        "epochs": num_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
    }

    save_graphs(save_dir, epochs, stiefel_hist, sgd_hist)
    save_log(save_dir, config, stiefel_hist, sgd_hist)

    print(f"\n保存完了: {save_dir}/")
