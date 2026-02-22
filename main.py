import torch

from train import train

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epoch = 30

    print("=== Stiefel 制約あり ===")
    train(device=device, epochs=num_epoch, batch_size=256, lr=0.1, use_stiefel=True)

    print()
    print("=== 制約なし (SGD) ===")
    train(device=device, epochs=num_epoch, batch_size=256, lr=0.1, use_stiefel=False)
