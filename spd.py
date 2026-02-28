import warnings

import torch


def image_to_spd_covariance(images: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    画像バッチを SPD 共分散行列に変換する。

    Parameters
    ----------
    images : (B, C, H, W)
    eps : 正定値保証のための正則化パラメータ

    Returns
    -------
    cov : (B, W, W) の SPD 行列
    """
    B, C, H, W = images.shape
    # (B, C*H, W) — 行方向にサンプル、列方向に特徴量
    X = images.reshape(B, C * H, W)
    # 中心化
    X_c = X - X.mean(dim=1, keepdim=True)
    n = C * H
    # 共分散行列 + 正則化
    cov = torch.bmm(X_c.transpose(1, 2), X_c) / (n - 1) + eps * torch.eye(W, device=images.device).unsqueeze(0)
    return cov


def spd_log_euclidean_features(images: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    画像バッチ → SPD 共分散行列 → 行列対数 → 上三角ベクトル化。

    Log-Euclidean 写像により SPD 多様体上の点をユークリッド空間へ写す。
    MPS では eigh が未対応のため CPU 上で計算する。

    Parameters
    ----------
    images : (B, C, H, W)
    eps : 正定値保証のための正則化パラメータ

    Returns
    -------
    features : (B, D) where D = W*(W+1)/2
    """
    orig_device = images.device
    # CPU 上で計算（MPS は eigh 非対応）
    images_cpu = images.cpu()
    cov = image_to_spd_covariance(images_cpu, eps)

    # 固有値分解
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*linalg_eigh.*")
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # 行列対数: S = V diag(log λ) V^T
    log_eigenvalues = torch.log(eigenvalues.clamp(min=1e-10))
    log_cov = eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)

    # 上三角ベクトル化（非対角要素は √2 倍して Frobenius ノルム等距離を保つ）
    B, W, _ = log_cov.shape
    idx = torch.triu_indices(W, W)
    features = log_cov[:, idx[0], idx[1]]
    # 非対角要素のマスク
    off_diag = idx[0] != idx[1]
    features[:, off_diag] *= 2.0 ** 0.5

    return features.to(orig_device)
