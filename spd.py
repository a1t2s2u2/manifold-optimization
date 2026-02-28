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


def _log_euclidean_vectorize(cov: torch.Tensor, min_eig: float = 1e-6) -> torch.Tensor:
    """
    SPD 行列 → Log-Euclidean 上三角ベクトル（対称ベクトル化、off-diag は sqrt(2) スケール）。

    cov: (B, D, D) SPD
    return: (B, D*(D+1)//2)
    """
    # 対称化（数値誤差対策）
    cov = 0.5 * (cov + cov.transpose(-2, -1))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*linalg_eigh.*")
        evals, evecs = torch.linalg.eigh(cov)

    # 固有値の下限を clamp（eps と連動させるのが自然）
    evals = evals.clamp(min=min_eig)
    log_evals = torch.log(evals)

    # logm(C) = V log(Λ) V^T
    log_cov = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-2, -1)

    D = log_cov.shape[-1]
    idx = torch.triu_indices(D, D, device=log_cov.device)
    feats = log_cov[:, idx[0], idx[1]]  # (B, ntri)

    # 対称ベクトル化：off-diagonal は sqrt(2)
    off = idx[0] != idx[1]
    feats[:, off] *= 2.0 ** 0.5
    return feats


def spd_log_euclidean_features(
    images: torch.Tensor,
    eps: float = 1e-4,
    grid: int = 4,
    include_mean_col: bool = True,
    min_eig: Optional[float] = None,
) -> torch.Tensor:
    """
    画像を grid×grid パッチに分割し、
      - row方向: 列間共分散 (pW×pW) の log-vector
      - col方向: 行間共分散 (pH×pH) の log-vector
      - 1次統計: row平均 (pW) + (option) col平均 (pH)
    を連結する。

    images: (B, C, H, W)
    return: (B, grid*grid * (pW + [pH] + tri(pW) + tri(pH)))
    """
    assert images.dim() == 4, "images must be (B, C, H, W)"
    B, C, H, W = images.shape
    assert H % grid == 0 and W % grid == 0, "H and W must be divisible by grid"

    pH, pW = H // grid, W // grid
    P = grid * grid

    # min_eig を eps と連動（log の安定性）
    if min_eig is None:
        min_eig = max(eps * 1e-2, 1e-8)

    # (B, C, grid, grid, pH, pW)
    patches = images.unfold(2, pH, pH).unfold(3, pW, pW)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, grid, grid, C, pH, pW)
    patches = patches.view(B * P, C, pH, pW)  # (BP, C, pH, pW)

    BP = B * P

    # --- row方向： (BP, C*pH, pW)
    X_row = patches.reshape(BP, C * pH, pW)
    mean_row = X_row.mean(dim=1)  # (BP, pW)
    X_row_c = X_row - mean_row.unsqueeze(1)

    # 共分散 (BP, pW, pW)
    denom_row = max(C * pH - 1, 1)  # 0割防止
    cov_row = (X_row_c.transpose(1, 2) @ X_row_c) / denom_row
    cov_row = cov_row + eps * torch.eye(pW, device=images.device, dtype=images.dtype).unsqueeze(0)
    log_row = _log_euclidean_vectorize(cov_row, min_eig=min_eig)

    # --- col方向： (BP, C*pW, pH)
    X_col = patches.permute(0, 1, 3, 2).reshape(BP, C * pW, pH)
    mean_col = X_col.mean(dim=1)  # (BP, pH)
    X_col_c = X_col - mean_col.unsqueeze(1)

    denom_col = max(C * pW - 1, 1)
    cov_col = (X_col_c.transpose(1, 2) @ X_col_c) / denom_col
    cov_col = cov_col + eps * torch.eye(pH, device=images.device, dtype=images.dtype).unsqueeze(0)
    log_col = _log_euclidean_vectorize(cov_col, min_eig=min_eig)

    # 特徴連結（平均は row だけだと情報が片寄ることがあるので col も option で追加）
    if include_mean_col:
        patch_feat = torch.cat([mean_row, mean_col, log_row, log_col], dim=1)  # (BP, F)
    else:
        patch_feat = torch.cat([mean_row, log_row, log_col], dim=1)

    # (B, P*F)
    patch_feat = patch_feat.view(B, P * patch_feat.shape[1])
    return patch_feat