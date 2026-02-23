import warnings

import torch


@torch.no_grad()
def retract_qr(W: torch.Tensor) -> torch.Tensor:
    """
    SVD ベースの polar retraction（Stiefel へ戻す）
    W^T W = I を満たす最近傍の直交行列を返す。
    MPS では SVD が未対応のため PyTorch が内部的に CPU フォールバックする。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*linalg_svd.*")
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    return U @ Vh


@torch.no_grad()
def project_to_tangent(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    ユークリッド勾配 G を Stiefel の接空間へ射影（canonical でよく使われる形）
      rgrad = G - W * sym(W^T G)
    """
    WT_G = W.T @ G
    sym = 0.5 * (WT_G + WT_G.T)
    rgrad = G - W @ sym
    return rgrad
