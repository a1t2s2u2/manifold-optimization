import torch


@torch.no_grad()
def retract_qr(W: torch.Tensor) -> torch.Tensor:
    """
    QR 分解によるリトラクション（Stiefel へ戻す）
    W^T W = I を満たすように、W を Q に置き換える
    """
    # torch.linalg.qr は W = Q R を返す
    # MPS では未実装のため CPU にフォールバック
    orig_device = W.device
    Q, R = torch.linalg.qr(W.cpu(), mode="reduced")
    Q, R = Q.to(orig_device), R.to(orig_device)

    # 符号の揺れを抑える（任意だが安定しやすい）
    diag = torch.diagonal(R, 0)
    sign = torch.sign(diag)
    sign[sign == 0] = 1.0
    Q = Q @ torch.diag(sign)

    return Q


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
