import torch
from torch.optim import Optimizer

from stiefel import project_to_tangent, retract_qr


class StiefelSGD(Optimizer):
    """
    SGD with optional Stiefel manifold constraint.

    Each param group can have a ``stiefel`` flag (default ``False``).
    - stiefel=False: standard SGD update  p -= lr * grad
    - stiefel=True:  Riemannian gradient descent on the Stiefel manifold
        W = p.T  (column-orthogonal convention)
        rgrad = project_to_tangent(W, G)
        W_new  = retract_qr(W - lr * rgrad)
        p <- W_new.T
    """

    def __init__(self, params, lr: float = 0.1):
        defaults = dict(lr=lr, stiefel=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            stiefel = group.get("stiefel", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                if stiefel:
                    W = p.T
                    G = p.grad.T
                    rgrad = project_to_tangent(W, G)
                    W_new = retract_qr(W - lr * rgrad)
                    p.copy_(W_new.T)
                else:
                    p.add_(p.grad, alpha=-lr)

        return loss
