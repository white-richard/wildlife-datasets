import torch
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def pairwise_cosine(z: torch.Tensor) -> torch.Tensor:
    z = _l2_normalize(z)
    return z @ z.T

def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Minkowski metric diag([-1, +1, +1, ...])
    # x, y: [..., D], with time-like coord at index 0
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def pairwise_lorentz_geodesic(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # y: [B, D] on the hyperboloid; curvature assumed 1 (we lock it elsewhere)
    # d(x,y) = arcosh( -<x,y>_L ), with clamp for numerical stability
    # Build BxB matrix efficiently
    # Compute -<x_i, y_j>_L for all i,j
    t = -lorentz_inner(y.unsqueeze(1), y.unsqueeze(0))  # [B, B]
    t = t.clamp_min(1.0 + eps)
    # acosh(t) = log(t + sqrt(t^2 - 1))
    return torch.log(t + torch.sqrt(t * t - 1.0))

def kl_rows(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    # P,Q: [B, B], row-stochastic, with zeros on diag
    # Return mean KL across rows
    P = P.clamp_min(1e-12)
    Q = Q.clamp_min(1e-12)
    return (P * (P.log() - Q.log())).sum(dim=1).mean()

def softmax_rows(logits: torch.Tensor, T: float) -> torch.Tensor:
    # mask self-entries with -inf then softmax along dim=1
    logits = logits.clone()
    logits.fill_diagonal_(-float('inf'))
    return F.softmax(logits / max(T, 1e-6), dim=1)
