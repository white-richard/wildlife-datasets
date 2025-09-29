import torch


class MemoryBankQueue(torch.nn.Module):
    """
    Fixed-size FIFO queue of Lorentz (hyperboloid) features and (optional) labels.
    Shape: features [K, D], labels [K] (optional).
    Assumes time-like coord is at index 0.

    Set project_to_hyperboloid=True if you want to (re)normalize incoming feats.
    """
    def __init__(self, manifold, K: int, feat_dim: int, store_labels: bool = True,
                 device="cuda", project_to_hyperboloid: bool = False):
        super().__init__()
        self.manifold = manifold
        self.K = K
        self.feat_dim = feat_dim
        self.store_labels = store_labels
        self.project_to_hyperboloid = project_to_hyperboloid

        # Buffers: we don't L2-normalize or attempt cosine. Init doesn't matter for invalid slots.
        self.register_buffer("features", torch.zeros(K, feat_dim))
        if store_labels:
            self.register_buffer("labels", torch.full((K,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))  # insertion pointer

        self.to(device)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor | None = None):
        """
        feats: [B, D] assumed to be Lorentz features (time coord first).
        If project_to_hyperboloid is True, recompute time coord so -x0^2 + ||x_{1:}||^2 = -1.
        """
        if self.project_to_hyperboloid:
            feats = self.manifold.projx(feats)
        bsz = feats.shape[0]
        K   = self.K
        ptr = int(self.ptr.item())

        end = ptr + bsz
        if end <= K:
            self.features[ptr:end, :] = feats
            if self.store_labels and labels is not None:
                self.labels[ptr:end] = labels
        else:
            first = K - ptr
            self.features[ptr:K, :] = feats[:first]
            self.features[0:end - K, :] = feats[first:]
            if self.store_labels and labels is not None:
                self.labels[ptr:K] = labels[:first]
                self.labels[0:end - K] = labels[first:]

        self.ptr[0] = (ptr + bsz) % K

    @torch.no_grad()
    def tensors(self):
        """Return (features [K, D], labels [K] or None) with current contents."""
        if self.store_labels:
            return self.features, self.labels
        return self.features, None

    @torch.no_grad()
    def valid_length(self):
        return min(self.K, int(self.ptr.item()) if self.ptr.item() > 0 else self.K)

# ---------- KNN using negative geodesic distance (Lorentz) ----------

@torch.no_grad()
def knn_top1_on_batch_lorentz(
    q: torch.Tensor, y: torch.Tensor, bank: MemoryBankQueue,
    k: int = 10, T: float = 0.07, num_classes: int | None = None,
    reproject_queries: bool = False
):
    """
    KNN monitor on Lorentz hyperboloid embeddings.
    Similarity = - geodesic distance  (i.e., larger is better).

    q: [B, D] (includes time coord at index 0)
    y: [B]
    bank: MemoryBankQueue with hyperbolic features
    """
    manifold = bank.manifold
    # Optionally reproject queries to the hyperboloid for numerical safety
    if reproject_queries:
        q = manifold.projx(q)

    F_bank, y_bank = bank.tensors()
    if y_bank is None:
        return float('nan')
    valid = (y_bank != -1)
    if not torch.any(valid):
        return float('nan')

    Fv = F_bank[valid]    # [Kv, D]
    yv = y_bank[valid]    # [Kv]
    q = q.to(Fv.dtype)

    # sims = - geodesic distance
    dists = manifold.pairwise_distance(q, Fv)      # [B, Kv], float32
    sims  = -dists                            # larger is better

    k = min(k, sims.size(1))
    # top-k by largest similarity (i.e., smallest distance)
    dist, idx = sims.topk(k, dim=1, largest=True, sorted=False)
    neigh_labels = yv[idx]                    # [B, k]

    if num_classes is None:
        num_classes = int(max(y.max(), yv.max()).item()) + 1

    if k == 1:
        pred = neigh_labels.squeeze(1)
        return (pred == y).float().mean().item()

    # Temperature-weighted voting with exp(-d / T)
    weights = (dist / T).exp()                # [B, k]
    votes = torch.zeros(q.size(0), num_classes, device=q.device, dtype=weights.dtype)
    votes.scatter_add_(1, neigh_labels, weights)
    pred = votes.argmax(dim=1)
    return (pred == y).float().mean().item()
