import torch
import torch.nn.functional as F

class MemoryBankQueue(torch.nn.Module):
    """
    Fixed-size FIFO queue of L2-normalized features and (optional) labels.
    Shape: features [K, D], labels [K] (optional)
    """
    def __init__(self, K: int, feat_dim: int, store_labels: bool = True, device="cuda"):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.store_labels = store_labels

        # MoCo-style: keep buffers so they move with .to(device)/DDP and get saved in state_dict
        self.register_buffer("features", F.normalize(torch.randn(K, feat_dim), dim=1))
        if store_labels:
            self.register_buffer("labels", torch.full((K,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))  # insertion pointer

        self.to(device)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor | None = None):
        """feats: [B, D] (will be L2-normalized here), labels: [B] or None"""
        feats = F.normalize(feats, dim=1)
        bsz = feats.shape[0]
        K   = self.K
        ptr = int(self.ptr.item())

        # If we would wrap, split into two writes
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
        # If you care about "filled yet?" semantics in the first epoch:
        # treat all entries as valid once ptr has wrapped at least once.
        return min(self.K, int(self.ptr.item()) if self.ptr.item() > 0 else self.K)

@torch.no_grad()
def knn_top1_on_batch(q: torch.Tensor, y: torch.Tensor, bank: MemoryBankQueue, k=1, T=0.07):
    # q: [B, D] (raw features)   y: [B]
    q = F.normalize(q, dim=1)
    F_bank, y_bank = bank.tensors()  # [K, D], [K]
    # cosine sim = dot product because both are L2-normalized
    sims = q @ F_bank.t()            # [B, K]
    # top-k neighbors
    dist, idx = sims.topk(k, dim=1, largest=True, sorted=True)
    if y_bank is None:
        # if you don't store labels, you can't compute accuracy
        return float('nan')
    neigh_labels = y_bank[idx]       # [B, k]
    # temperature-weighted voting (same idea as DINO/MMSelfSup)
    weights = (dist / T).exp()       # [B, k]
    num_classes = int(y.max().item()) + 1  # or pass explicitly
    votes = torch.zeros(q.size(0), num_classes, device=q.device)
    votes.scatter_add_(1, neigh_labels, weights)  # sum weights per class
    pred = votes.argmax(dim=1)
    return (pred == y).float().mean().item()
