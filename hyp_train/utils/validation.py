import torch
import numpy as np
from typing import Dict, Tuple, Iterable, Literal

SimMetric = Literal["cosine", "neg_lorentz_geo"]

@torch.no_grad()
def _extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    use_amp: bool = False,
    sim_metric: SimMetric = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        feats:  (N, D) float32 tensor on CPU
        labels: (N,)   int64 tensor on CPU

    Behavior:
      - sim_metric == 'cosine'        -> L2-normalize feats (Euclidean)
      - sim_metric == 'neg_lorentz_geo' -> DO NOT normalize; feats are on Lorentz manifold
    Expects dataset __getitem__ to yield (image, label, ...)
    """
    model.eval()
    feats_list, labels_list = [], []

    dtype = torch.float16 if (use_amp and torch.cuda.is_available()) else torch.float32
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                            dtype=torch.float16, enabled=use_amp):
            f = model(x)

        if sim_metric == "cosine":
            # L2-normalize features (typical for cosine sim)
            f = torch.nn.functional.normalize(f, dim=1)

        feats_list.append(f.detach().to("cpu", dtype=torch.float32))
        labels_list.append(y.detach().to("cpu", dtype=torch.int64))

    feats  = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return feats, labels


def _retrieval_metrics(
    S: np.ndarray,               # (N_qry, N_ref) similarity (higher is better)
    qry_labels: np.ndarray,      # (N_qry,)
    ref_labels: np.ndarray,      # (N_ref,)
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Closed-set ReID retrieval:
      - mAP: mean of average precision over queries
      - CMC@k / top-k: hit if any correct identity appears within top-k
    """
    Nq, Nr = S.shape
    ranks = np.argsort(-S, axis=1)  # descending similarity

    mAP_sum = 0.0
    topk_hits = {k: 0 for k in ks}

    for i in range(Nq):
        y = qry_labels[i]
        order = ranks[i]
        rel = (ref_labels[order] == y).astype(np.int32)

        n_pos = rel.sum()
        if n_pos == 0:
            continue

        cumsum = np.cumsum(rel)
        pos_idx = np.flatnonzero(rel)
        prec_at_pos = cumsum[pos_idx] / (pos_idx + 1)
        AP = float(prec_at_pos.mean())
        mAP_sum += AP

        for k in ks:
            topk_hits[k] += int(rel[:k].any())

    valid_q = max(1, Nq)
    metrics = {"mAP": mAP_sum / valid_q}
    for k in ks:
        metrics[f"top{k}"] = topk_hits[k] / valid_q
    return metrics


@torch.no_grad()
def _pairwise_cosine(qry_feats: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
    """
    Assumes qry_feats/ref_feats are L2-normalized.
    Returns (N_qry, N_ref) cosine similarities on CPU (float32).
    """
    return (qry_feats @ ref_feats.T).to("cpu", dtype=torch.float32)


@torch.no_grad()
def _pairwise_neg_lorentz_geodesic(
    qry_feats: torch.Tensor,
    ref_feats: torch.Tensor,
    eps: float = 1e-6,
    use_float64: bool = True,
) -> torch.Tensor:
    """
    Compute S_ij = - d_H(x_i, y_j) where d_H is geodesic distance
    on the Lorentz model (signature diag([-1, +1, ..., +1])).

    For x=(x0, x1..xd) and y=(y0, y1..yd):
      <x,y>_L = -x0*y0 + sum_{k=1}^d xk*yk
      d_H(x,y) = acosh( - <x,y>_L ), where -<x,y>_L >= 1.

    Returns (N_qry, N_ref) similarities on CPU (float32).
    """
    X = qry_feats
    Y = ref_feats
    if use_float64:
        X = X.to(dtype=torch.float64)
        Y = Y.to(dtype=torch.float64)

    # Split time-like vs spatial parts (time-like is first coord)
    X0 = X[:, :1]             # (Nq, 1)
    Y0 = Y[:, :1]             # (Nr, 1)
    Xs = X[:, 1:]             # (Nq, d)
    Ys = Y[:, 1:]             # (Nr, d)

    # Minkowski Gram matrix: <x_i, y_j>_L = (Xs @ Ys^T) - (X0 @ Y0^T)
    spatial = Xs @ Ys.T                   # (Nq, Nr)
    time_outer = X0 @ Y0.T                # (Nq, Nr)
    minkowski = spatial - time_outer      # (Nq, Nr)

    # acosh argument: z = -<x,y>_L, should be >= 1
    z = (-minkowski).clamp(min=1.0 + eps)

    # Geodesic distance and negative similarity
    d = torch.acosh(z)                    # (Nq, Nr)
    S = (-d).to("cpu", dtype=torch.float32)
    return S


@torch.no_grad()
def validate_split(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    split_prefix: str = "val",   # "val" or "test"
    device: str = "cuda",
    use_amp: bool = False,
    ks: Iterable[int] = (1, 5, 10),
    sim_metric: SimMetric = "cosine",
) -> Dict[str, float]:
    """
    Uses loaders[f'{split_prefix}_ref'] and loaders[f'{split_prefix}_qry'].

    sim_metric:
      - "cosine":          cosine similarity on L2-normalized Euclidean features
      - "neg_lorentz_geo": negative hyperbolic geodesic distance on Lorentz-model features
                           (expects first channel = time-like coord, x0 > 0)
    Produces retrieval metrics (mAP, top1/top5/top10).
    """
    ref_loader = loaders[f"{split_prefix}_ref"]
    qry_loader = loaders[f"{split_prefix}_qry"]

    ref_feats, ref_labels = _extract_embeddings(
        model, ref_loader, device=device, use_amp=use_amp, sim_metric=sim_metric
    )
    qry_feats, qry_labels = _extract_embeddings(
        model, qry_loader, device=device, use_amp=use_amp, sim_metric=sim_metric
    )

    if sim_metric == "cosine":
        S = _pairwise_cosine(qry_feats, ref_feats).numpy()     # (N_qry, N_ref)
    elif sim_metric == "neg_lorentz_geo":
        S = _pairwise_neg_lorentz_geodesic(qry_feats, ref_feats).numpy()
    else:
        raise ValueError(f"Unknown sim_metric: {sim_metric}")

    metrics = _retrieval_metrics(
        S=S,
        qry_labels=qry_feats.new_tensor(qry_labels, dtype=torch.int64).cpu().numpy() if False else qry_labels.cpu().numpy(),
        ref_labels=ref_feats.new_tensor(ref_labels, dtype=torch.int64).cpu().numpy() if False else ref_labels.cpu().numpy(),
        ks=ks,
    )

    pretty = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    mode_txt = "cosine" if sim_metric == "cosine" else "neg-hyperbolic-geodesic"
    # print(f"[{split_prefix.upper()} • {mode_txt}] {pretty}")
    return {f"{k}": v for k, v in metrics.items()}


# import torch
# import numpy as np
# from typing import Dict, Tuple, Iterable

# @torch.no_grad()
# def _extract_embeddings(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     device: str = "cuda",
#     use_amp: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#         feats:  (N, D) L2-normalized float32 tensor on CPU
#         labels: (N,)   int64 tensor on CPU
#     Expects dataset __getitem__ to yield (image, label, ...)
#     """
#     model.eval()
#     feats_list, labels_list = [], []

#     dtype = torch.float16 if (use_amp and torch.cuda.is_available()) else torch.float32
#     for batch in loader:
#         x = batch[0].to(device, non_blocking=True)
#         y = batch[1].to(device, non_blocking=True)

#         with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
#                             dtype=torch.float16, enabled=use_amp):
#             f = model(x)

#         # L2-normalize features (common for cosine similarity)
#         f = torch.nn.functional.normalize(f, dim=1)
#         feats_list.append(f.detach().to("cpu", dtype=torch.float32))
#         labels_list.append(y.detach().to("cpu", dtype=torch.int64))

#     feats  = torch.cat(feats_list, dim=0)
#     labels = torch.cat(labels_list, dim=0)
#     return feats, labels


# def _retrieval_metrics(
#     S: np.ndarray,               # (N_qry, N_ref) similarity (higher is better)
#     qry_labels: np.ndarray,      # (N_qry,)
#     ref_labels: np.ndarray,      # (N_ref,)
#     ks: Iterable[int] = (1, 5, 10),
# ) -> Dict[str, float]:
#     """
#     Closed-set ReID retrieval:
#       - mAP: mean of average precision over queries
#       - CMC@k / top-k: hit if any correct identity appears within top-k
#     """
#     Nq, Nr = S.shape
#     # Sort refs for each query (descending similarity)
#     ranks = np.argsort(-S, axis=1)  # (N_qry, N_ref)

#     # Build relevance matrix on the fly
#     mAP_sum = 0.0
#     topk_hits = {k: 0 for k in ks}

#     for i in range(Nq):
#         y = qry_labels[i]
#         order = ranks[i]              # indices into ref set
#         rel = (ref_labels[order] == y).astype(np.int32)  # (N_ref,)

#         # Skip queries with no positives (shouldn't happen in closed-set, but safe)
#         n_pos = rel.sum()
#         if n_pos == 0:
#             continue

#         # Precision at each rank where a relevant item occurs
#         cumsum = np.cumsum(rel)
#         pos_idx = np.flatnonzero(rel)             # ranks where relevant appears
#         prec_at_pos = cumsum[pos_idx] / (pos_idx + 1)
#         AP = prec_at_pos.mean()
#         mAP_sum += AP

#         # Top-k hits
#         for k in ks:
#             topk_hits[k] += int(rel[:k].any())

#     valid_q = max(1, Nq)  # avoid div-by-zero
#     metrics = {
#         "mAP": mAP_sum / valid_q,
#     }
#     for k in ks:
#         metrics[f"top{k}"] = topk_hits[k] / valid_q
#     return metrics


# @torch.no_grad()
# def validate_split(
#     model: torch.nn.Module,
#     loaders: Dict[str, torch.utils.data.DataLoader],
#     split_prefix: str = "val",   # "val" or "test"
#     device: str = "cuda",
#     use_amp: bool = False,
#     ks: Iterable[int] = (1, 5, 10),
# ) -> Dict[str, float]:
#     """
#     Uses loaders[f'{split_prefix}_ref'] and loaders[f'{split_prefix}_qry'].
#     Produces cosine-sim retrieval metrics (mAP, top1/top5/top10).
#     """
#     ref_loader = loaders[f"{split_prefix}_ref"]
#     qry_loader = loaders[f"{split_prefix}_qry"]

#     ref_feats, ref_labels = _extract_embeddings(model, ref_loader, device=device, use_amp=use_amp)
#     qry_feats, qry_labels = _extract_embeddings(model, qry_loader, device=device, use_amp=use_amp)

#     # Cosine similarity via dot product (features are L2-normalized)
#     S = (qry_feats @ ref_feats.T).cpu().numpy()              # (N_qry, N_ref)
#     metrics = _retrieval_metrics(
#         S=S,
#         qry_labels=qry_labels.cpu().numpy(),
#         ref_labels=ref_labels.cpu().numpy(),
#         ks=ks,
#     )
#     # Optional: pretty print
#     pretty = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
#     print(f"[{split_prefix.upper()}] {pretty}")
#     return {f"{split_prefix}/{k}": v for k, v in metrics.items()}
