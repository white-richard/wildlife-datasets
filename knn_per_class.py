from collections import defaultdict
from typing import Dict, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def _embed_loader(
    model,
    loader: DataLoader,
    device: str = "cuda",
    l2_normalize: bool = True,
    idx2id_map: Dict[int, Union[str, int]] = None,
) -> Tuple[torch.Tensor, List[Union[str, int]]]:
    """
    Runs the model over a DataLoader and returns:
      - embs: (N, D) torch tensor on 'device'
      - labels: list of class labels (strings or ints), length N
    If idx2id_map is provided, numeric targets are mapped to human-readable IDs.
    """
    model.eval().to(device)
    embs, labels = [], []

    for xb, yb, *rest in tqdm(loader, total=len(loader), desc="Embedding"):
        xb = xb.to(device, non_blocking=True)
        zb = model(xb)
        if l2_normalize:
            zb = F.normalize(zb, dim=1)
        embs.append(zb)
        if idx2id_map is not None:
            labels.extend([idx2id_map[int(y)] for y in yb])
        else:
            labels.extend([int(y) for y in yb])

    embs = torch.cat(embs, dim=0)  # (N, D) on device
    return embs, labels


def _maybe_idx2id(ds) -> Dict[int, Union[str, int]]:
    """
    Try to get an int->identity mapping from a dataset.
    Looks for 'id2idx' (common) or 'idx2id'. Returns None if not found.
    """
    if hasattr(ds, "id2idx") and isinstance(ds.id2idx, dict):
        return {v: k for k, v in ds.id2idx.items()}
    if hasattr(ds, "idx2id") and isinstance(ds.idx2id, dict):
        return ds.idx2id
    return None


def _select_nn_indices(
    Q: torch.Tensor,
    R: torch.Tensor,
    metric: str = "cosine",
    treat_as: str = "auto",  # 'auto' | 'similarity' | 'distance'
) -> torch.Tensor:
    """
    Given query (B, D) and reference (M, D) embeddings, return the 1-NN
    indices into R according to the chosen metric.
    Metrics:
      - 'cosine'    : similarity (uses normalized vectors internally)
      - 'dot'       : similarity
      - 'euclidean'/'l2': distance
      - 'manhattan'/'l1': distance
    treat_as controls whether we maximize (similarity) or minimize (distance):
      - 'auto'       : cosine/dot -> maximize; l1/l2 -> minimize
      - 'similarity' : maximize the score
      - 'distance'   : minimize the score
    """
    metric = metric.lower()
    if metric in ("cosine", "dot"):
        # Similarity scores
        if metric == "cosine":
            Qn = F.normalize(Q, dim=1)
            Rn = F.normalize(R, dim=1)
            S = Qn @ Rn.T  # (B, M)
        else:  # 'dot'
            S = Q @ R.T  # (B, M)
        if treat_as == "distance":
            # If caller insists, treat similarities as distances by negating.
            idx = (-S).argmin(dim=1)
        else:
            idx = S.argmax(dim=1)
        return idx

    # Distance metrics
    if metric in ("euclidean", "l2"):
        D = torch.cdist(Q, R, p=2)  # (B, M)
    elif metric in ("manhattan", "l1"):
        D = torch.cdist(Q, R, p=1)  # (B, M)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use cosine, dot, euclidean/l2, manhattan/l1.")

    if treat_as == "similarity":
        # If caller insists, turn distances into similarities by negating.
        idx = (-D).argmax(dim=1)
    else:
        idx = D.argmin(dim=1)
    return idx


@torch.inference_mode()
def evaluate_knn1(
    model,
    ref_loader: DataLoader,
    qry_loader: DataLoader,
    device: str = "cuda",
    l2_normalize: bool = True,
    metric: str = "cosine",   # 'cosine' | 'dot' | 'euclidean'/'l2' | 'manhattan'/'l1'
    treat_as: str = "auto",   # 'auto' | 'similarity' | 'distance'
) -> Tuple[float, Dict[Union[str, int], float]]:
    """
    Compute 1-NN classification using reference set as the "database".
    Returns:
      overall_acc (float), per_class_acc (dict: class_label -> accuracy)

    - No FeatureDatabase / KNNMatcher.
    - Per-class accuracy is aggregated by TRUE class (from the query labels).
    - If metric='cosine', you can pass l2_normalize=True for fastest path
      (dot product == cosine). If l2_normalize=False, the function still
      computes cosine correctly by normalizing inside the scorer.
    - treat_as controls similarity vs distance behavior (see _select_nn_indices).
    """
    # Build idx->id mappings (best-effort) so we can report human-readable classes
    ref_idx2id = _maybe_idx2id(ref_loader.dataset)
    qry_idx2id = _maybe_idx2id(qry_loader.dataset) or ref_idx2id

    # 1) Embed the reference set
    R, ref_labels = _embed_loader(
        model, ref_loader, device=device, l2_normalize=l2_normalize, idx2id_map=ref_idx2id
    )  # R: (M, D)

    # 2) Evaluate queries in batches
    model.eval().to(device)
    n_correct = 0
    n_total = 0
    per_class = defaultdict(list)

    for xb, yb, *rest in tqdm(qry_loader, total=len(qry_loader), desc="Evaluating 1-NN"):
        xb = xb.to(device, non_blocking=True)
        Zq = model(xb)
        if l2_normalize and metric != "cosine":
            # For cosine we re-normalize inside scorer when needed.
            Zq = F.normalize(Zq, dim=1)

        # True labels (human-readable when possible)
        if qry_idx2id is not None:
            true_batch = [qry_idx2id[int(y)] for y in yb]
        else:
            true_batch = [int(y) for y in yb]

        # 1-NN selection
        nn_idx = _select_nn_indices(Zq, R, metric=metric, treat_as=treat_as)  # (B,)

        # Map predicted indices to reference labels
        pred_batch = [ref_labels[int(i)] for i in nn_idx.tolist()]

        # Update stats
        for t, p in zip(true_batch, pred_batch):
            correct = (p == t)
            per_class[t].append(1.0 if correct else 0.0)
            n_correct += int(correct)
            n_total += 1

    overall_acc = float(n_correct / max(1, n_total))
    per_class_acc = {cls: float(np.mean(v)) for cls, v in per_class.items()}
    return overall_acc, per_class_acc
