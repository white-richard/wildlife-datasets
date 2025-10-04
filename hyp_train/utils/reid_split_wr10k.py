from dataclasses import dataclass
from typing import Tuple, Set, Dict, Any
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets import splits
from .wr10k_dataset import WR10kDataset


# Assumed available from your codebase:
# - WildlifeReID10k
# - WR10kDataset
# - splits.ClosedSetSplit
# - _train_tfms(), _eval_tfms()
# - cfg with attributes: root, train_batch, eval_batch, num_workers, seed (optional)

@dataclass
class ReIDSplits:
    # Datasets
    train_fit_set: Any
    dev_ref_set: Any
    dev_qry_set: Any
    test_ref_set: Any
    test_qry_set: Any
    # DataLoaders
    train_fit_loader: DataLoader
    dev_ref_loader: DataLoader
    dev_qry_loader: DataLoader
    test_ref_loader: DataLoader
    test_qry_loader: DataLoader
    # (optional) bookkeeping
    id_counts: Dict[str, int]

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _split_identities(
    df: pd.DataFrame,
    frac_train_ids: float = 0.8,
    seed: int = 42,
    col_label: str = "identity",
) -> Tuple[Set, Set]:
    """Split identities into train vs test (no overlap)."""
    ids = df[col_label].drop_duplicates().to_numpy()
    g = _rng(seed)
    g.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * frac_train_ids))
    n_train = min(max(n_train, 1), n - 1)  # keep at least 1 test id

    train_ids = set(ids[:n_train])
    test_ids  = set(ids[n_train:])
    return train_ids, test_ids

def _closed_split_gallery_query(
    df_block: pd.DataFrame,
    ratio_gallery: float = 0.5,
    col_label: str = "identity",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make gallery/query using your ClosedSetSplit, then enforce closed-set
    by intersecting identities on both sides.
    """
    splitter = splits.ClosedSetSplit(ratio_train=ratio_gallery, col_label=col_label)

    # idx_ref, idx_qry = splitter.split(df_block)[0]
    # df_ref = df_block.iloc[idx_ref].copy()
    # df_qry = df_block.iloc[idx_qry].copy()

    idx_ref, idx_qry = splitter.split(df_block)[0]
    df_ref = df_block.loc[idx_ref].copy()
    df_qry = df_block.loc[idx_qry].copy()

    ref_ids = set(df_ref[col_label].unique())
    qry_ids = set(df_qry[col_label].unique())
    if ref_ids != qry_ids:
        common = ref_ids & qry_ids
        df_ref = df_ref[df_ref[col_label].isin(common)].copy()
        df_qry = df_qry[df_qry[col_label].isin(common)].copy()

    return df_ref, df_qry

def _train_fit_vs_dev(
    df_train_all: pd.DataFrame,
    ratio_fit: float = 0.9,
    col_label: str = "identity",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *images* of the training identities into:
      - train_fit (for optimizing weights)
      - dev_all   (for tuning/calibration & eval)
    We reuse ClosedSetSplit so both sides keep identical identity sets.
    """
    splitter = splits.ClosedSetSplit(ratio_train=ratio_fit, col_label=col_label)

    # idx_fit, idx_dev = splitter.split(df_train_all)[0]
    # df_fit = df_train_all.iloc[idx_fit].copy()
    # df_dev = df_train_all.iloc[idx_dev].copy()

    idx_fit, idx_dev = splitter.split(df_train_all)[0]
    df_fit = df_train_all.loc[idx_fit].copy()
    df_dev = df_train_all.loc[idx_dev].copy()

    # Enforce same identities on both sides (closed-set among train identities)
    fit_ids = set(df_fit[col_label].unique())
    dev_ids = set(df_dev[col_label].unique())
    if fit_ids != dev_ids:
        common = fit_ids & dev_ids
        df_fit = df_fit[df_fit[col_label].isin(common)].copy()
        df_dev = df_dev[df_dev[col_label].isin(common)].copy()

    return df_fit, df_dev

def _make_eval_sets_and_loaders(
    df_ref: pd.DataFrame,
    df_qry: pd.DataFrame,
    root: str,
    eval_batch: int,
    transform,
    num_workers: int,
) -> Tuple[Any, Any, DataLoader, DataLoader]:
    """
    Build WR10kDataset for ref/qry with a consistent id_list,
    and return datasets + dataloaders.
    """
    # Bootstrap id_list from the reference set to pin label mapping within the split
    ref_tmp = WR10kDataset(df_ref, root, transform)
    id_list = sorted(ref_tmp.id2idx.keys())

    ref_set = WR10kDataset(df_ref, root, transform, id_list=id_list)
    qry_set = WR10kDataset(df_qry, root, transform, id_list=id_list)

    ref_loader = DataLoader(ref_set, batch_size=eval_batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    qry_loader = DataLoader(qry_set, batch_size=eval_batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return ref_set, qry_set, ref_loader, qry_loader


def build_reid_pipeline(
    cfg,
    train_transform,
    val_transform,
    *,
    col_label: str = "identity",
    frac_train_ids: float = 0.8,
    ratio_fit: float = 0.9,
    ratio_gallery_dev: float = 0.5,
    ratio_gallery_test: float = 0.5,
) -> ReIDSplits:
    seed = getattr(cfg, "seed", 42)

    meta = WildlifeReID10k(cfg.root)
    df = meta.df

    # 1) Split identities (unchanged)
    train_ids, test_ids = _split_identities(
        df, frac_train_ids=frac_train_ids, seed=seed, col_label=col_label
    )
    df_train_all = df[df[col_label].isin(train_ids)].copy()
    df_test_all  = df[df[col_label].isin(test_ids)].copy()

    # 2) Within training identities: per-ID split -> fit vs dev (guaranteed non-empty if possible)
    df_train_fit, df_dev_all = _safe_train_fit_vs_dev(
        df_train_all, ratio_fit=ratio_fit, col_label=col_label, seed=seed
    )

    # 3) Dev: gallery/query per-ID split
    df_dev_ref, df_dev_qry = _safe_closed_split_gallery_query(
        df_dev_all, ratio_gallery=ratio_gallery_dev, col_label=col_label, seed=seed + 100
    )

    # 4) Test: first drop test IDs with <2 total images; then per-ID split
    df_test_all2 = _filter_min_images_per_id(df_test_all, col_label, min_images=2)
    df_test_ref, df_test_qry = _safe_closed_split_gallery_query(
        df_test_all2, ratio_gallery=ratio_gallery_test, col_label=col_label, seed=seed + 200
    )

    # 5) Sanity checks: fail early with clear messages if fundamentally impossible
    def _assert_non_empty(name, frame):
        if len(frame) == 0:
            raise ValueError(
                f"{name} is empty. Likely no identities with ≥2 images available "
                f"after filtering. Check your dataset distribution or relax ratios."
            )

    _assert_non_empty("train_fit", df_train_fit)
    _assert_non_empty("dev_ref", df_dev_ref)
    _assert_non_empty("dev_qry", df_dev_qry)
    _assert_non_empty("test_ref", df_test_ref)
    _assert_non_empty("test_qry", df_test_qry)

    # 6) Build datasets/loaders (unchanged)
    train_fit_set = WR10kDataset(df_train_fit, cfg.root, train_transform)
    train_fit_loader = DataLoader(
        train_fit_set,
        batch_size=cfg.train_batch,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    dev_ref_set, dev_qry_set, dev_ref_loader, dev_qry_loader = _make_eval_sets_and_loaders(
        df_dev_ref, df_dev_qry, cfg.root, cfg.eval_batch, val_transform, cfg.num_workers
    )
    test_ref_set, test_qry_set, test_ref_loader, test_qry_loader = _make_eval_sets_and_loaders(
        df_test_ref, df_test_qry, cfg.root, cfg.eval_batch, val_transform, cfg.num_workers
    )

    id_counts = {
        "train_fit_ids": len(set(df_train_fit[col_label].unique())),
        "dev_ids":       len(set(df_dev_all[col_label].unique())),
        "test_ids":      len(set(df_test_all2[col_label].unique())),
        "dev_ref_n":     len(df_dev_ref),
        "dev_qry_n":     len(df_dev_qry),
        "test_ref_n":    len(df_test_ref),
        "test_qry_n":    len(df_test_qry),
    }

    return ReIDSplits(
        train_fit_set=train_fit_set,
        dev_ref_set=dev_ref_set,
        dev_qry_set=dev_qry_set,
        test_ref_set=test_ref_set,
        test_qry_set=test_qry_set,
        train_fit_loader=train_fit_loader,
        dev_ref_loader=dev_ref_loader,
        dev_qry_loader=dev_qry_loader,
        test_ref_loader=test_ref_loader,
        test_qry_loader=test_qry_loader,
        id_counts=id_counts,
    )


# def build_reid_pipeline(
#     cfg,
#     train_transform,
#     val_transform,
#     *,
#     col_label: str = "identity",
#     frac_train_ids: float = 0.8,   # fraction of identities used for training
#     ratio_fit: float = 0.9,        # within-train images: fraction to fit (vs dev)
#     ratio_gallery_dev: float = 0.5,# dev gallery vs query
#     ratio_gallery_test: float = 0.5,# test gallery vs query
# ) -> ReIDSplits:
#     """
#     Correct pipeline:
#       1) Split identities -> train_ids, test_ids
#       2) Within train_ids: split images -> train_fit vs dev_all
#       3) dev_all -> (dev_ref, dev_qry) closed-set
#       4) test_ids -> (test_ref, test_qry) closed-set
#       5) Build datasets/loaders
#     """
#     seed = getattr(cfg, "seed", 42)

#     # 1) Load metadata
#     meta = WildlifeReID10k(cfg.root)
#     df = meta.df  # pandas DataFrame

#     print(df.head())
#     print(df.columns)
#     # print(df['species'].unique())
#     # df = df[df['species'] == 'sea turtle']
#     # print(f"Unique datasets used to train: {df['dataset'].unique()} !!!")
#     # print(df.head())
#     # print(df.columns)
#     # print("Total images:", len(df))
#     # Filter for multiple species at once, e.g. turtles + beluga_whales
#     # subset = df[df['species'].isin(['turtle', 'beluga_whale'])]

#     # 2) Identities: train vs test (no overlap)
#     train_ids, test_ids = _split_identities(
#         df, frac_train_ids=frac_train_ids, seed=seed, col_label=col_label
#     )

#     df_train_all = df[df[col_label].isin(train_ids)].copy()
#     df_test_all  = df[df[col_label].isin(test_ids)].copy()

#     # 3) Within training identities: images -> fit vs dev (same IDs on both sides)
#     df_train_fit, df_dev_all = _train_fit_vs_dev(
#         df_train_all, ratio_fit=ratio_fit, col_label=col_label
#     )

#     # 4) Dev: gallery/query (closed-set)
#     df_dev_ref, df_dev_qry = _closed_split_gallery_query(
#         df_dev_all, ratio_gallery=ratio_gallery_dev, col_label=col_label
#     )

#     # 5) Test: gallery/query (closed-set)
#     df_test_ref, df_test_qry = _closed_split_gallery_query(
#         df_test_all, ratio_gallery=ratio_gallery_test, col_label=col_label
#     )

#     # 6) Datasets & Loaders
#     train_fit_set = WR10kDataset(df_train_fit, cfg.root, train_transform)
#     train_fit_loader = DataLoader(
#         train_fit_set,
#         batch_size=cfg.train_batch,
#         num_workers=cfg.num_workers,
#         pin_memory=True,
#         drop_last=False,
#         shuffle=True,
#     )

#     dev_ref_set, dev_qry_set, dev_ref_loader, dev_qry_loader = _make_eval_sets_and_loaders(
#         df_dev_ref, df_dev_qry, cfg.root, cfg.eval_batch, val_transform, cfg.num_workers
#     )
#     test_ref_set, test_qry_set, test_ref_loader, test_qry_loader = _make_eval_sets_and_loaders(
#         df_test_ref, df_test_qry, cfg.root, cfg.eval_batch, val_transform, cfg.num_workers
#     )

#     # Optional: some quick bookkeeping you might find handy for sanity checks
#     id_counts = {
#         "train_fit_ids": len(set(df_train_fit[col_label].unique())),
#         "dev_ids":       len(set(df_dev_all[col_label].unique())),
#         "test_ids":      len(set(df_test_all[col_label].unique())),
#         "dev_ref_n":     len(df_dev_ref),
#         "dev_qry_n":     len(df_dev_qry),
#         "test_ref_n":    len(df_test_ref),
#         "test_qry_n":    len(df_test_qry),
#     }

#     return ReIDSplits(
#         # datasets
#         train_fit_set=train_fit_set,
#         dev_ref_set=dev_ref_set,
#         dev_qry_set=dev_qry_set,
#         test_ref_set=test_ref_set,
#         test_qry_set=test_qry_set,
#         # loaders
#         train_fit_loader=train_fit_loader,
#         dev_ref_loader=dev_ref_loader,
#         dev_qry_loader=dev_qry_loader,
#         test_ref_loader=test_ref_loader,
#         test_qry_loader=test_qry_loader,
#         # info
#         id_counts=id_counts,
#     )


# --- splitting helpers ---
def _filter_min_images_per_id(df: pd.DataFrame, col_label: str, min_images: int) -> pd.DataFrame:
    """Keep only identities with at least `min_images` total images."""
    counts = df.groupby(col_label).size()
    keep = counts[counts >= min_images].index
    return df[df[col_label].isin(keep)].copy()

def _per_id_image_split(
    df_block: pd.DataFrame,
    ratio_a: float,
    col_label: str = "identity",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *within each identity* so that each surviving identity contributes
    at least 1 image to both sides. Identities with <2 images are dropped.
    """
    g = _rng(seed)

    parts_a = []
    parts_b = []
    for _, gdf in df_block.groupby(col_label, sort=False):
        n = len(gdf)
        if n < 2:
            # can't make both sides non-empty -> drop this identity
            continue
        # how many go to A (e.g., gallery or train_fit)
        n_a = int(round(n * ratio_a))
        # force at least 1 on each side
        n_a = max(1, min(n_a, n - 1))

        idx = g.permutation(gdf.index.to_numpy())
        a_idx = idx[:n_a]
        b_idx = idx[n_a:]

        parts_a.append(gdf.loc[a_idx])
        parts_b.append(gdf.loc[b_idx])

    if parts_a:
        df_a = pd.concat(parts_a, axis=0)
        df_b = pd.concat(parts_b, axis=0)
    else:
        # nothing had >=2 images
        df_a = df_block.iloc[0:0].copy()
        df_b = df_block.iloc[0:0].copy()
    return df_a, df_b

def _safe_closed_split_gallery_query(
    df_block: pd.DataFrame,
    ratio_gallery: float = 0.5,
    col_label: str = "identity",
    seed: int = 42,
    max_tries: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gallery/Query split that guarantees both sides non-empty if possible.
    We: (1) filter to ids with >=2 images; (2) per-identity split; (3) retry seeds if needed.
    """
    min_images = 2
    df_work = _filter_min_images_per_id(df_block, col_label, min_images=min_images)
    import warnings
    warnings.warn(f"Limiting to dataset to individuals with >={min_images} per identity for gallery/query split.")

    for t in range(max_tries):
        df_ref, df_qry = _per_id_image_split(
            df_work, ratio_gallery, col_label=col_label, seed=seed + t
        )
        if len(df_ref) > 0 and len(df_qry) > 0:
            return df_ref, df_qry

    # If we get here, there are simply no identities with >=2 images.
    # Return empty frames (caller can decide what to do).
    return df_work.iloc[0:0].copy(), df_work.iloc[0:0].copy()

def _safe_train_fit_vs_dev(
    df_train_all: pd.DataFrame,
    ratio_fit: float = 0.9,
    col_label: str = "identity",
    seed: int = 42,
    max_tries: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training *images* into fit/dev such that every id present appears
    on both sides; drop ids with <2 images; retry a few seeds if necessary.
    """
    df_work = _filter_min_images_per_id(df_train_all, col_label, min_images=2)

    for t in range(max_tries):
        df_fit, df_dev = _per_id_image_split(
            df_work, ratio_fit, col_label=col_label, seed=seed + t
        )
        if len(df_fit) > 0 and len(df_dev) > 0:
            return df_fit, df_dev

    return df_work.iloc[0:0].copy(), df_work.iloc[0:0].copy()
