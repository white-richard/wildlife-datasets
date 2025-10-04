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
    frac_train_ids: float = 0.8,   # fraction of identities used for training
    ratio_fit: float = 0.9,        # within-train images: fraction to fit (vs dev)
    ratio_gallery_dev: float = 0.5,# dev gallery vs query
    ratio_gallery_test: float = 0.5,# test gallery vs query
) -> ReIDSplits:
    """
    Correct pipeline:
      1) Split identities -> train_ids, test_ids
      2) Within train_ids: split images -> train_fit vs dev_all
      3) dev_all -> (dev_ref, dev_qry) closed-set
      4) test_ids -> (test_ref, test_qry) closed-set
      5) Build datasets/loaders
    """
    seed = getattr(cfg, "seed", 42)

    # 1) Load metadata
    meta = WildlifeReID10k(cfg.root)
    df = meta.df  # pandas DataFrame

    # print(df.head())
    # print(df.columns)

    # print(df['species'].unique())

    df = df[df['species'] == 'sea turtle']
    print(f"Unique datasets used to train: {df['dataset'].unique()} !!!")
    # print(df.head())
    # print(df.columns)
    # print("Total images:", len(df))

    # Filter for multiple species at once, e.g. turtles + beluga_whales
    # subset = df[df['species'].isin(['turtle', 'beluga_whale'])]

    # 2) Identities: train vs test (no overlap)
    train_ids, test_ids = _split_identities(
        df, frac_train_ids=frac_train_ids, seed=seed, col_label=col_label
    )

    df_train_all = df[df[col_label].isin(train_ids)].copy()
    df_test_all  = df[df[col_label].isin(test_ids)].copy()

    # 3) Within training identities: images -> fit vs dev (same IDs on both sides)
    df_train_fit, df_dev_all = _train_fit_vs_dev(
        df_train_all, ratio_fit=ratio_fit, col_label=col_label
    )

    # 4) Dev: gallery/query (closed-set)
    df_dev_ref, df_dev_qry = _closed_split_gallery_query(
        df_dev_all, ratio_gallery=ratio_gallery_dev, col_label=col_label
    )

    # 5) Test: gallery/query (closed-set)
    df_test_ref, df_test_qry = _closed_split_gallery_query(
        df_test_all, ratio_gallery=ratio_gallery_test, col_label=col_label
    )

    # 6) Datasets & Loaders
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

    # Optional: some quick bookkeeping you might find handy for sanity checks
    id_counts = {
        "train_fit_ids": len(set(df_train_fit[col_label].unique())),
        "dev_ids":       len(set(df_dev_all[col_label].unique())),
        "test_ids":      len(set(df_test_all[col_label].unique())),
        "dev_ref_n":     len(df_dev_ref),
        "dev_qry_n":     len(df_dev_qry),
        "test_ref_n":    len(df_test_ref),
        "test_qry_n":    len(df_test_qry),
    }

    return ReIDSplits(
        # datasets
        train_fit_set=train_fit_set,
        dev_ref_set=dev_ref_set,
        dev_qry_set=dev_qry_set,
        test_ref_set=test_ref_set,
        test_qry_set=test_qry_set,
        # loaders
        train_fit_loader=train_fit_loader,
        dev_ref_loader=dev_ref_loader,
        dev_qry_loader=dev_qry_loader,
        test_ref_loader=test_ref_loader,
        test_qry_loader=test_qry_loader,
        # info
        id_counts=id_counts,
    )
