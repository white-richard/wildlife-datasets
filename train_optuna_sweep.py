"""
Modular WR10k trainer with clean structure and flexible Weights & Biases (W&B) usage.

Supports three modes:
  - --wandb off      : no W&B imports or logging (safe to run without wandb installed)
  - --wandb online   : standard wandb.init / wandb.log run
  - --wandb sweep    : sweep-friendly; hyperparams are pulled from wandb.config if present

Training paradigms are pluggable via a simple Strategy registry (see `TrainingParadigm`).
You can add new paradigms without touching the main training loop.

This file keeps your original custom dependencies but isolates them behind builders.


python train_mega_descriptor.py \
  --tune-trials -1 \
  --tune-direction maximize \
  --tune-storage postgresql+psycopg2://optuna:optuna@100.90.126.94:5432/wr10k \
  --tune-study wr10k_sweep \
  --wandb online --project reproduce_mega_descriptor \
  --tune-seed 42

"""
from __future__ import annotations
import os
import copy
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
import torch.nn.init as init
import torchvision.transforms as T
import timm
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm, trange
import wandb as _wandb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

from wildlife_tools.train.objective import ArcFaceLoss
from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets import splits

from geoopt import ManifoldParameter
import hypercore.nn as hnn
from hypercore.utils.manifold_utils import lock_curvature_to_one
from hypercore.models.Swin_LViT import LSwin_small
from hypercore.manifolds.lorentzian import Lorentz
from hypercore.optimizers import RiemannianSGD, RiemannianAdam
from hypercore.modules.loss import LorentzTripletLoss

from eucTohyp_Swin import replace_stages_with_hyperbolic
from wr10k_dataset import WR10kDataset
from knn_per_class import evaluate_knn1
from hyp_knn_per_class import evaluate_knn1 as hyp_evaluate_knn1
from wandb_session import WandbSession, WandbMode
from augmentations import AugCfg, build_train_tfms
from knn_val_monitor import MemoryBankQueue, knn_top1_on_batch
from hyp_knn_val_monitor import knn_top1_on_batch_lorentz as hyp_knn_top1_on_batch, MemoryBankQueue as HypMemoryBankQueue
from reid_split_wr10k import build_reid_pipeline
from validation import validate_split


torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 0) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@dataclass
class Config:
    # run meta
    run_name: str = "wr10k_megadesc_arcface"
    seed: int = 42
    save_dir: str = "checkpoints"

    # data
    root: str = "../wildlifereid-10k"
    img_size: int = 224
    num_workers: int = 16
    train_batch: int = 128
    eval_batch: int = 128
    aug_policy: str = "baseline"  # baseline | weak | strong | randaug | augmix

    # model / paradigm
    model_name: str = "megadesc_replace_last_layer_hyperbolic"  # choices below
    loss_type: str = "triplet"  # arcface | triplet
    hyperbolic: bool = True

    # optimization
    epochs: int = 40
    lr: float = 1e-3
    wd: float = 1e-4
    momentum: float = 0.9
    warmup_t: int = 5
    lr_min: float = 1e-6
    optimizer_name: str = "sgd"  # sgd | adam

    tune_trials: int = 0              # 0 = no tuning; >0 runs optuna
    tune_direction: str = "maximize"  # "maximize" mAP
    tune_storage: Optional[str] = None  # e.g. "sqlite:///wr10k_optuna.db"
    tune_study: Optional[str] = None    # study name if you want persistence
    tune_pruner: str = "hyperband"       # median | sha | hyperband | none
    tune_sampler: str = "tpe"         # currently only tpe wired
    tune_seed: int = 42

    # training loop
    accumulation_steps: int = 1
    use_amp: bool = False
    val_interval: int = 1
    patience: int = 10

    # logging
    wandb_mode: WandbMode = WandbMode.OFF
    project: str = "reproduce_mega_descriptor"

class LorentzEmbToTangent(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        v = self.manifold.logmap0(x)
        return v[..., 1:]


def init_weights_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def split_decay_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    ln_param_names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (nn.LayerNorm, hnn.LorentzLayerNorm)):
            for p_name, _ in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                ln_param_names.add(full)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or name in ln_param_names:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def remove_curvature_from_optimizer(optimizer, manifold) -> None:
    params_to_remove = []
    for attr in ("k", "c"):
        if hasattr(manifold, attr):
            val = getattr(manifold, attr)
            if isinstance(val, torch.nn.Parameter):
                params_to_remove.append(val)
    if not params_to_remove:
        return
    remove_ids = {id(p) for p in params_to_remove}
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in remove_ids]
    for p in list(optimizer.state.keys()):
        if id(p) in remove_ids:
            optimizer.state.pop(p, None)
            print(f"Dropped curvature param {p} from optimizer.")

def suggest_params(trial: "optuna.trial.Trial", base_cfg: Config) -> Dict[str, object]:
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "epochs": trial.suggest_int("epochs", 10, 60, step=10),
        "train_batch": trial.suggest_categorical("train_batch", [64, 96, 128, 160]),
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["sgd", "adam"]),
        "aug_policy": trial.suggest_categorical("aug_policy", ["baseline","weak","strong","randaug","augmix"]),
    }
    return params

def _run_one_trial(trial: "optuna.trial.Trial", base_cfg: Config):
    cfg = copy.deepcopy(base_cfg)

    suggested = suggest_params(trial, cfg)
    for k, v in suggested.items():
        setattr(cfg, k, type(getattr(cfg, k))(v))

    cfg.run_name = f"{base_cfg.run_name}-t{trial.number}"

    if _wandb.run is not None:
        _wandb.config.update(suggested, allow_val_change=True)

    set_seed(cfg.seed)
    with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        data = DataBuilder(cfg)
        loaders, datasets = data.build()

        mb = ModelBuilder(cfg)
        backbone, emb_dim, manifold = mb.build()
        num_classes = datasets["train"].num_classes
        objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold)

        ob = OptimBuilder(cfg)
        optimizer, scheduler = ob.build(backbone, objective, manifold)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone = backbone.to(device); objective = objective.to(device)

        try:
            best_mAP = fit(cfg, backbone, objective, loaders, optimizer, scheduler,
                device, wb, emb_dim, manifold, trial=trial)
        except optuna.exceptions.TrialPruned:
            raise

        overall, _ = _eval_knn(backbone, loaders, device, hyperbolic=cfg.hyperbolic, manifold=manifold)
        return best_mAP


def run_optuna(cfg: Config):
    if optuna is None:
        raise RuntimeError("optuna is not installed. pip install optuna")

    if cfg.tune_sampler == "tpe":
        sampler = TPESampler(seed=cfg.seed)
    else:
        sampler = TPESampler(seed=cfg.seed)

    # pruner
    pruner_map = {
        "median": MedianPruner(n_warmup_steps=3),
        "sha": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
        "none": None,
    }
    pruner = pruner_map[cfg.tune_pruner]

    study = optuna.create_study(
        direction=cfg.tune_direction,
        sampler=sampler,
        pruner=pruner,
        storage=cfg.tune_storage,
        study_name=cfg.tune_study,
        load_if_exists=bool(cfg.tune_storage and cfg.tune_study),
    )
    if isinstance(study._storage, optuna.storages._rdb.storage.RDBStorage):
        study._storage._heartbeat_interval = 60  # seconds
        study._storage._fail_stale_trials = True


    def _objective(trial: optuna.trial.Trial):
        return _run_one_trial(trial, cfg)

    # study.optimize(_objective, n_trials=cfg.tune_trials, show_progress_bar=True)
    n_trials = None if cfg.tune_trials < 0 else cfg.tune_trials
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)


    print(f"[Optuna] Best value={study.best_value:.6f}")
    print(f"[Optuna] Best params={study.best_trial.params}")
    return study


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
class DataBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.aug_cfg = AugCfg(self.cfg.aug_policy)
    def _train_tfms(self) -> T.Compose:
        return build_train_tfms(self.cfg.img_size, self.aug_cfg)
    
    def _eval_tfms(self) -> T.Compose:
        d = self.cfg.img_size
        return T.Compose([
            T.Resize(int(d / 0.875), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(d),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def build(self):

        splits_out = build_reid_pipeline(
            self.cfg,
            train_transform=self._train_tfms(),
            val_transform=self._eval_tfms(),
            col_label="identity",
            frac_train_ids=0.8,      # 80% IDs used for training, 20% reserved for test
            ratio_fit=0.9,           # within-training images: 90% for fitting, 10% for dev
            ratio_gallery_dev=0.5,   # dev gallery/query ratio
            ratio_gallery_test=0.5,  # test gallery/query ratio
        )
        train_fit_loader = splits_out.train_fit_loader  # batches for optimizing weights
        val_ref_loader   = splits_out.dev_ref_loader    # eval: gallery (val)
        val_qry_loader   = splits_out.dev_qry_loader    # eval: query  (val)
        test_ref_loader  = splits_out.test_ref_loader   # eval: gallery (test)
        test_qry_loader  = splits_out.test_qry_loader   # eval: query  (test)
        train_fit_set = splits_out.train_fit_set
        loaders = {
            "train": train_fit_loader,
            "val_ref": val_ref_loader,
            "val_qry": val_qry_loader,
            "test_ref": test_ref_loader,
            "test_qry": test_qry_loader,
        }
        dev_ref_set   = splits_out.dev_ref_set
        dev_qry_set   = splits_out.dev_qry_set
        test_ref_set  = splits_out.test_ref_set
        test_qry_set  = splits_out.test_qry_set
        datasets = {
            "train": train_fit_set,
            "val_ref": dev_ref_set,
            "val_qry": dev_qry_set,
            "test_ref": test_ref_set,
            "test_qry": test_qry_set,
        }
        return loaders, datasets

        meta = WildlifeReID10k(self.cfg.root)
        df = meta.df
        splitter = splits.ClosedSetSplit(ratio_train=0.9, col_label="identity")
        idx_ref, idx_qry = splitter.split(df)[0]
        df_ref = df.iloc[idx_ref].copy()
        df_qry = df.iloc[idx_qry].copy()

        # Enforce identical identity sets across splits (closed-set)
        train_ids = set(df_ref['identity'].unique())
        val_ids = set(df_qry['identity'].unique())
        if train_ids != val_ids:
            common = train_ids & val_ids
            df_ref = df_ref[df_ref['identity'].isin(common)].copy()
            df_qry = df_qry[df_qry['identity'].isin(common)].copy()

        train_set = WR10kDataset(df_ref, self.cfg.root, self._train_tfms())
        id_list = sorted(train_set.id2idx.keys())
        ref_eval_set = WR10kDataset(df_ref, self.cfg.root, self._eval_tfms(), id_list=id_list)
        val_set = WR10kDataset(df_qry, self.cfg.root, self._eval_tfms(), id_list=id_list)

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.train_batch,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
        )
        ref_eval_set_loader = DataLoader(ref_eval_set, batch_size=self.cfg.eval_batch, shuffle=False,
                                         num_workers=self.cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.cfg.eval_batch, shuffle=False,
                                num_workers=self.cfg.num_workers, pin_memory=True)

        return train_set, ref_eval_set, val_set, train_loader, ref_eval_set_loader, val_loader


# ------------------------------------------------------------------------------------------------
# Model/Objective/Optim builders
# ------------------------------------------------------------------------------------------------
class ModelBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self):
        model_name = self.cfg.model_name
        manifold = None
        emb_dim = None

        if model_name == 'hyp_swin':
            manifold = Lorentz()
            lock_curvature_to_one(manifold)
            backbone = LSwin_small(manifold, manifold, manifold, num_classes=0, embed_dim=32+1)
            for p in backbone.parameters():
                if isinstance(p, ManifoldParameter):
                    p.requires_grad_(False)
            backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)
                out = backbone(dummy)
            backbone.train()
            emb_dim = out.shape[1]
            # backbone = nn.Sequential(backbone, LorentzEmbToTangent(manifold))

        elif model_name == 'megadescriptor_swin':
            backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=False)
            emb_dim = backbone.num_features
            backbone.apply(init_weights_xavier)

        elif model_name == 'megadescriptor_swin_pretrained':
            backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True)
            emb_dim = backbone.num_features

        elif model_name == 'megadesc_replace_last_layer_hyperbolic':
            backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True)
            backbone, hyp_stages = replace_stages_with_hyperbolic(backbone, num_stages=1)
            if hasattr(backbone, "norm"):
                backbone.norm = nn.Identity()
            if hasattr(backbone, "global_pool"):
                backbone.global_pool = ""
            backbone.reset_classifier(num_classes=0, global_pool='')
            # freeze original weights & unfreeze inserted hyp blocks
            for n, p in backbone.named_parameters():
                p.requires_grad = False
            for hyp_stage in hyp_stages:
                for p in hyp_stage.parameters():
                    p.requires_grad = True
            # emb_dim = 2049  # 2048 + 1 (time coord)
            backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)
                out = backbone(dummy)
            backbone.train()
            emb_dim = out.shape[1]
            manifold = hyp_stages[-1].manifold
        else:
            raise ValueError(f"Unknown model_name {model_name}")

        return backbone, emb_dim, manifold

class ObjectiveBuilder:
    @staticmethod
    def build(loss_type: str, num_classes: int, emb_dim: int, hyperbolic: bool, manifold=None):
        if loss_type == 'arcface':
            return ArcFaceLoss(num_classes=num_classes, embedding_size=emb_dim, margin=0.5, scale=64)
        if loss_type == 'triplet':
            if hyperbolic:
                return LorentzTripletLoss(manifold, margin=0.2, type_of_triplets='hard', feature_dim=emb_dim)
            return nn.TripletMarginLoss(margin=0.2, p=2)
        raise ValueError(f"Unknown loss_type {loss_type}")


class OptimBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def build(self, backbone: nn.Module, head: nn.Module, manifold=None):
        wd = self.cfg.wd
        decay, no_decay = split_decay_groups(backbone)
        head_params = [p for p in head.parameters() if p.requires_grad]

        if self.cfg.hyperbolic:
            if self.cfg.optimizer_name == "adam":
                optimizer = RiemannianAdam([
                    {"params": decay,    "weight_decay": wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ], lr=self.cfg.lr, stabilize=1)
                if self.cfg.loss_type != 'triplet':
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
                if manifold is not None:
                    remove_curvature_from_optimizer(optimizer, manifold)
            else:
                optimizer = RiemannianSGD([
                    {"params": decay,    "weight_decay": wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ], lr=self.cfg.lr, momentum=self.cfg.momentum, stabilize=1)
                if self.cfg.loss_type != 'triplet':
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
                if manifold is not None:
                    remove_curvature_from_optimizer(optimizer, manifold)
        else:
            if self.cfg.optimizer_name == "adam":
                optimizer = torch.optim.Adam([
                    {"params": decay,    "weight_decay": wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ], lr=self.cfg.lr)
                if self.cfg.loss_type != 'triplet':
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
            else:
                optimizer = torch.optim.SGD([
                    {"params": decay,    "weight_decay": wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ], lr=self.cfg.lr, momentum=self.cfg.momentum)
                if self.cfg.loss_type != 'triplet':
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})

        # Cosine schedule in epochs
        for g in optimizer.param_groups:
            g.setdefault('initial_lr', g['lr'])
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.cfg.epochs,
            lr_min=self.cfg.lr_min,
            warmup_lr_init=self.cfg.lr_min,
            warmup_t=self.cfg.warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
        )
        return optimizer, scheduler


# ------------------------------------------------------------------------------------------------
# Training paradigms (Strategy)
# ------------------------------------------------------------------------------------------------
class TrainingParadigm:
    REGISTRY: Dict[str, 'TrainingParadigm'] = {}

    def __init_subclass__(cls, key: Optional[str] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if key:
            TrainingParadigm.REGISTRY[key] = cls()  # type: ignore

    def criterion_forward(self, head: nn.Module, feats: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def post_forward(self, feats: torch.Tensor, use_amp: bool) -> torch.Tensor:
        # default: no change
        return feats


class ArcFaceParadigm(TrainingParadigm, key="arcface"):
    def criterion_forward(self, head, feats, y):
        return head(feats, y)


class TripletParadigm(TrainingParadigm, key="triplet"):
    def criterion_forward(self, head, feats, y):
        return head(feats, y)


@torch.no_grad()
def _eval_knn(backbone: nn.Module, loaders: List[DataLoader], device: torch.device, hyperbolic: bool = False, manifold=None) -> Tuple[float, Dict[str, float]]:
    ref_loader = loaders['test_ref']
    qry_loader = loaders['test_qry']
    if hyperbolic:
        return hyp_evaluate_knn1(backbone, ref_loader, qry_loader, device=device, manifold=manifold)
    return evaluate_knn1(backbone, ref_loader, qry_loader, device=device)


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loaders: List[DataLoader],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: Optional[amp.GradScaler] = None,
    accumulation_steps: int = 1,
    device: str = "cuda",
    use_amp: bool = False,
    paradigm: TrainingParadigm | None = None,
    hyperbolic: bool = False,
):
    train_loader = loaders['train']
    model.train(); head.train()
    steps_per_epoch = len(train_loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, leave=False)

    validate_every = max(1, steps_per_epoch - 1)

    running_metric = 0.0
    metric_count = 0

    if paradigm is None:
        raise ValueError("Training paradigm must be provided")

    for i, batch in pbar:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16, enabled=use_amp):
            feats = model(x)
            loss = paradigm.criterion_forward(head, feats, y) / accumulation_steps
        feats = paradigm.post_forward(feats, use_amp)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.detach()
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if (i % validate_every == 0) or (i == steps_per_epoch - 1):
            with torch.no_grad():
                metrics = validate_split(
                    model=model,
                    loaders=loaders,
                    split_prefix="val",  # uses "val_ref" and "val_qry"
                    device=device,
                    use_amp=use_amp,
                    sim_metric="neg_lorentz_geo" if hyperbolic else "cosine",
                )
                mAP = metrics['mAP']
                acc1 = metrics['top1']
                acc5 = metrics['top5']
                acc10 = metrics['top10']
                running_metric += mAP
                metric_count += 1
                if _wandb.run is not None:
                    _wandb.log({
                        "train/val_mAP": mAP,
                        "train/val_top1": acc1,
                        "train/val_top5": acc5,
                        "train/val_top10": acc10,
                        "train/iter": epoch * steps_per_epoch + i + 1,
                        "epoch": epoch,
                    })
                else:
                    print(f"epoch {epoch} | iter {i+1}/{steps_per_epoch} "
                        f"| loss {float(running_loss)/(i+1):.4f} | val_mAP {mAP:.4f} | val_top1 {acc1:.4f} | val_top5 {acc5:.4f} | val_top10 {acc10:.4f}")
        
        pbar.set_description(f"epoch {epoch} | loss {float(running_loss)/(i+1):.4f}",)
    avg_acc = float(running_metric) / metric_count if metric_count > 0 else float("nan")
    return float(running_loss) / steps_per_epoch, avg_acc

def fit(
    cfg: Config,
    backbone: nn.Module,
    objective: nn.Module,
    loaders: List[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    device: torch.device,
    logger: WandbSession,
    emb_dim: int,
    manifold=None,
    trial: "optuna.trial.Trial | None" = None,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_loss = float('inf')
    best_mAP = -float('inf')
    best_epoch = -1
    no_improve = 0

    if cfg.hyperbolic and cfg.loss_type != 'triplet':
        raise ValueError("Hyperbolic training is only supported with triplet loss currently.")

    paradigm = TrainingParadigm.REGISTRY[cfg.loss_type]
    scaler = amp.GradScaler('cuda', enabled=cfg.use_amp)

    for epoch in trange(1, cfg.epochs + 1, desc="Epochs"):
        train_loss, train_mAP = train_one_epoch(
            backbone, objective, loaders, optimizer, epoch,
            scaler=scaler, accumulation_steps=cfg.accumulation_steps, device=str(device),
            use_amp=cfg.use_amp, paradigm=paradigm,
            hyperbolic=cfg.hyperbolic,
        )
        scheduler.step(epoch + 1)

        if train_mAP > best_mAP:
            best_mAP = train_mAP; best_epoch = epoch; no_improve = 0
            torch.save({
                'epoch': epoch,
                'model': backbone.state_dict(),
                'head': objective.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'seed': cfg.seed,
                'best_mAP': best_mAP,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, os.path.join(cfg.save_dir, f'{cfg.run_name}_best_epoch{epoch}.pt'))
            print(f"[VAL] New best @ epoch {epoch}: val_mAP={best_mAP:.4f} (saved)")
        else:
            no_improve += 1
            print(f"[VAL] train_loss={train_loss:.4f} (best={best_mAP:.4f} @ {best_epoch}) | patience {no_improve}/{cfg.patience}")
        
        if _wandb.run is not None:
            _wandb.log({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train/mAP": float(train_mAP),
            })
        if trial is not None:
            trial.report(train_mAP, step=epoch)
            if trial.should_prune():
                if _wandb.run is not None:
                    _wandb.summary["pruned_at_epoch"] = epoch
                raise optuna.exceptions.TrialPruned()
        
        logger.log({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val/mAP": float(train_mAP),
        })
        if no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    torch.save({
        'epoch': epoch,
        'model': backbone.state_dict(),
        'head': objective.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'seed': cfg.seed,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }, os.path.join(cfg.save_dir, f'{cfg.run_name}_final.pt'))
    print(f"\nTraining done. Best loss={best_loss:.4f} at epoch {best_epoch}. Final checkpoint saved.")

    overall, per_dataset_acc = _eval_knn(backbone, loaders, device, hyperbolic=cfg.hyperbolic, manifold=manifold)
    logger.log({"metrics/overall_acc": overall})
    if logger.active and _wandb is not None:
        table = _wandb.Table(columns=["dataset", "accuracy"])
        for ds, acc in sorted(per_dataset_acc.items()):
            table.add_data(ds, acc)
            _wandb.summary[f"acc_by_dataset/{ds}"] = acc
        logger.log({"per_dataset_accuracy_table": table})
        logger.log({"plots/per_dataset_accuracy": _wandb.plot.bar(table, "dataset", "accuracy", title="Per-dataset Top-1 Accuracy")})
    else:
        print(f"Overall acc: {overall}.")
    
    return best_mAP


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Modular WR10k trainer")
    p.add_argument('--run-name', type=str, default=Config.run_name)
    p.add_argument('--seed', type=int, default=Config.seed)
    p.add_argument('--save-dir', type=str, default=Config.save_dir)
    p.add_argument('--root', type=str, default=Config.root)
    p.add_argument('--img-size', type=int, default=Config.img_size)
    p.add_argument('--num-workers', type=int, default=Config.num_workers)
    p.add_argument('--train-batch', type=int, default=Config.train_batch)
    p.add_argument('--eval-batch', type=int, default=Config.eval_batch)

    p.add_argument('--model-name', type=str, default=Config.model_name,
                   choices=['hyp_swin','megadescriptor_swin','megadescriptor_swin_pretrained','megadesc_replace_last_layer_hyperbolic'])
    p.add_argument('--loss-type', type=str, default=Config.loss_type, choices=list(TrainingParadigm.REGISTRY.keys()))
    p.add_argument('--hyperbolic', action='store_true', default=Config.hyperbolic)

    p.add_argument('--epochs', type=int, default=Config.epochs)
    p.add_argument('--lr', type=float, default=Config.lr)
    p.add_argument('--wd', type=float, default=Config.wd)
    p.add_argument('--momentum', type=float, default=Config.momentum)
    p.add_argument('--warmup-t', type=int, default=Config.warmup_t)
    p.add_argument('--lr-min', type=float, default=Config.lr_min)
    p.add_argument('--optimizer-name', type=str, default=Config.optimizer_name,
                   choices=['sgd','adam'])

    p.add_argument('--accumulation-steps', type=int, default=Config.accumulation_steps)
    p.add_argument('--use-amp', action='store_true', default=Config.use_amp)
    p.add_argument('--val-interval', type=int, default=Config.val_interval)
    p.add_argument('--patience', type=int, default=Config.patience)

    p.add_argument('--wandb', type=str, default=Config.wandb_mode.value, choices=[m.value for m in WandbMode])
    p.add_argument('--project', type=str, default=Config.project)

    p.add_argument('--tune-trials', type=int, default=Config.tune_trials)
    p.add_argument('--tune-direction', type=str, default=Config.tune_direction, choices=['minimize','maximize'])
    p.add_argument('--tune-storage', type=str, default=Config.tune_storage)
    p.add_argument('--tune-study', type=str, default=Config.tune_study)
    p.add_argument('--tune-pruner', type=str, default=Config.tune_pruner, choices=['median','sha','hyperband','none'])
    p.add_argument('--tune-sampler', type=str, default=Config.tune_sampler, choices=['tpe'])
    p.add_argument('--tune-seed', type=int, default=Config.tune_seed)

    args = p.parse_args()
    cfg = Config(
        run_name=args.run_name,
        seed=args.seed,
        save_dir=args.save_dir,
        root=args.root,
        img_size=args.img_size,
        num_workers=args.num_workers,
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        model_name=args.model_name,
        loss_type=args.loss_type,
        hyperbolic=args.hyperbolic,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        momentum=args.momentum,
        warmup_t=args.warmup_t,
        lr_min=args.lr_min,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        val_interval=args.val_interval,
        patience=args.patience,
        wandb_mode=WandbMode(args.wandb),
        project=args.project,
        optimizer_name=args.optimizer_name,
        tune_trials=args.tune_trials,
        tune_direction=args.tune_direction,
        tune_storage=args.tune_storage,
        tune_study=args.tune_study,
        tune_pruner=args.tune_pruner,
        tune_sampler=args.tune_sampler,
        tune_seed=args.seed,
    )
    return cfg

# CHANGE: rewrite main() tail
def main():
    cfg = _parse_args()
    if cfg.tune_trials and cfg.tune_trials > 0:
        run_optuna(cfg)
        return

    set_seed(cfg.seed)
    with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        data = DataBuilder(cfg)
        loaders, datasets = data.build()

        mb = ModelBuilder(cfg)
        backbone, emb_dim, manifold = mb.build()
        num_classes = datasets["train"].num_classes
        objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold)

        ob = OptimBuilder(cfg)
        optimizer, scheduler = ob.build(backbone, objective, manifold)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone = backbone.to(device); objective = objective.to(device)

        fit(cfg, backbone, objective, loaders, optimizer, scheduler, device, wb, emb_dim, manifold)


# def main():
#     cfg = _parse_args()
#     set_seed(cfg.seed)
#     # Open W&B first so sweep values are available
#     with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
#         live = wb.cfg  # this is wandb.config (sweep values if present)
#         # allow-list of keys we accept from sweeps
#         for k in [
#             "lr","wd","epochs","train_batch","eval_batch","img_size",
#             "accumulation_steps","use_amp","val_interval","patience","optimizer_name",
#             "aug_policy"
#         ]:
#             if k in live:
#                 setattr(cfg, k, type(getattr(cfg, k))(live[k]))

#         data = DataBuilder(cfg)
#         loaders, datasets = data.build()

#         mb = ModelBuilder(cfg)
#         backbone, emb_dim, manifold = mb.build()
#         num_classes = datasets["train"].num_classes
#         objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold)

#         ob = OptimBuilder(cfg)
#         optimizer, scheduler = ob.build(backbone, objective, manifold)

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         backbone = backbone.to(device); objective = objective.to(device)

#         # Train
#         fit(cfg, backbone, objective, loaders, optimizer, scheduler,
#             device, wb, emb_dim, manifold)

if __name__ == '__main__':
    main()