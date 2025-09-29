"""
Modular WR10k trainer with clean structure and flexible Weights & Biases (W&B) usage.

Supports three modes:
  - --wandb off      : no W&B imports or logging (safe to run without wandb installed)
  - --wandb online   : standard wandb.init / wandb.log run
  - --wandb sweep    : sweep-friendly; hyperparams are pulled from wandb.config if present

Training paradigms are pluggable via a simple Strategy registry (see `TrainingParadigm`).
You can add new paradigms without touching the main training loop.

This file keeps your original custom dependencies but isolates them behind builders.
"""
from __future__ import annotations
import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Tuple, Dict, Any, Iterable, List

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
from wandb_session import WandbSession, WandbMode

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
    root: str = "/home/richw/.code/wildlifereid-10k"
    img_size: int = 224
    num_workers: int = 16
    train_batch: int = 128
    eval_batch: int = 128

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

    # training loop
    accumulation_steps: int = 1
    use_amp: bool = False
    val_interval: int = 1
    patience: int = 15

    # logging
    wandb_mode: WandbMode = WandbMode.OFF
    project: str = "reproduce_mega_descriptor"


# ------------------------------------------------------------------------------------------------
# Helper modules & builders
# ------------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
class DataBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _train_tfms(self) -> T.Compose:
        d = self.cfg.img_size
        return T.Compose([
            T.RandomResizedCrop(size=d, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=True),
        ])
    
    def _train_tfms(self) -> T.Compose:
        d = self.cfg.img_size
        return T.Compose([
            T.RandomResizedCrop(size=d, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=True),
        ])

    def _eval_tfms(self) -> T.Compose:
        d = self.cfg.img_size
        return T.Compose([
            T.Resize(int(d / 0.875), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(d),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def build(self):
        meta = WildlifeReID10k(self.cfg.root)
        df = meta.df
        splitter = splits.ClosedSetSplit(ratio_train=0.8, col_label="identity")
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
            emb_dim = backbone.width - 1
            backbone = nn.Sequential(backbone, LorentzEmbToTangent(manifold))

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
            emb_dim = 1025
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


# ------------------------------------------------------------------------------------------------
# Core train/eval loop
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def _eval_knn(backbone: nn.Module, ref_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    return evaluate_knn1(backbone, ref_loader, val_loader, device=device)


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: Optional[amp.GradScaler] = None,
    accumulation_steps: int = 1,
    device: str = "cuda",
    use_amp: bool = False,
    paradigm: TrainingParadigm | None = None,
):
    model.train(); head.train()
    steps_per_epoch = len(loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(loader), total=steps_per_epoch, leave=False)

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
        pbar.set_description(f"epoch {epoch} | loss {float(running_loss)/(i+1):.4f}")
    return float(running_loss) / steps_per_epoch


def fit(
    cfg: Config,
    backbone: nn.Module,
    objective: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    ref_eval_set: WR10kDataset,
    device: torch.device,
    logger: WandbSession,
    emb_dim: int,
    manifold=None,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_loss = float('inf')
    best_epoch = -1
    no_improve = 0

    paradigm = TrainingParadigm.REGISTRY[cfg.loss_type]
    scaler = amp.GradScaler('cuda', enabled=cfg.use_amp)

    for epoch in trange(1, cfg.epochs + 1, desc="Epochs"):
        train_loss = train_one_epoch(
            backbone, objective, train_loader, optimizer, epoch,
            scaler=scaler, accumulation_steps=cfg.accumulation_steps, device=str(device),
            use_amp=cfg.use_amp, paradigm=paradigm
        )
        scheduler.step(epoch + 1)

        # Periodic validation (here we monitor train loss; retrieval eval is expensive)
        if epoch % cfg.val_interval == 0 or epoch == cfg.epochs:
            improved = train_loss < best_loss
            if improved:
                best_loss = train_loss; best_epoch = epoch; no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model': backbone.state_dict(),
                    'head': objective.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'seed': cfg.seed,
                    'best_loss': best_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }, os.path.join(cfg.save_dir, f'{cfg.run_name}_best_epoch{epoch}.pt'))
                print(f"[VAL] New best @ epoch {epoch}: loss={best_loss:.4f} (saved)")
            else:
                no_improve += 1
                print(f"[VAL] train_loss={train_loss:.4f} (best={best_loss:.4f} @ {best_epoch}) | patience {no_improve}/{cfg.patience}")

            logger.log({
                "val/train_loss": train_loss,
                "val/best_loss": best_loss,
                "epoch": epoch,
                "train_loss": float(train_loss),
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

    # Evaluation (optionally convert to tangent for hyp)
    eval_backbone = backbone
    if cfg.hyperbolic and cfg.loss_type == 'triplet' and manifold is not None:
        eval_backbone = nn.Sequential(backbone, LorentzEmbToTangent(manifold))
        import warnings; warnings.warn("USING TANGENT HEAD FOR EVALUATION")

    overall, per_dataset_acc = _eval_knn(eval_backbone, ref_eval_set, val_loader, device)
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


# ------------------------------------------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------------------------------------------

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
    )
    return cfg


def main():
    cfg = _parse_args()
    set_seed(cfg.seed)
    # Open W&B first so sweep values are available
    with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        live = wb.cfg  # this is wandb.config (sweep values if present)
        # allow-list of keys we accept from sweeps
        for k in [
            "lr","wd","epochs","train_batch","eval_batch","img_size",
            "accumulation_steps","use_amp","val_interval","patience","optimizer_name"
        ]:
            if k in live:
                setattr(cfg, k, type(getattr(cfg, k))(live[k]))

        # Build everything from the (possibly) overridden cfg
        data = DataBuilder(cfg)
        train_set, ref_eval_set, val_set, train_loader, ref_eval_loader, val_loader = data.build()

        mb = ModelBuilder(cfg)
        backbone, emb_dim, manifold = mb.build()
        num_classes = train_set.num_classes
        objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold)

        ob = OptimBuilder(cfg)
        optimizer, scheduler = ob.build(backbone, objective, manifold)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone = backbone.to(device); objective = objective.to(device)

        # Train
        fit(cfg, backbone, objective, train_loader, val_loader, optimizer, scheduler,
            ref_eval_loader, device, wb, emb_dim, manifold)
if __name__ == '__main__':
    main()