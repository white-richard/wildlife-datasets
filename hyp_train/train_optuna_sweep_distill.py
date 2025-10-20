"""
Starts a PostgreSQL server for Optuna with:
docker container stop optuna-pg
docker container rm optuna-pg

docker run -d --name optuna-pg \
  -e POSTGRES_USER=optuna \
  -e POSTGRES_PASSWORD=optuna-pg \
  -e POSTGRES_DB=wr10k \
  -v optuna_pgdata:/var/lib/postgresql/data \
  -p 100.121.43.41:5432:5432 \
  postgres:15

python train_optuna_sweep_distill.py \
  --tune-trials -1 \
  --tune-direction maximize \
  --tune-storage postgresql+psycopg2://optuna:optuna-pg@100.121.43.41:5432/wr10k \
  --tune-study wr10k_megadesc_lastLayer \
  --wandb online --project reproduce_mega_descriptor \
  --tune-seed 42
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import timm
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as T
import wandb as _wandb
from geoopt import ManifoldParameter
from hypercore.manifolds.lorentzian import Lorentz
from hypercore.models.Swin_LViT import LSwin_base, LSwin_small, LSwin_tiny
from hypercore.modules.loss import LorentzTripletLoss
from hypercore.optimizers import RiemannianAdam, RiemannianSGD
import hypercore.nn as hnn
from hypercore.utils.manifold_utils import lock_curvature_to_one
from timm.scheduler import CosineLRScheduler
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from wildlife_tools.train.objective import ArcFaceLoss

from utils.augmentations import AugCfg, build_train_tfms
from utils.eucTohyp_Swin import replace_stages_with_hyperbolic
from utils.hyp_knn_per_class import evaluate_knn1 as hyp_evaluate_knn1
from utils.knn_per_class import evaluate_knn1
from utils.math_utils import kl_rows, pairwise_cosine, softmax_rows
from utils.reid_split_wr10k import build_reid_pipeline
from utils.validation import validate_split
from utils.wandb_session import WandbMode, WandbSession
from utils.load_checkpoint import load_checkpoint

torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    run_name: str = "wr10k_megadesc_lastLayerHyp"
    seed: int = 42
    save_dir: str = "checkpoints"

    # data
    # root: str = "../../../datasets/wildlifereid-10k"
    root: str = "/home/richw/.code/datasets/CzechLynx"

    img_size: int = 224
    num_workers: int = 8
    train_batch: int = 96
    eval_batch: int = 128
    aug_policy: str = "randaug"  # baseline | weak | strong | randaug | augmix
    data_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    data_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # model / paradigm
    model_name: str = "wr10k_megadesc_lastLayerHyp" # megadescriptor_replace_last_layer_hyp  # choices below
    teacher_name: str = "megadescriptor"  # choices below
    loss_type: str = "triplet"  # arcface | triplet
    use_xbm: bool = False  # for triplet loss
    type_of_triplets: str = "semihard"  # all | hard | semihard | easy | None
    hyperbolic: bool = True

    # --- KD / Distillation ---
    kd_enable: bool = False  # turn KD on
    kd_weight_pkt: float = 1.0  # λ for PKT
    kd_Te: float = 0.1  # teacher temperature
    kd_Th: float = 1.0  # student temperature
    kd_symmetrize: bool = False  # use 0.5*(KL(p||q)+KL(q||p))

    # keep room for future RKD/listwise
    kd_weight_rkd: float = 0.0  # not used yet (stub)
    kd_weight_list: float = 0.0  # not used yet (stub)

    # optimization
    epochs: int = 60
    lr: float = 0.0000957148605087535
    wd: float = 0.0010688021642081323 #1e-4
    momentum: float = 0.9
    warmup_t: int = 5
    lr_min: float = 1e-6
    optimizer_name: str = "sgd"  # sgd | adam

    tune_trials: int = 0  # 0 = no tuning; >0 runs optuna
    tune_direction: str = "maximize"  # "maximize" mAP
    tune_storage: Optional[str] = None  # e.g. "sqlite:///wr10k_optuna.db"
    tune_study: Optional[str] = None  # study name if you want persistence
    tune_pruner: str = "hyperband"  # median | sha | hyperband | none
    tune_sampler: str = "tpe"  # currently only tpe wired
    tune_seed: int = 42

    # training loop
    accumulation_steps: int = 1
    use_amp: bool = True
    val_interval: int = 1
    patience: int = 15
    use_scheduler: bool = True
    print("Using scheduler:", use_scheduler)

    # logging
    wandb_mode: WandbMode = WandbMode.OFF
    project: str = "reproduce_mega_descriptor"


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
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 30, 100, step=10),
        # "use_xbm": trial.suggest_categorical("use_xbm", [True, False]),
        # "train_batch": trial.suggest_categorical("train_batch", [64, 96, 128, 160]),
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["sgd", "adam"]),
        # "type_of_triplets": trial.suggest_categorical("type_of_triplets", ["semihard", "all", "hard"]),
    }
    return params


class DataBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.aug_cfg = AugCfg(self.cfg.aug_policy)

    def _train_tfms(self) -> T.Compose:
        return build_train_tfms(self.cfg.img_size, self.aug_cfg, self.cfg.data_mean, self.cfg.data_std)

    def _eval_tfms(self) -> T.Compose:
        d = self.cfg.img_size
        return T.Compose(
            [
                T.Resize(int(d / 0.875), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(d),
                T.ToTensor(),
                T.Normalize(self.cfg.data_mean, self.cfg.data_std),
            ]
        )

    def build(self):
        splits_out = build_reid_pipeline(
            self.cfg,
            train_transform=self._train_tfms(),
            val_transform=self._eval_tfms(),
            col_label="identity",
            frac_train_ids=0.8,  # 80% IDs used for training, 20% reserved for test
            ratio_fit=0.9,  # within-training images: 90% for fitting, 10% for dev
            ratio_gallery_dev=0.5,  # dev gallery/query ratio
            ratio_gallery_test=0.5,  # test gallery/query ratio
        )
        train_fit_loader = splits_out.train_fit_loader  # batches for optimizing weights
        val_ref_loader = splits_out.dev_ref_loader  # eval: gallery (val)
        val_qry_loader = splits_out.dev_qry_loader  # eval: query  (val)
        test_ref_loader = splits_out.test_ref_loader  # eval: gallery (test)
        test_qry_loader = splits_out.test_qry_loader  # eval: query  (test)
        train_fit_set = splits_out.train_fit_set

        loaders = {
            "train": train_fit_loader,
            "val_ref": val_ref_loader,
            "val_qry": val_qry_loader,
            "test_ref": test_ref_loader,
            "test_qry": test_qry_loader,
        }

        dev_ref_set = splits_out.dev_ref_set
        dev_qry_set = splits_out.dev_qry_set
        test_ref_set = splits_out.test_ref_set
        test_qry_set = splits_out.test_qry_set

        datasets = {
            "train": train_fit_set,
            "val_ref": dev_ref_set,
            "val_qry": dev_qry_set,
            "test_ref": test_ref_set,
            "test_qry": test_qry_set,
        }

        return loaders, datasets


class ModelBuilder:
    def __init__(
        self,
        cfg: Config,
        model_name: str,
        freeze_backbone: bool = False,
        ViT_size: str = "base",
        pretrained: bool = False,
    ):
        self.cfg = cfg
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.ViT_size = ViT_size  # 'base', 'small', etc...
        self.pretrained = pretrained

    def build(self):
        manifold = None
        emb_dim = None
        def reinit_module(module):
            for name, p in module.named_parameters():
                if p.requires_grad:
                    if p.dim() > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        torch.nn.init.zeros_(p)

        if self.model_name == "hyp_swin":
            if self.pretrained:
                raise ValueError("hyp_swin does not support pretrained=True")
            manifold = Lorentz()
            lock_curvature_to_one(manifold)
            if self.ViT_size == "base":
                backbone = LSwin_base(manifold, manifold, manifold, num_classes=0, embed_dim=48 + 1)
            elif self.ViT_size == "small":
                backbone = LSwin_small(manifold, manifold, manifold, num_classes=0, embed_dim=32 + 1)
            elif self.ViT_size == "tiny":
                backbone = LSwin_small(manifold, manifold, manifold, num_classes=0, embed_dim=32 + 1)
            else:
                raise ValueError(f"Unsupported ViT_size {self.ViT_size} for hyperbolic swin")
            for p in backbone.parameters():
                if isinstance(p, ManifoldParameter):
                    p.requires_grad_(False)
            backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)
                out = backbone(dummy)
            backbone.train()
            emb_dim = out.shape[1]
            manifold = backbone.manifold

        elif self.model_name == "megadescriptor":
            ViT_size = self.ViT_size[0].upper()
            backbone = timm.create_model(
                f"hf-hub:BVRA/MegaDescriptor-{ViT_size}-224", num_classes=0, pretrained=self.pretrained
            )
            emb_dim = backbone.num_features
            if not self.pretrained:
                backbone.apply(init_weights_xavier)
        
        elif self.model_name == "megadescriptor_lastLayer":
            
            ViT_size = self.ViT_size[0].upper()
            backbone = timm.create_model(
                f"hf-hub:BVRA/MegaDescriptor-{ViT_size}-224", num_classes=0, pretrained=True
            )
            emb_dim = backbone.num_features
            reinit_module(backbone.layers[-1])
            for _, p in backbone.named_parameters():
                p.requires_grad = False
            # unfreeze last layer
            for p in backbone.layers[-1].parameters():
                p.requires_grad = True
           

        elif self.model_name == "megadescriptor_replace_last_layer_hyp":
            if self.ViT_size != "base":
                raise ValueError("megadescriptor_replace_last_layer_hyp only supports ViT_size='base'")
            # if not self.pretrained:
            #     raise ValueError("megadescriptor_replace_last_layer_hyp only supports pretrained=True")
            backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-B-224", num_classes=0, pretrained=True)
            backbone, hyp_stages = replace_stages_with_hyperbolic(backbone, num_stages=1)
            if hasattr(backbone, "norm"):
                backbone.norm = nn.Identity()
            if hasattr(backbone, "global_pool"):
                backbone.global_pool = ""
            backbone.reset_classifier(num_classes=0, global_pool="")
            # freeze original weights & unfreeze inserted hyp blocks
            for _, p in backbone.named_parameters():
                p.requires_grad = False
            for hyp_stage in hyp_stages:
                for p in hyp_stage.parameters():
                    p.requires_grad = True
            backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.cfg.img_size, self.cfg.img_size)
                out = backbone(dummy)
            backbone.train()
            emb_dim = out.shape[1]
            manifold = hyp_stages[-1].manifold
        else:
            raise ValueError(f"Unknown model_name {self.model_name}")

        if self.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        return backbone, emb_dim, manifold


class ObjectiveBuilder:
    @staticmethod
    def build(loss_type: str, num_classes: int, emb_dim: int, hyperbolic: bool, manifold=None, use_xbm: bool = False, type_of_triplets: str = "semihard") -> nn.Module:
        if loss_type == "arcface":
            return ArcFaceLoss(num_classes=num_classes, embedding_size=emb_dim, margin=0.5, scale=64)
        if loss_type == "triplet":
            if hyperbolic:
                return LorentzTripletLoss(manifold, margin=0.2, type_of_triplets=type_of_triplets, feature_dim=emb_dim, use_xbm=use_xbm)
            return nn.TripletMarginLoss(margin=0.2, p=2)
        raise ValueError(f"Unknown loss_type {loss_type}")


class OptimBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.use_scheduler = cfg.use_scheduler

    def build(self, backbone: nn.Module, head: nn.Module, manifold=None):
        wd = self.cfg.wd
        decay, no_decay = split_decay_groups(backbone)
        head_params = [p for p in head.parameters() if p.requires_grad]

        if self.cfg.hyperbolic:
            if self.cfg.optimizer_name == "adam":
                optimizer = RiemannianAdam(
                    [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.cfg.lr,
                    stabilize=1,
                )
                if self.cfg.loss_type != "triplet":
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
                if manifold is not None:
                    remove_curvature_from_optimizer(optimizer, manifold)
            else:
                optimizer = RiemannianSGD(
                    [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.cfg.lr,
                    momentum=self.cfg.momentum,
                    stabilize=1,
                )
                if self.cfg.loss_type != "triplet":
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
                if manifold is not None:
                    remove_curvature_from_optimizer(optimizer, manifold)
        else:
            if self.cfg.optimizer_name == "adam":
                optimizer = torch.optim.Adam(
                    [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.cfg.lr,
                )
                if self.cfg.loss_type != "triplet":
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})
            else:
                optimizer = torch.optim.SGD(
                    [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                    lr=self.cfg.lr,
                    momentum=self.cfg.momentum,
                )
                if self.cfg.loss_type != "triplet":
                    optimizer.add_param_group({"params": head_params, "weight_decay": 0.0})

        # Cosine schedule in epochs
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
        
        scheduler = None
        if self.use_scheduler:
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
    REGISTRY: Dict[str, "TrainingParadigm"] = {}

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
def _eval_knn(
    backbone: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    hyperbolic: bool = False,
    manifold=None,
) -> Tuple[float, Dict[str, float]]:
    ref_loader = loaders["test_ref"]
    qry_loader = loaders["test_qry"]
    if hyperbolic:
        return hyp_evaluate_knn1(backbone, ref_loader, qry_loader, device=device, manifold=manifold)
    return evaluate_knn1(backbone, ref_loader, qry_loader, device=device)


class TripletKDParadigm(TrainingParadigm):
    def __init__(self, head_triplet: nn.Module, teacher: nn.Module, kd_cfg: Config):
        super().__init__()
        self.head_triplet = head_triplet
        self.teacher = teacher
        self.kd = kd_cfg

    def criterion_forward(self, head_unused, feats_student, y_labels):
        # base Lorentz triplet on student embeddings
        loss_triplet = self.head_triplet(feats_student, y_labels)

        if not self.kd.kd_enable or self.kd.kd_weight_pkt <= 0:
            return loss_triplet

        with torch.no_grad():
            # teacher Euclidean features for PKT
            z_teacher = self.teacher._modules.get("global_pool")(feats_student) if False else None  # placeholder
        # ^ the line above is intentionally dead; we don't have a ref to images here.

        # We need the input images to compute teacher features; we'll override post_forward to stash them.
        raise RuntimeError("TripletKDParadigm requires images to compute teacher features. Use post_forward hook.")


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: Optional[amp.GradScaler] = None,
    accumulation_steps: int = 1,
    device: str = "cuda",
    use_amp: bool = False,
    paradigm: TrainingParadigm | None = None,
    hyperbolic: bool = False,
    teacher: nn.Module | None = None,
    cfg: Config | None = None,
    manifold=None,
):
    train_loader = loaders["train"]
    model.train()
    head.train()
    if teacher is not None:
        teacher.eval()

    steps_per_epoch = len(train_loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, leave=False)

    if _wandb.run is not None:
        print("Run name:", _wandb.run.name)

    validate_every = max(1, steps_per_epoch - 1)

    running_metric = 0.0
    metric_count = 0

    wandb_step_name = 'train_loop/iter'

    if paradigm is None:
        raise ValueError("Training paradigm must be provided")

    for i, batch in pbar:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            feats = model(x)  # student hyperbolic embeddings (Lorentz coords)
            # if i < 1000:
            #     with torch.no_grad():
            #         _ = head(feats.detach(), y)  # just fill memory
            #     continue

            base_loss = paradigm.criterion_forward(head, feats, y) / accumulation_steps
            loss = base_loss

            if cfg is not None and cfg.kd_enable and cfg.kd_weight_pkt > 0.0:
                if teacher is None:
                    raise RuntimeError("KD enabled but teacher is None.")

                # Teacher forward (Euclidean), L2-normalized for cosine
                with torch.no_grad():
                    z = teacher(x)  # [B, d_euc]
                    S = pairwise_cosine(z)  # [B, B]
                    P = softmax_rows(S, cfg.kd_Te)  # teacher neighbor dists

                # Student geodesic distances (Lorentz)
                D = manifold.pairwise_distance(feats)  # [B, B]
                A = -(D ** 2)  # attraction scores
                Q = softmax_rows(A, cfg.kd_Th)  # student neighbor dists

                loss_pkt = kl_rows(P, Q)
                if cfg.kd_symmetrize:
                    loss_pkt = 0.5 * (loss_pkt + kl_rows(Q, P))

                loss = loss + (cfg.kd_weight_pkt / accumulation_steps) * loss_pkt

        feats = paradigm.post_forward(feats, use_amp)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.detach()
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
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
                mAP = metrics["mAP"]
                acc1 = metrics["top1"]
                acc5 = metrics["top5"]
                acc10 = metrics["top10"]
                running_metric += mAP
                metric_count += 1
                if _wandb.run is not None:
                    _wandb.define_metric(step_metric = wandb_step_name, name = "train_loop/val_mAP")
                    _wandb.define_metric(step_metric = wandb_step_name, name = "train_loop/val_top1")
                    _wandb.define_metric(step_metric = wandb_step_name, name = "train_loop/val_top5")
                    _wandb.define_metric(step_metric = wandb_step_name, name = "train_loop/val_top10") 
                    # _wandb.define_metric(step_metric = wandb_step, name = "train_loop/iter")
                    _wandb.log(
                        {
                            "train_loop/val_mAP": mAP,
                            "train_loop/val_top1": acc1,
                            "train_loop/val_top5": acc5,
                            "train_loop/val_top10": acc10,
                            "train_loop/iter": epoch * steps_per_epoch + i + 1,
                        },
                    )
                else:
                    print(
                        f"epoch {epoch} | iter {i+1}/{steps_per_epoch} "
                        f"| loss {float(running_loss)/(i+1):.4f} | val_mAP {mAP:.4f} | val_top1 {acc1:.4f} | val_top5 {acc5:.4f} | val_top10 {acc10:.4f}"
                    )

        pbar.set_description(f"epoch {epoch} | loss {float(running_loss)/(i+1):.4f}")
        if _wandb.run is not None and cfg is not None and cfg.kd_enable and cfg.kd_weight_pkt > 0.0:
            _wandb.define_metric(step_metric = wandb_step_name, name = "train_loop/loss_pkt")

            _wandb.log(
                {
                    "train_loop/loss_pkt": float(loss_pkt.detach().cpu()),
                    "train_loop/iter": epoch * steps_per_epoch + i + 1,
                },
            )

    avg_acc = float(running_metric) / metric_count if metric_count > 0 else float("nan")
    return float(running_loss) / steps_per_epoch, avg_acc


def fit(
    cfg: Config,
    backbone: nn.Module,
    objective: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    device: torch.device,
    logger: WandbSession,
    manifold=None,
    trial: "optuna.trial.Trial | None" = None,
    teacher_backbone: nn.Module | None = None,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_loss = float("inf")
    best_mAP = -float("inf")
    best_epoch = -1
    no_improve = 0

    if cfg.hyperbolic and cfg.loss_type != "triplet":
        raise ValueError("Hyperbolic training is only supported with triplet loss currently.")

    paradigm = TrainingParadigm.REGISTRY[cfg.loss_type]
    scaler = amp.GradScaler(enabled=cfg.use_amp)

    global_step = 0

    for epoch in trange(1, cfg.epochs + 1, desc="Epochs"):
        train_loss, train_mAP = train_one_epoch(
            backbone,
            objective,
            loaders,
            optimizer,
            epoch,
            scaler=scaler,
            accumulation_steps=cfg.accumulation_steps,
            device=str(device),
            use_amp=cfg.use_amp,
            paradigm=paradigm,
            hyperbolic=cfg.hyperbolic,
            teacher=teacher_backbone,
            cfg=cfg,
            manifold=manifold,
        )
        if scheduler is not None:
            scheduler.step(epoch + 1)

        if train_mAP > best_mAP:
            best_mAP = train_mAP
            best_epoch = epoch
            no_improve = 0
            save_path=os.path.join(cfg.save_dir, f"{_wandb.run.name}_best_epoch{epoch}.pt")
            print(f"Saving best checkpoint to {save_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model": backbone.state_dict(),
                    "head": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "seed": cfg.seed,
                    "best_mAP": best_mAP,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                save_path,
            )
            print(f"[VAL] New best @ epoch {epoch}: val_mAP={best_mAP:.4f} (saved)")
        else:
            no_improve += 1
            print(
                f"[VAL] train_loss={train_loss:.4f} (best={best_mAP:.4f} @ {best_epoch}) | patience {no_improve}/{cfg.patience}"
            )

        if _wandb.run is not None:
            wandb_step_name = 'train/epoch'
            _wandb.define_metric(step_metric = wandb_step_name, name = "train/train_loss")
            _wandb.define_metric(step_metric = wandb_step_name, name = "train/_epoch_mAP")
            _wandb.log(
                {
                    "train/epoch": epoch,
                    "train/train_loss": float(train_loss),
                    "train/_epoch_mAP": float(train_mAP),
                },
            )
        if trial is not None:
            trial.report(train_mAP, step=epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch}.")
                if _wandb.run is not None:
                    _wandb.summary["pruned_at_epoch"] = epoch
                raise optuna.exceptions.TrialPruned()

        logger.log(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val/mAP": float(train_mAP),
            }
        )
        if no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    save_path=os.path.join(cfg.save_dir, f"{_wandb.run.name}_final.pt")
    print(f"Saving final checkpoint to {save_path}")
    torch.save(
        {
            "epoch": epoch,
            "model": backbone.state_dict(),
            "head": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "seed": cfg.seed,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        },
        save_path,
    )

    print(f"\nTraining done. Best loss={best_loss:.4f} at epoch {best_epoch}. Final checkpoint saved.")

    overall, per_dataset_acc = _eval_knn(backbone, loaders, device, hyperbolic=cfg.hyperbolic, manifold=manifold)
    logger.log({"metrics/overall_acc": overall})
    if logger.active and _wandb is not None:
        table = _wandb.Table(columns=["dataset", "accuracy"])
        for ds, acc in sorted(per_dataset_acc.items()):
            table.add_data(ds, acc)
            _wandb.summary[f"acc_by_dataset/{ds}"] = acc
        logger.log({"per_dataset_accuracy_table": table})
        logger.log(
            {
                "plots/per_dataset_accuracy": _wandb.plot.bar(
                    table, "dataset", "accuracy", title="Per-dataset Top-1 Accuracy"
                )
            }
        )
    else:
        print(f"Overall acc: {overall}.")

    return best_mAP


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Modular WR10k trainer")
    p.add_argument("--run-name", type=str, default=Config.run_name)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--save-dir", type=str, default=Config.save_dir)
    p.add_argument("--root", type=str, default=Config.root)
    p.add_argument("--img-size", type=int, default=Config.img_size)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--train-batch", type=int, default=Config.train_batch)
    p.add_argument("--eval-batch", type=int, default=Config.eval_batch)

    p.add_argument("--model-name", type=str, default=Config.model_name)
    p.add_argument("--loss-type", type=str, default=Config.loss_type, choices=list(TrainingParadigm.REGISTRY.keys()))
    p.add_argument("--hyperbolic", action="store_true", default=Config.hyperbolic)

    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--wd", type=float, default=Config.wd)
    p.add_argument("--momentum", type=float, default=Config.momentum)
    p.add_argument("--warmup-t", type=int, default=Config.warmup_t)
    p.add_argument("--lr-min", type=float, default=Config.lr_min)
    p.add_argument("--optimizer-name", type=str, default=Config.optimizer_name, choices=["sgd", "adam"])
    p.add_argument("--use-scheduler", action="store_true", default=Config.use_scheduler)

    p.add_argument("--accumulation-steps", type=int, default=Config.accumulation_steps)
    p.add_argument("--use-amp", action="store_true", default=Config.use_amp)
    p.add_argument("--val-interval", type=int, default=Config.val_interval)
    p.add_argument("--patience", type=int, default=Config.patience)
    p.add_argument("--use-xbm", action="store_true", default=Config.use_xbm)
    p.add_argument("--type-of-triplets", type=str, default=Config.type_of_triplets, choices=["semihard", "all", "hard"])

    p.add_argument("--wandb", type=str, default=Config.wandb_mode.value, choices=[m.value for m in WandbMode])
    p.add_argument("--project", type=str, default=Config.project)

    p.add_argument("--tune-trials", type=int, default=Config.tune_trials)
    p.add_argument("--tune-direction", type=str, default=Config.tune_direction, choices=["minimize", "maximize"])
    p.add_argument("--tune-storage", type=str, default=Config.tune_storage)
    p.add_argument("--tune-study", type=str, default=Config.tune_study)
    p.add_argument("--tune-pruner", type=str, default=Config.tune_pruner, choices=["median", "sha", "hyperband", "none"])
    p.add_argument("--tune-sampler", type=str, default=Config.tune_sampler, choices=["tpe"])
    p.add_argument("--tune-seed", type=int, default=Config.tune_seed)

    p.add_argument("--kd-enable", action="store_true", default=Config.kd_enable)
    p.add_argument("--kd-weight-pkt", type=float, default=Config.kd_weight_pkt)
    p.add_argument("--kd-Te", type=float, default=Config.kd_Te)
    p.add_argument("--kd-Th", type=float, default=Config.kd_Th)
    p.add_argument("--kd-symmetrize", action="store_true", default=Config.kd_symmetrize)
    # optional stubs for later:
    p.add_argument("--kd-weight-rkd", type=float, default=Config.kd_weight_rkd)
    p.add_argument("--kd-weight-list", type=float, default=Config.kd_weight_list)

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
        tune_seed=args.tune_seed,
        kd_enable=args.kd_enable,
        kd_weight_pkt=args.kd_weight_pkt,
        kd_Te=args.kd_Te,
        kd_Th=args.kd_Th,
        kd_symmetrize=args.kd_symmetrize,
        kd_weight_rkd=args.kd_weight_rkd,
        kd_weight_list=args.kd_weight_list,
        use_scheduler=args.use_scheduler,
        use_xbm=args.use_xbm,
        type_of_triplets=args.type_of_triplets,
    )
    return cfg


def _run_one_trial(trial: "optuna.trial.Trial", base_cfg: Config):
    cfg = copy.deepcopy(base_cfg)

    suggested = suggest_params(trial, cfg)

    def _coerce(val, target):
        return bool(val) if isinstance(target, bool) else type(target)(val)

    for k, v in suggested.items():
        if hasattr(cfg, k):
            setattr(cfg, k, _coerce(v, getattr(cfg, k)))

    cfg.run_name = f"{base_cfg.run_name}-t{trial.number}"

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        data = DataBuilder(cfg)
        loaders, datasets = data.build()

        if _wandb.run is not None:
            _wandb.config.update(
                {
                    "optuna/study": cfg.tune_study,
                    "optuna/trial_number": trial.number,
                    "optuna/seed_effective": cfg.tune_seed,
                },
                allow_val_change=True,
            )
        _wandb.summary["effective_config"] = {k: getattr(cfg, k) for k in suggest_params(trial, cfg).keys()}

        teacher_backbone = None
        manifold=None
        if cfg.kd_enable:
            tb = ModelBuilder(cfg, cfg.teacher_name, pretrained=True, ViT_size="base", freeze_backbone=True)
            teacher_backbone, _, _ = tb.build()
            teacher_backbone = teacher_backbone.to(device)
            teacher_backbone.eval()

        mb = ModelBuilder(cfg, cfg.model_name, pretrained=False, ViT_size="base")
        backbone, emb_dim, manifold = mb.build()
        num_classes = datasets["train"].num_classes
        objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold, use_xbm=cfg.use_xbm, type_of_triplets=cfg.type_of_triplets)

        ob = OptimBuilder(cfg)
        optimizer, scheduler = ob.build(backbone, objective, manifold)

        backbone = backbone.to(device)
        objective = objective.to(device)


        try:
            best_mAP = fit(
                cfg,
                backbone,
                objective,
                loaders,
                optimizer,
                scheduler,
                device,
                wb,
                manifold,
                trial=trial,
                teacher_backbone=teacher_backbone,
            )
        except optuna.exceptions.TrialPruned:
            raise
        return best_mAP


def run_optuna(cfg: Config):
    if optuna is None:
        raise RuntimeError("optuna is not installed. pip install optuna")

    if cfg.tune_sampler == "tpe":
        sampler = TPESampler(seed=cfg.tune_seed)
    else:
        sampler = TPESampler(seed=cfg.tune_seed)

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

    n_trials = None if cfg.tune_trials < 0 else cfg.tune_trials
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna] Best value={study.best_value:.6f}")
    print(f"[Optuna] Best params={study.best_trial.params}")
    return study


def main():
    cfg = _parse_args()
    if cfg.tune_trials != 0:
        print(f"Running Optuna sweep with {cfg.tune_trials} trials.")
        run_optuna(cfg)
        return

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        data = DataBuilder(cfg)
        loaders, datasets = data.build()

        teacher_backbone = None
        manifold=None
        if cfg.kd_enable:
            tb = ModelBuilder(cfg, cfg.teacher_name, pretrained=True, ViT_size="base", freeze_backbone=True)
            teacher_backbone, _, _ = tb.build()
            teacher_backbone = teacher_backbone.to(device)
            teacher_backbone.eval()

        mb = ModelBuilder(cfg, cfg.model_name, pretrained=False, ViT_size="base")
        backbone, emb_dim, manifold = mb.build()
        num_classes = datasets["train"].num_classes
        objective = ObjectiveBuilder.build(cfg.loss_type, num_classes, emb_dim, cfg.hyperbolic, manifold, use_xbm=cfg.use_xbm, type_of_triplets=cfg.type_of_triplets)

        ob = OptimBuilder(cfg)
        optimizer, scheduler = ob.build(backbone, objective, manifold)

        backbone = backbone.to(device)
        objective = objective.to(device)

        # with WandbSession(cfg.wandb_mode, cfg.project, cfg.run_name, asdict(cfg)) as wb:
        #     overall, per_dataset_acc = _eval_knn(backbone, loaders, device, hyperbolic=cfg.hyperbolic, manifold=manifold)
        #     logger = wb
        #     logger.log({"metrics/overall_acc": overall})
        #     if logger.active and _wandb is not None:
        #         table = _wandb.Table(columns=["dataset", "accuracy"])
        #         for ds, acc in sorted(per_dataset_acc.items()):
        #             table.add_data(ds, acc)
        #             _wandb.summary[f"acc_by_dataset/{ds}"] = acc
        #         logger.log({"per_dataset_accuracy_table": table})
        #         logger.log(
        #             {
        #                 "plots/per_dataset_accuracy": _wandb.plot.bar(
        #                     table, "dataset", "accuracy", title="Per-dataset Top-1 Accuracy"
        #                 )
        #             }
        #         )
        # checkpoint_path = "/home/richw/.code/repos/wildlife-datasets/hyp_train/dandy-darkness-143_best_epoch73.pt"
        # load_checkpoint(checkpoint_path, backbone)
        # overall, per_dataset_acc = _eval_knn(backbone, loaders, device, hyperbolic=cfg.hyperbolic, manifold=manifold)
        # print(f"Overall acc: {overall}.")

        _ = fit(cfg, backbone, objective, loaders, optimizer, scheduler, device, wb, manifold, teacher_backbone=teacher_backbone)


if __name__ == "__main__":
    main()
