import os
import random
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
import torchvision.transforms as T
import timm
from tqdm import tqdm, trange
import wandb
import numpy as np
from timm.scheduler import CosineLRScheduler
import torch.nn.init as init

from wildlife_tools.train import ArcFaceLoss
from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets import splits
from wildlife_tools.data import FeatureDatabase
from wildlife_tools.inference import KnnMatcher

from geoopt import ManifoldParameter
import hypercore.nn as hnn
from hypercore.utils.manifold_utils import lock_curvature_to_one
from hypercore.models.Swin_LViT import LSwin_small
from hypercore.manifolds.lorentzian import Lorentz
from hypercore.optimizers import RiemannianSGD
from hypercore.modules.loss import LorentzTripletLoss

from eucTohyp_Swin import replace_stages_with_hyperbolic
from wr10k_dataset import WR10kDataset

torch.backends.cudnn.benchmark = True


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN / cuBLAS determinism
    torch.use_deterministic_algorithms(True, warn_only=True) # will raise if you hit a non-deterministic op
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For cuBLAS deterministic reductions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # or ":16:8"


@torch.no_grad()
def build_feature_database_for_refds(
    model,
    ref_ds,
    ref_loader: DataLoader,
    out_path: str = "ref_database.pkl",
    device: str = "cuda",
    l2_normalize: bool = True,
):
    """
    Writes a FeatureDatabase-compatible pickle with keys:
      - features:  float32 (N, D)
      - metadata:  pandas.DataFrame with column 'identity'
      - col_label: string (default 'identity')
      - load_label: bool
    """
    import pandas as pd
    model.eval().to(device)
    assert hasattr(ref_ds, "id2idx"), "ref_ds must have an 'id2idx' mapping"

    idx2id = {v: k for k, v in ref_ds.id2idx.items()}

    embs, ids = [], []
    for xb, yb, *_ in tqdm(ref_loader, total=len(ref_loader), desc="Building reference DB"):
        xb = xb.to(device, non_blocking=True)
        zb = model(xb)
        if l2_normalize:
            zb = torch.nn.functional.normalize(zb, dim=1)
        embs.append(zb.detach().cpu().to(torch.float32).numpy())
        ids.extend([idx2id[int(y)] for y in yb])

    features = np.concatenate(embs, axis=0).astype(np.float32)
    metadata = pd.DataFrame({"identity": np.array(ids, dtype=object)})

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "features": features,
                "metadata": metadata,
                "col_label": "identity",
                "load_label": True,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return out_path


@torch.inference_mode()
def evaluate_per_dataset_with_knnmatcher(
    model,
    ref_ds,
    qry_loader,
    database_file: str = "ref_database.pkl",
    device: str = "cuda",
    dataset_col: str = "dataset",
    force_rebuild: bool = False,
):
    """
    Evaluates 1-NN top-1 per-dataset accuracy using a FeatureDatabase loaded from .pkl.
    If database_file does not exist (or force_rebuild=True), it will be built first.
    """
    assert hasattr(ref_ds, 'id2idx'), "ref_ds must have an 'id2idx' mapping"
    assert hasattr(ref_ds, 'df'), "ref_ds must have a 'df' DataFrame"
    assert dataset_col in ref_ds.df.columns, (
        f"'{dataset_col}' not in df columns: {list(ref_ds.df.columns)}"
    )
    ref_loader = DataLoader(ref_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model.eval().to(device)

    idx2id = {v: k for k, v in ref_ds.id2idx.items()}  # int label -> str identity
    
    # Fix: Build id2dataset from both reference AND query datasets
    ref_id2dataset = (
        ref_ds.df.drop_duplicates('identity')
        .set_index('identity')[dataset_col]
        .to_dict()
    )
    
    # Get the query dataset's dataframe to build complete mapping
    qry_ds = qry_loader.dataset
    if hasattr(qry_ds, 'df'):
        qry_id2dataset = (
            qry_ds.df.drop_duplicates('identity')
            .set_index('identity')[dataset_col]
            .to_dict()
        )
        # Combine both mappings
        id2dataset = {**ref_id2dataset, **qry_id2dataset}
    else:
        id2dataset = ref_id2dataset

    if force_rebuild or not os.path.exists(database_file):
        build_feature_database_for_refds(
            model=model,
            ref_ds=ref_ds,
            ref_loader=ref_loader,
            out_path=database_file,
            device=device,
            l2_normalize=True,
        )

    database = FeatureDatabase.from_file(database_file)
    matcher = KnnMatcher(database, k=1)

    Q_batches = [] # collect (B, D) arrays
    true_ids = []
    for xb, yb, *_ in tqdm(qry_loader, total=len(qry_loader), desc="Embedding queries"):
        xb = xb.to(device, non_blocking=True)
        zb = torch.nn.functional.normalize(model(xb), dim=1)  # (B, D)
        Q_batches.append(zb.detach().cpu().to(torch.float32).numpy())  # (B, D)
        true_ids.extend([idx2id[int(y)] for y in yb])

    Q = np.vstack(Q_batches) # (N, D)
    pred_ids = matcher(Q)

    per_dataset_acc = defaultdict(list)
    for t_id, p_id in zip(true_ids, pred_ids):
        # Fix: Handle missing identities gracefully
        if t_id not in id2dataset:
            print(f"Warning: Identity '{t_id}' not found in dataset mapping, skipping...")
            continue
            
        ds_name = id2dataset[t_id]
        per_dataset_acc[ds_name].append(p_id == t_id)

    per_dataset_acc = {ds: float(np.mean(v)) for ds, v in per_dataset_acc.items()}
    overall = float(np.mean(list(per_dataset_acc.values()))) if per_dataset_acc else float('nan')
    return overall, per_dataset_acc


def train_one_epoch(
    model,
    head,
    loader,
    optimizer,
    epoch,
    scaler=None,
    accumulation_steps=1,
    device="cuda",
    use_amp=False,
):
    model.train()
    head.train()

    steps_per_epoch = len(loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    iteration = 0

    pbar = tqdm(enumerate(loader), total=steps_per_epoch, leave=False)

    for micro_step, batch in pbar:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            feats = model(x)
            loss = head(feats, y) / accumulation_steps
        
        if use_amp:
            feats = feats.float()
     
        scaler.scale(loss).backward()
        running_loss += loss.detach()

        if (micro_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iteration += 1
        pbar.set_description(f"epoch {epoch} | loss {running_loss/(micro_step+1):.4f}")
    return running_loss / steps_per_epoch


def train(
    epochs,
    backbone,
    objective,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    ref_eval_set,
    seed,
    use_wandb,
    use_amp,
    scaler=None,
    accumulation_steps=1,
    val_interval=5,  # Validate every 5 epochs instead of every step
    patience=3,      # Reduce patience since validation is less frequent  
    device='cuda',
    save_dir='checkpoints',
    run_name='wr10k_megadesc_arcface',
    embed_dim=128,
    num_classes=1000,
):
    os.makedirs(save_dir, exist_ok=True)

    # best_acc = -float('inf')
    best_loss = -float('inf')

    best_epoch = -1
    no_improve_evals = 0
    
    # Create validation subsets for faster evaluation during training
    # ref_subset_size = min(len(ref_eval_set), 2000)  # Limit reference set size
    # val_subset_size = min(len(val_loader.dataset), 1000)  # Limit query set size
    
    # ref_indices = torch.randperm(len(ref_eval_set))[:ref_subset_size]
    # val_indices = torch.randperm(len(val_loader.dataset))[:val_subset_size]
    
    # Fix: Create a new dataset instance instead of Subset to preserve attributes
    # ref_subset_df = ref_eval_set.df.iloc[ref_indices.numpy()].reset_index(drop=True)
    # val_subset_df = val_loader.dataset.df.iloc[val_indices.numpy()].reset_index(drop=True)
    
    # Create new dataset instances with the same transforms and id_list
    # ref_subset = WR10kDataset(ref_subset_df, ref_eval_set.root, ref_eval_set.transform, 
    #                          id_list=ref_eval_set.id_list)
    # val_subset = WR10kDataset(val_subset_df, val_loader.dataset.root, val_loader.dataset.transform,
                            #  id_list=val_loader.dataset.id_list)
    
    # val_subset_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=8)

    for epoch in trange(1, epochs + 1, desc="Epochs"):
        train_loss = train_one_epoch(
            backbone, objective, train_loader, optimizer, epoch,
            scaler=scaler, accumulation_steps=accumulation_steps, 
            device=device, use_amp=use_amp,
        )
        scheduler.step(epoch + 1)

        # Periodic validation using actual retrieval metric
        if epoch % val_interval == 0 or epoch == epochs:

           
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                no_improve_evals = 0
                torch.save({
                    'epoch': epoch,
                    'model': backbone.state_dict(),
                    'head': objective.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'seed': seed,
                    'best_loss': best_loss,
                }, os.path.join(save_dir, f'{run_name}_best_epoch{epoch}.pt'))
                print(f"[VAL] New best @ epoch {epoch}: loss={best_loss:.4f} (saved)")
            else:
                no_improve_evals += 1
                print(f"[VAL] train_loss={train_loss:.4f} (best={best_loss:.4f} @ {best_epoch}) "
                      f"| patience {no_improve_evals}/{patience}")

            if use_wandb:
                wandb.log({
                    "val/train_loss": train_loss,
                    "val/best_loss": best_loss,
                    "epoch": epoch,
                    "train_loss": float(train_loss)
                })

            if no_improve_evals >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    final_ckpt = {
        'epoch': epoch,
        'model': backbone.state_dict(),
        'head': objective.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'seed': seed,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }
    torch.save(final_ckpt, os.path.join(save_dir, f'{run_name}_final.pt'))
    print(f"\nTraining done. Best loss={best_loss:.4f} at epoch {best_epoch}. Final checkpoint saved.")


def split_decay_groups(model: nn.Module):
    """Return (decay, no_decay) parameter lists for a model.
    - no_decay: LayerNorm weights & biases, plus all biases
    - decay: everything else
    """
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

def remove_curvature_from_optimizer(optimizer, manifold):
    # Collect the actual Parameter objects we want to remove
    params_to_remove = []
    for attr in ("k", "c"):
        if hasattr(manifold, attr):
            val = getattr(manifold, attr)
            if isinstance(val, torch.nn.Parameter):
                params_to_remove.append(val)

    if not params_to_remove:
        return

    remove_ids = {id(p) for p in params_to_remove}

    # Remove from param groups (compare by identity via id())
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in remove_ids]

    # Also drop any optimizer state tied to those params
    for p in list(optimizer.state.keys()):
        if id(p) in remove_ids:
            print(p)
            optimizer.state.pop(p, None)
            print(f"Dropped {p} from optimizer to disable gradients.")

def init_weights_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight) # or xavier_normal_
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)


class LorentzEmbToTangent(nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        v = self.manifold.logmap0(x)
        v = v[..., 1:]
        return v


def debug_data_splits(train_set, val_set, ref_eval_set):
    """Debug the train/val split to ensure no data leakage"""
    train_ids = set(train_set.df['identity'].unique())
    val_ids = set(val_set.df['identity'].unique()) 
    ref_ids = set(ref_eval_set.df['identity'].unique())
    
    print(f"Train unique IDs: {len(train_ids)}")
    print(f"Val unique IDs: {len(val_ids)}")  
    print(f"Ref unique IDs: {len(ref_ids)}")
    print(f"Train/Val overlap: {len(train_ids & val_ids)}")
    print(f"Train/Ref overlap: {len(train_ids & ref_ids)}")
    
    # Check if same identity appears in both splits (should be 100% for closed set)
    if train_ids == val_ids == ref_ids:
        print("✓ Closed set split correct - all identities in train/val/ref")
    else:
        print("✗ Identity mismatch between splits!")
        

def main():
    use_wandb = True
    hyperbolic = True
    checkpoint_path = None
    P, K = 32, 4
    effective_batch = P * K
    accumulation_steps = 1
    #checkpoint_path = "checkpoints/MegaDescriptor-B-224_best_epoch65.pt"
    model_name = 'megadesc_replace_last_layer_hyperbolic'
    
    loss_type = 'triplet' # 'arcface' or 'triplet'
    run_name='HypMegaDescriptor-S-224__lastlayerHypTrip'
    if use_wandb:
        wandb.login()
        wandb.init(project="reproduce_mega_descriptor", name=run_name)
    seed = 42;set_seed(seed)
    dim = 224
    train_tfms = T.Compose([
         T.RandomResizedCrop(size=dim, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
         #T.Resize((dim,dim)),
         # T.RandomHorizontalFlip(p=0.5), # Breaks symmetry for some animals in some poses
         T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
         T.RandomGrayscale(p=0.10),
         #T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
         T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.2),
         T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
         T.ToTensor(),
         T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
         T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=True),
     ])
    #train_tfms = T.Compose([
    #    T.Resize((dim,dim)),
    #    T.ToTensor(),
    #    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    #])
    val_tfms = T.Compose([
        T.Resize(int(dim / 0.875), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(dim),
        #T.Resize((dim,dim)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


    root = "/home/richw/.richie/repos/wildlifereid-10k"
    meta = WildlifeReID10k(root)
    df = meta.df
    splitter = splits.ClosedSetSplit(ratio_train=0.8, col_label="identity") # keep IDs same in train/test
    idx_ref, idx_qry = splitter.split(df)[0]
    df_ref = df.iloc[idx_ref].copy()
    df_qry = df.iloc[idx_qry].copy()

    # ----- Ensure both splits contain the same identities -----
    train_identities = set(df_ref['identity'].unique())
    val_identities = set(df_qry['identity'].unique()) 
    print(f"Train identities: {len(train_identities)}")
    print(f"Val identities: {len(val_identities)}")
    print(f"Identity overlap: {len(train_identities & val_identities)}")
    # For ClosedSetSplit, these should be identical
    if train_identities != val_identities:
        print("WARNING: ClosedSetSplit should have identical identities in train/val!")
        print(f"Train only: {train_identities - val_identities}")
        print(f"Val only: {val_identities - train_identities}")
        # Fix by using only common identities
        common_identities = train_identities & val_identities
        df_ref = df_ref[df_ref['identity'].isin(common_identities)].copy()
        df_qry = df_qry[df_qry['identity'].isin(common_identities)].copy()
        print(f"Using {len(common_identities)} common identities")
    # Create datasets with consistent identity mapping
    train_set = WR10kDataset(df_ref, root, train_tfms)
    id_list = sorted(train_set.id2idx.keys())  # Get the canonical ordering
    # Both ref and val datasets use the same id_list
    ref_eval_set = WR10kDataset(df_ref, root, val_tfms, id_list=id_list)
    val_set = WR10kDataset(df_qry, root, val_tfms, id_list=id_list)
    debug_data_splits(train_set, val_set, ref_eval_set)

    train_classes = set(train_set.id2idx.values())
    val_classes   = set(val_set.id2idx.values())
    assert train_classes == val_classes, \
        f"Train/val class index mismatch: {train_classes ^ val_classes}"
    
    # train_labels = [train_set.id2idx[i] for i in train_set.df['identity']]
    print("Total number of images TRAIN:", len(train_set))
    print("Total number of images VAL:", len(val_set))
    # sampler = PKSampler(train_labels, P=P, K=K)
    # train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=128, num_workers=16, pin_memory=True, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)

    if model_name == 'hyp_swin':
        manifold = Lorentz()
        lock_curvature_to_one(manifold)
        backbone = LSwin_small(manifold, manifold, manifold, num_classes=0, embed_dim=32+1)
        for p in backbone.parameters():
            if isinstance(p, ManifoldParameter):
                p.requires_grad_(False)
        emb_dim = backbone.width - 1

        backbone = nn.Sequential(
            backbone,
            LorentzEmbToTangent(manifold)
        )
    elif model_name == 'megadescriptor_swin':
        backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=False)
        emb_dim = backbone.num_features
        backbone.apply(init_weights_xavier)
    elif model_name == 'megadescriptor_swin_pretrained':
        backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True)
        emb_dim = backbone.num_features
        # write backbone model to txt file
    elif model_name == 'megadesc_replace_last_layer_hyperbolic':
        backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True)
        backbone, hyp_stages = replace_stages_with_hyperbolic(backbone, num_stages=1)

        if hasattr(backbone, "norm"):  # LayerNorm(C_e) would break on C_e+1
            backbone.norm = nn.Identity()
        if hasattr(backbone, "global_pool"):  # ensure forward_features returns tokens
            backbone.global_pool = ""
        backbone.reset_classifier(num_classes=0, global_pool='')
        # freeze original weights
        for n, p in backbone.named_parameters():
            p.requires_grad = False
        # unfreeze the inserted hyperbolic block
        for hyp_stage in hyp_stages:
            for p in hyp_stage.parameters():
                p.requires_grad = True
        emb_dim = 1025
        manifold = hyp_stages[-1].manifold
    else:
        raise ValueError(f"Unknown model_name {model_name}")

    with open(f"{model_name}_model.txt", "w") as f:
        f.write(str(backbone))


    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        backbone.load_state_dict(ckpt["model"])
    

    num_classes = train_set.num_classes
    if loss_type == 'arcface':
        objective = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=emb_dim,
            margin=0.5, scale=64
        )
    elif loss_type == 'triplet':
        if hyperbolic:
            objective = LorentzTripletLoss(manifold, margin=0.2, type_of_triplets='hard', feature_dim=1025)
        else:
            raise
            objective = nn.TripletMarginLoss(margin=0.2, p=2)
    
    for name, param in objective.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    epochs = 40
    wd = 1e-4
    #backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    backbone_decay, backbone_no_decay = split_decay_groups(backbone)

    head_params = [p for p in objective.parameters() if p.requires_grad]
    print(f"Num classes: {train_set.num_classes}")
    print(f"Backbone: decay={sum(p.numel() for p in backbone_decay)}, "
      f"no_decay={sum(p.numel() for p in backbone_no_decay)}, "
      f"head={sum(p.numel() for p in head_params)}")
    if hyperbolic:
        optimizer = RiemannianSGD(
            [
                {"params": backbone_decay,     "weight_decay": wd},
                {"params": backbone_no_decay,  "weight_decay": 0.0},  
            ],
            lr=1e-3, momentum=0.9, stabilize=1
        )
        optimizer.add_param_group({"params": head_params, "weight_decay": 0.0}) if not loss_type == 'triplet' else None
        remove_curvature_from_optimizer(optimizer, manifold)
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": backbone_decay,     "weight_decay": wd},
                {"params": backbone_no_decay,  "weight_decay": 0.0},  # LayerNorms & biases
            ],
            lr=1e-3, momentum=0.9
        )
        optimizer.add_param_group({"params": head_params, "weight_decay": 0.0}) if not loss_type == 'triplet' else None
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    objective = objective.to(device)

    steps_per_epoch = len(train_loader)
    for g in optimizer.param_groups:
        if 'initial_lr' not in g:
            g['initial_lr'] = g['lr']

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=5,
        cycle_limit=1,
        t_in_epochs=True
    )
    use_amp=False
    scaler = amp.GradScaler('cuda', enabled=use_amp)
    if not checkpoint_path:
        train(epochs, backbone, objective, train_loader, val_loader, optimizer, scheduler, 
              ref_eval_set, seed, use_wandb, use_amp=use_amp, scaler=scaler, 
              accumulation_steps=accumulation_steps, run_name=run_name, embed_dim=emb_dim, 
              device=device, num_classes=num_classes,
              )
    
    if hyperbolic:
        backbone = nn.Sequential(
                backbone,
                LorentzEmbToTangent(manifold)
            )
        import warnings
        warnings.warn("USING TANGENT HEAD FOR EVALUATION")

    # Per dataset accuracy and wandb logging
    overall, per_dataset_acc = evaluate_per_dataset_with_knnmatcher(backbone, ref_eval_set, val_loader, device=device, dataset_col='dataset', force_rebuild=True)
    if use_wandb:
        wandb.log({"metrics/overall_acc": overall})
        for ds, acc in per_dataset_acc.items():
            wandb.summary[f"acc_by_dataset/{ds}"] = acc
        table = wandb.Table(columns=["dataset", "accuracy"])
        for ds, acc in sorted(per_dataset_acc.items()):
            table.add_data(ds, acc)
        wandb.log({"per_dataset_accuracy_table": table})
        wandb.log({
            "plots/per_dataset_accuracy":
                wandb.plot.bar(table, "dataset", "accuracy", title="Per-dataset Top-1 Accuracy")
        })
        wandb.finish()
    else:
        print(f"Overall acc: {overall}.See plot for per dataset acc.")

if __name__ == '__main__':
    main()
