import os
import random
from collections import defaultdict
import itertools
import pickle
import time

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import amp
import torchvision.transforms as T
from PIL import Image
import timm
from tqdm import tqdm, trange
import wandb
import numpy as np
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler

from wildlife_tools.train import ArcFaceLoss
from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets import splits
from wildlife_tools.data import FeatureDatabase 
from wildlife_tools.inference import KnnMatcher  

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

class WR10kDataset(Dataset):
    def __init__(self, df, root, transform, id_list=None):
        self.df = df.reset_index(drop=True)
        ids = sorted(id_list if id_list is not None else self.df['identity'].unique().tolist())
        self.id2idx = {k: i for i, k in enumerate(ids)}
        self.root = root
        self.tfm = transform
        self.num_classes = len(self.id2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r['path']}").convert("RGB")
        x = self.tfm(img)
        y = self.id2idx[r['identity']]
        return x, y, r['path'], r['identity']

class PKSampler(Sampler):
    def __init__(self, labels, P=32, K=4):
        self.P, self.K = P, K
        self.label_to_indices = defaultdict(list)
        for idx, y in enumerate(labels):
            self.label_to_indices[y].append(idx)
        self.labels = list(self.label_to_indices.keys())
        for y in self.labels:
            random.shuffle(self.label_to_indices[y])

    def __iter__(self):
        while True:
            ids = random.sample(self.labels, self.P)
            batch = []
            for y in ids:
                pool = self.label_to_indices[y]
                take = (random.sample(pool, self.K)
                        if len(pool) >= self.K else
                        random.choices(pool, k=self.K))
                batch.extend(take)
            yield batch

    def __len__(self):
        return 10**9

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
def evaluate_top1(model, ref_ds, qry_loader, device="cuda"):
    t0 = time.time()
    model.eval()
    ref_loader = DataLoader(ref_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    def _extract(loader):
        embs, labels = [], []
        for xb, yb, *_ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            zb = torch.nn.functional.normalize(model(xb), dim=1)
            embs.append(zb)
            labels.append(yb)
        return torch.cat(embs, 0), torch.cat(labels, 0)

    G, gid = _extract(ref_loader)
    Q, qid = _extract(qry_loader)
    S = Q @ G.T
    top1 = S.argmax(dim=1)
    acc = (gid[top1] == qid).float().mean()
    t1 = time.time()
    end_t = t1 - t0
    print(f"Elapsed time: {end_t:.4f} seconds")
    return acc.item()

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
    # Basic checks and maps
    assert hasattr(ref_ds, 'id2idx'), "ref_ds must have an 'id2idx' mapping"
    assert hasattr(ref_ds, 'df'), "ref_ds must have a 'df' DataFrame"
    assert dataset_col in ref_ds.df.columns, (
        f"'{dataset_col}' not in df columns: {list(ref_ds.df.columns)}"
    )
    ref_loader = DataLoader(ref_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model.eval().to(device)

    idx2id = {v: k for k, v in ref_ds.id2idx.items()}  # int label -> str identity
    id2dataset = (
        ref_ds.df.drop_duplicates('identity')
        .set_index('identity')[dataset_col]
        .to_dict()
    )

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
        ds_name = id2dataset[t_id]
        per_dataset_acc[ds_name].append(p_id == t_id)

    per_dataset_acc = {ds: float(np.mean(v)) for ds, v in per_dataset_acc.items()}
    overall = float(np.mean(list(per_dataset_acc.values()))) if per_dataset_acc else float('nan')
    return overall, per_dataset_acc


def plot_per_dataset_accuracy(backbone, ref_ds, val_loader, device='cuda', dataset_col='dataset'):
    overall, per_dataset_acc = evaluate_per_dataset_with_knnmatcher(
        backbone, ref_ds, val_loader,
        database_file="ref_database.pkl",
        device="cuda",
        dataset_col="dataset",
        force_rebuild=False,
    )

    ds_names = list(per_dataset_acc.keys())
    vals = np.array([per_dataset_acc[k] for k in ds_names], dtype=float)

    plt.figure(figsize=(max(6, 0.35*len(ds_names)), 4.5))
    plt.bar(range(len(ds_names)), vals)
    plt.xticks(range(len(ds_names)), ds_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Top-1 accuracy")
    plt.title(f"Per-dataset accuracy (overall={overall:.3f})")
    plt.tight_layout()
    plt.savefig("per_dataset_accuracy.png", dpi=200, bbox_inches="tight")
    return overall, per_dataset_acc

def train_one_epoch(model, head, loader, optimizer, scheduler, epoch,
                    scaler=None, accumulation_steps=1, device='cuda', use_amp=False):
    model.train()
    head.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    steps_per_epoch = len(loader)
    pbar = tqdm(enumerate(loader), total=steps_per_epoch,
                        leave=False)

    running_loss = torch.zeros((), device=device)

    for step, batch in pbar:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            feats = model(x)
            loss = head(feats, y) / accumulation_steps

        scaler.scale(loss).backward()
        running_loss += loss.detach()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
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
    eval_warmup_frac=0.40, # start validation after 75% of epochs
    eval_interval=1, # run validation every N epochs once warmup passed
    patience=5, # early-stop after N evals with no top1 improvement
    device='cuda',
    save_dir='checkpoints',
    run_name='wr10k_megadesc_arcface'
):
    os.makedirs(save_dir, exist_ok=True)

    best_top1 = -float('inf')
    best_epoch = -1
    this_top1 = None
    start_eval_epoch = max(1, int(np.ceil(eval_warmup_frac * epochs)))
    no_improve_evals = 0

    for epoch in trange(1, epochs + 1, desc="Epochs"):
        train_loss = train_one_epoch(
            backbone, objective, train_loader, optimizer, scheduler, epoch,
            scaler=scaler, accumulation_steps=accumulation_steps, device=device, use_amp=use_amp
        )
        scheduler.step(epoch + 1)

        do_validate = (epoch >= start_eval_epoch) and ((epoch - start_eval_epoch) % eval_interval == 0)
        if do_validate:
            this_top1 = evaluate_top1(backbone, ref_eval_set, val_loader, device=device)
            
            if this_top1 > best_top1:
                best_top1 = this_top1
                best_epoch = epoch
                no_improve_evals = 0
                torch.save({
                    'epoch': epoch,
                    'model': backbone.state_dict(),
                    'head': objective.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'seed': seed,
                    'best_top1': best_top1,
                }, os.path.join(save_dir, f'{run_name}_best_epoch{epoch}.pt'))
                print(f"[VAL] New best @ epoch {epoch}: top1={best_top1:.4f} (saved)")
            else:
                no_improve_evals += 1
                print(f"[VAL] top1={this_top1:.4f} (best={best_top1:.4f} @ {best_epoch}) "
                      f"| patience {no_improve_evals}/{patience}")

            if (no_improve_evals >= patience):
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} validation rounds).")
                # break
        else:
            pass

        if use_wandb:
            log_dict = {"epoch": epoch, "train_loss": float(train_loss)}
            if this_top1 is not None:
                log_dict["val_top1"] = float(this_top1)
                log_dict["best_top1"] = float(best_top1)
            wandb.log(log_dict, step=epoch)
        else:
            if this_top1 is not None:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_top1={this_top1:.4f} | best_top1={best_top1:.4f}")
            else:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | (val skipped until epoch {start_eval_epoch})")

    final_ckpt = {
        'epoch': epoch,
        'model': backbone.state_dict(),
        'head': objective.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'seed': seed,
        'best_top1': best_top1,
        'best_epoch': best_epoch,
    }
    torch.save(final_ckpt, os.path.join(save_dir, f'{run_name}_final.pt'))
    print(f"\nTraining done. Best top1={best_top1:.4f} at epoch {best_epoch}. Final checkpoint saved.")

import torch.nn as nn

def split_decay_groups(model: nn.Module):
    """Return (decay, no_decay) parameter lists for a model.
    - no_decay: LayerNorm weights & biases, plus all biases
    - decay: everything else
    """
    ln_param_names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
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

import torch.nn.init as init

def init_weights_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight) # or xavier_normal_
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)

def main():
    use_wandb = True
    run_name='MegaDescriptor-B-224'
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

    db_path = "ref_database.pkl"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted old {db_path}")
        
    root = "/home/richw/.richie/repos/wildlifereid-10k"
    meta = WildlifeReID10k(root)
    df = meta.df
    splitter = splits.ClosedSetSplit(ratio_train=0.8, col_label="identity") # keep IDs same in train/test
    idx_ref, idx_qry = splitter.split(df)[0]
    df_ref = df.iloc[idx_ref].copy()
    df_qry = df.iloc[idx_qry].copy()

    # from restructure_wildlife_dataset_prompt_cam import export_to_class_folders
    # train_out = "./tmp/wr10k_promptrun/train"
    # val_out   = "./tmp/wr10k_promptrun/val"
    # os.makedirs(train_out, exist_ok=True)
    # os.makedirs(val_out, exist_ok=True)
    # export_to_class_folders(df_ref, root, train_out)
    # export_to_class_folders(df_qry, root, val_out)

    train_set = WR10kDataset(df_ref, root, train_tfms)
    id_list = sorted(train_set.id2idx.keys())
    ref_eval_set = WR10kDataset(df_ref, root, val_tfms, id_list=id_list)
    val_set = WR10kDataset(df_qry, root, val_tfms, id_list=id_list)

    train_classes = set(train_set.id2idx.values())
    val_classes   = set(val_set.id2idx.values())
    assert train_classes == val_classes, \
        f"Train/val class index mismatch: {train_classes ^ val_classes}"
    
    # train_labels = [train_set.id2idx[i] for i in train_set.df['identity']]
    print("Total number of images TRAIN:", len(train_set))
    print("Total number of images VAL:", len(val_set))
    P, K = 32, 4
    # sampler = PKSampler(train_labels, P=P, K=K)
    # train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=128, num_workers=8, pin_memory=True, drop_last=False, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    checkpoint_path = None
    #checkpoint_path = "checkpoints/MegaDescriptor-B-224_best_epoch65.pt"
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=False)
    backbone.apply(init_weights_xavier)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        backbone.load_state_dict(ckpt["model"])

    objective = ArcFaceLoss(
        num_classes=train_set.num_classes,
        embedding_size=backbone.num_features,
        margin=0.5, scale=64
    )

    epochs = 100
    wd = 1e-4
    #backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    backbone_decay, backbone_no_decay = split_decay_groups(backbone)

    head_params = [p for p in objective.parameters() if p.requires_grad]
    print(f"Num classes: {train_set.num_classes}")
    print(f"Backbone: decay={sum(p.numel() for p in backbone_decay)}, "
      f"no_decay={sum(p.numel() for p in backbone_no_decay)}, "
      f"head={sum(p.numel() for p in head_params)}")
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_decay,     "weight_decay": wd},
            {"params": backbone_no_decay,  "weight_decay": 0.0},  # LayerNorms & biases
            {"params": head_params,        "weight_decay": 0.0},  # classification/embedding head
        ],
        lr=1e-3, momentum=0.9
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    objective = objective.to(device)

    effective_batch = P * K
    accumulation_steps = 1

    steps_per_epoch = len(train_loader)
    for g in optimizer.param_groups:
        if 'initial_lr' not in g:
            g['initial_lr'] = g['lr']
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=epochs, eta_min=1e-6
    #)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=5,
        cycle_limit=1,
        t_in_epochs=True
    )
    use_amp=True
    scaler = amp.GradScaler('cuda', enabled=use_amp)
    if not checkpoint_path:
        train(epochs, backbone, objective, train_loader, val_loader, optimizer, scheduler, ref_eval_set, seed, use_wandb, use_amp=use_amp, scaler=scaler, accumulation_steps=accumulation_steps, run_name=run_name)
    
    # Per dataset accuracy and wandb logging
    overall, per_dataset_acc = plot_per_dataset_accuracy(backbone, ref_eval_set, val_loader, device, dataset_col='dataset')
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
