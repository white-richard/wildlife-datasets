import torch
import torch.nn as nn
from tqdm import tqdm


import torch
from torch.utils.data import Dataset

class WithIndex(Dataset):
    """Wraps any dataset to return (x, y, idx) regardless of the base sample shape.

    Accepts base items like:
      - (x, y)
      - (x, y, ...)
      - dict with keys like 'image'/'img'/'x' and optional 'label'/'y'
      - bare x
    """
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]

        x, y = None, None
        if isinstance(sample, (list, tuple)):
            # take first as x, second (if present) as y; ignore the rest
            x = sample[0]
            y = sample[1] if len(sample) > 1 else None
        elif isinstance(sample, dict):
            x = sample.get("image") or sample.get("img") or sample.get("x")
            if x is None:
                # fallback: take the first value
                x = next(iter(sample.values()))
            y = sample.get("label", sample.get("y", None))
        else:
            # bare tensor/image
            x = sample
            y = None

        # ensure y is a tensor/int (some losses expect a label tensor)
        if y is None:
            y = -1  # dummy label for unlabeled rows
        if not torch.is_tensor(y):
            try:
                y = torch.as_tensor(y, dtype=torch.long)
            except Exception:
                y = torch.tensor(-1, dtype=torch.long)

        return x, y, i


import os, numpy as np, torch.nn.functional as F

def _dtype_np(kind: str):
    return np.float16 if kind.lower() == "fp16" else np.float32

class TeacherFeatureBank:
    """
    Memory-mapped N x D feature bank (row = dataset index). Random-access friendly.
    Features are stored L2-normalized.
    """
    def __init__(self, path: str, num_items: int, dim: int, dtype: str = "fp16", create: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "w+" if create else "r+"
        self._arr = np.memmap(path, dtype=_dtype_np(dtype), mode=mode, shape=(num_items, dim))
        self.dim = dim
        self.num_items = num_items
        self.dtype = dtype

    def write_rows(self, idxs: np.ndarray, feats: torch.Tensor):
        arr = self._arr
        f = feats.detach().cpu().to(dtype=torch.float16 if self.dtype=='fp16' else torch.float32).numpy()
        arr[idxs] = f

    def flush(self):
        self._arr.flush()

    def gather(self, idxs: torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(self._arr[idxs.cpu().numpy()])  # [B, D]

        return x

def precompute_teacher_features(cfg, teacher: nn.Module, dataset: torch.utils.data.Dataset,
                                device: torch.device, img_size: int, out_path: str, dtype: str = "fp16",
                                batch_size: int = 256, num_workers: int = 8) -> TeacherFeatureBank:
    teacher.eval()
    class _IdxOnly(torch.utils.data.Dataset):
        """Wrap a dataset and return only (image_tensor, global_index)."""
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            sample = self.base[i]
            # Handle (x, y), (x, y, idx), dicts, or bare x
            if isinstance(sample, (list, tuple)):
                x = sample[0]
            elif isinstance(sample, dict):
                # common pattern in custom datasets
                x = sample.get("image", sample.get("img", sample.get("x", None)))
                if x is None:
                    # fallback: take first value
                    x = next(iter(sample.values()))
            else:
                x = sample
            return x, i


    from torch.utils.data import DataLoader
    dl = DataLoader(_IdxOnly(dataset), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    # Probe teacher dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        d = teacher(dummy).shape[1]

    bank = TeacherFeatureBank(out_path, num_items=len(dataset), dim=d, dtype=dtype, create=True)

    with torch.no_grad():
        for xb, idx in tqdm(dl, desc="Precompute teacher feats"):
            xb = xb.to(device, non_blocking=True)
            z = teacher(xb)                 # [B, d]
            z = F.normalize(z, dim=-1)      # cosine space
            bank.write_rows(idx.numpy(), z)
    bank.flush()
    return bank
