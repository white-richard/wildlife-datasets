import torch
import torch.nn as nn
import timm

from hypercore.manifolds import Lorentz
from hypercore.models.Swin_LViT import LorentzSwinLayer
from hypercore.utils.manifold_utils import lift_spatial_with_projx


def _get_window_size_from_block(block):
    attn = getattr(block, 'attn', None)
    ws = getattr(attn, 'window_size', 7) if attn is not None else 7
    return int(ws[0] if isinstance(ws, (tuple, list)) else ws)

def _stage_width(stage):
    """Return the channel width C for a (Euclidean) Swin stage (from its first block)."""
    blk0 = stage.blocks[0]
    if hasattr(blk0, 'norm1') and hasattr(blk0.norm1, 'normalized_shape'):
        return int(blk0.norm1.normalized_shape[0])
    if hasattr(stage, 'norm') and hasattr(stage.norm, 'normalized_shape'):
        return int(stage.norm.normalized_shape[0])
    # fallback via qkv
    qkv = getattr(getattr(blk0, 'attn', object()), 'qkv', None)
    if qkv is not None and hasattr(qkv, 'out_features'):
        return int(qkv.out_features // 3)
    raise RuntimeError("Couldn't infer stage width.")

def _get_hw0(model):
    """Grid at stage 0."""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'grid_size'):
        gh, gw = model.patch_embed.grid_size
    else:
        img = model.default_cfg.get('input_size', (3, 224, 224))[1:]
        ps = model.patch_embed.patch_size if hasattr(model.patch_embed, 'patch_size') else (4, 4)
        gh, gw = img[0] // ps[0], img[1] // ps[1]
    return int(gh), int(gw)

def _hw_before_last_stage(model):
    """Compute H,W at the input of the last stage (what timm will feed it)."""
    H, W = _get_hw0(model)
    for i, stage in enumerate(model.layers):
        # downsample belongs to THIS stage in many timm Swins (applied at the end of the stage)
        # so H,W are halved AFTER processing this stage if stage.downsample is set
        if i == len(model.layers) - 1:
            # stop before applying last stage's downsample (there usually isn't one)
            break
        if hasattr(stage, 'downsample') and stage.downsample is not None:
            if stage.downsample.__class__.__name__.lower() != 'identity':
                H //= 2; W //= 2
    return H, W


# --------- Euclid → Lorentz last stage wrapper ---------
class EuclidToLorentzLastLayer(nn.Module):
    """
    Replaces the last Swin stage.

    Input from timm (Euclidean): (B, H, W, C_in) or (B, N, C_in)
    1) Euclidean align: C_in -> C_e (linear, tokenwise)
    2) Lift: (B, N, C_e) -> (B, N, C_e+1) Lorentz
    3) Hyperbolic stage: LorentzSwinLayer (no downsample)
    Output: (B, N, C_e+1) on the Lorentz manifold.
    """
    def __init__(self, manifold, C_in, C_e, depth, num_heads, H_l, W_l, window_size, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.manifold = manifold
        self.C_in = int(C_in)   # width coming FROM previous stage (e.g., 512)
        self.C_e  = int(C_e)    # width OF the last stage (e.g., 1024)
        self.H_l, self.W_l = int(H_l), int(W_l)
        self.window_size = int(min(window_size, H_l, W_l))

        # Small Euclidean aligner to match last-stage width, if needed.
        self.align = nn.Identity() if self.C_in == self.C_e else nn.Linear(self.C_in, self.C_e, bias=False)

        # Choose hyper heads and per-head dim for C_h = C_e + 1
        C_h = self.C_e + 1
        Hhyp = int(num_heads)
        if C_h % Hhyp != 0:
            divisors = [d for d in range(min(Hhyp, C_h), 0, -1) if C_h % d == 0]
            Hhyp = divisors[0]
        Dhyp = C_h // Hhyp

        # Build the Lorentz last stage (no downsample here)
        self.hlayer = LorentzSwinLayer(
            manifold=self.manifold,
            dim=Dhyp,
            depth=depth,
            num_heads=Hhyp,
            window_size=self.window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            downsample=False
        )

    def _to_tokens(self, x):
        # Accept (B,N,C) or (B,H,W,C); return (B,N,C) + (H,W)
        if x.dim() == 3:
            B, N, C = x.shape
            H, W = self.H_l, self.W_l
            assert N == H * W, f"Expected N=H*W; got {N} vs {H}*{W}"
            return x, H, W
        elif x.dim() == 4:
            # timm Swin commonly uses channels-last here: (B, H, W, C)
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            return x, H, W
        else:
            raise ValueError(f"Expected 3-D or 4-D input, got {tuple(x.shape)}")

    def forward(self, x):
        # 1) ensure tokens & get spatial size
        x, H, W = self._to_tokens(x)
        # 2) Euclidean align to last-stage width
        assert x.shape[-1] == self.C_in, f"Expected C_in={self.C_in}, got {x.shape[-1]}"
        x = self.align(x)  # (B, N, C_e)
        # 3) lift to Lorentz
        x = lift_spatial_with_projx(self.manifold, x)     # (B, N, C_e+1) Lorentz
        x = self.manifold.projx(x)
        # 4) hyperbolic stage
        x, _, _ = self.hlayer(x, H, W, output_attentions=False)
        x = self.manifold.projx(x)
        x = self.manifold.lorentzian_centroid(x, dim=1)  # (B, C_e+1)
        x = self.manifold.projx(x)
        return x  # (B, N, C_e+1) Lorentz
        

# --------- swapper ---------
def replace_last_layer_with_hyperbolic(model, manifold=None, mlp_ratio=4.0, dropout=0.0):
    manifold = Lorentz() if manifold is None else manifold

    stages = model.layers
    last_euclid_stage = stages[-1]
    prev_stage = stages[-2] if len(stages) >= 2 else None

    C_e = _stage_width(last_euclid_stage)  # e.g., 1024 (width of last stage)
    C_in = _stage_width(prev_stage) if prev_stage else C_e  # e.g., 512 (from previous stage)
    depth = len(last_euclid_stage.blocks)
    heads = getattr(last_euclid_stage.blocks[0].attn, 'num_heads')
    H_l, W_l = _hw_before_last_stage(model)
    window_size = _get_window_size_from_block(last_euclid_stage.blocks[-1])

    hyp_layer = EuclidToLorentzLastLayer(
        manifold=manifold,
        C_in=C_in,
        C_e=C_e,
        depth=depth,
        num_heads=heads,
        H_l=H_l, W_l=W_l,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )

    # swap
    if hasattr(model, 'layers'):
        model.layers[-1] = hyp_layer
    else:
        model.stages[-1] = hyp_layer

    return model, hyp_layer


# ----------------- example -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True).to(device).eval()

    # swap last layer → Lorentz
    model, hyp_layer = replace_last_layer_with_hyperbolic(model)
    model.to(device).eval()

    model.head = nn.Identity()  # disable any classifier head

    # disable Euclidean post-processing
    if hasattr(model, "norm"):        # LayerNorm(C_e) would break on C_e+1
        model.norm = nn.Identity()
    if hasattr(model, "global_pool"): # ensure forward_features returns tokens
        model.global_pool = ""
    model.reset_classifier(num_classes=0, global_pool='')

    # forward
    x = torch.randn(2, 3, 224, 224, device=device)
    z_L = model.forward_features(x)   # (B, N, C_e+1)

    # Lorentz pooling (CLS-like)
    print("Lorentz embedding:", z_L.shape)
