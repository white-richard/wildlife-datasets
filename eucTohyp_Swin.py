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

def _hw_at_stage(model, stage_idx):
    """Compute H,W at the input of stage_idx."""
    H, W = _get_hw0(model)
    for i, stage in enumerate(model.layers[:stage_idx]):
        # downsample belongs to THIS stage in many timm Swins (applied at the end of the stage)
        if hasattr(stage, 'downsample') and stage.downsample is not None:
            if stage.downsample.__class__.__name__.lower() != 'identity':
                H //= 2; W //= 2
    return H, W

def _has_downsample(stage):
    """Check if stage has a meaningful downsample operation."""
    if not hasattr(stage, 'downsample') or stage.downsample is None:
        return False
    return stage.downsample.__class__.__name__.lower() != 'identity'


# --------- Euclid → Lorentz transition layer ---------
class EuclidToLorentzTransition(nn.Module):
    """
    Transition layer from Euclidean to Lorentz space.
    
    Input from timm (Euclidean): (B, H, W, C_in) or (B, N, C_in)
    1) Euclidean align: C_in -> C_e (linear, tokenwise)
    2) Lift: (B, N, C_e) -> (B, N, C_e+1) Lorentz
    Output: (B, N, C_e+1) on the Lorentz manifold.
    """
    def __init__(self, manifold, C_in, C_e, H, W):
        super().__init__()
        self.manifold = manifold
        self.C_in = int(C_in)
        self.C_e = int(C_e)
        self.H, self.W = int(H), int(W)
        
        # Euclidean aligner to match target width
        self.align = nn.Identity() if self.C_in == self.C_e else nn.Linear(self.C_in, self.C_e, bias=False)

    def _to_tokens(self, x):
        # Accept (B,N,C) or (B,H,W,C); return (B,N,C) + (H,W)
        if x.dim() == 3:
            B, N, C = x.shape
            H, W = self.H, self.W
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
        # 2) Euclidean align to target width
        assert x.shape[-1] == self.C_in, f"Expected C_in={self.C_in}, got {x.shape[-1]}"
        x = self.align(x)  # (B, N, C_e)
        # 3) lift to Lorentz
        x = lift_spatial_with_projx(self.manifold, x)     # (B, N, C_e+1) Lorentz
        x = self.manifold.projx(x)
        return x, H, W


# --------- Hyperbolic Swin Stage ---------
class HyperbolicSwinStage(nn.Module):
    """
    A hyperbolic Swin stage that operates on Lorentz manifold.
    """
    def __init__(self, manifold, C_in, C_out, depth, num_heads, window_size, 
                 mlp_ratio=4.0, dropout=0.0, downsample=False):
        super().__init__()
        self.manifold = manifold
        self.C_in = int(C_in)
        self.C_out = int(C_out)
        self.downsample_flag = downsample
        
        # The input C_in includes the time coordinate (+1 from Euclidean)
        # LorentzSwinLayer creates LayerNorm for (num_heads * dim - 1)
        # We need to pass dim such that num_heads * dim = C_in
        # so LayerNorm will be created for C_in - 1 (spatial dimensions only)
        
        Hhyp = int(num_heads)
        
        # Find the right number of heads that divides C_in evenly
        if self.C_in % Hhyp != 0:
            divisors = [d for d in range(min(Hhyp, self.C_in), 0, -1) if self.C_in % d == 0]
            Hhyp = divisors[0] if divisors else 1
        
        # Per-head dimension INCLUDING time coordinate
        # This makes total dimension = num_heads * dim = C_in
        # LayerNorm will be created for C_in - 1
        Dhyp = self.C_in // Hhyp
        
        # Build the Lorentz stage
        self.hlayer = LorentzSwinLayer(
            manifold=self.manifold,
            dim=Dhyp,  # Per-head dimension including time coordinate
            depth=depth,
            num_heads=Hhyp,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            downsample=self.downsample_flag
        )

    def forward(self, x, H, W):
        # x is already on Lorentz manifold: (B, N, C_in) where C_in includes time coord
        x = self.manifold.projx(x)
        x, H_out, W_out = self.hlayer(x, H, W, output_attentions=False)
        x = self.manifold.projx(x)
        return x, H_out, W_out


# --------- Combined hyperbolic stage sequence ---------
class HyperbolicStageSequence(nn.Module):
    """
    Combines transition layer and multiple hyperbolic stages.
    """
    def __init__(self, transition_layer, hyp_stages, manifold):
        super().__init__()
        self.transition = transition_layer
        self.hyp_stages = nn.ModuleList(hyp_stages)
        self.manifold = manifold
        self.is_hyperbolic = True
    
    def forward(self, x):
        # Transition from Euclidean to hyperbolic
        x, H, W = self.transition(x)
        
        # Apply hyperbolic stages
        for stage in self.hyp_stages:
            x, H, W = stage(x, H, W)
        
        # Final pooling for classification (optional)
        x = self.manifold.lorentzian_centroid(x, dim=1)  # (B, C+1)
        x = self.manifold.projx(x)
        
        return x


# --------- Sequential Layers Module ---------
class SequentialLayers(nn.Module):
    """
    A wrapper that makes a list of layers callable with proper forward method.
    This replaces the ModuleList to provide the required forward method.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __len__(self):
        return len(self.layers)
    
    def __iter__(self):
        return iter(self.layers)
    
    def forward(self, x, use_amp=True):
        # Use CUDA AMP only for Euclidean layers
        for layer in self.layers:
            is_hyp = getattr(layer, "is_hyperbolic", False)

            if is_hyp:
                # Ensure FP32 for numerical stability in hyperbolic ops
                if x.dtype != torch.float32:
                    x = x.float()
                x = layer(x)
            else:
                ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp)
                with ctx:
                    x = layer(x)
        return x


# --------- Generalized replacement function ---------
def replace_stages_with_hyperbolic(model, num_stages=1, manifold=None, mlp_ratio=4.0, dropout=0.0):
    """
    Replace the last num_stages stages with hyperbolic equivalents.
    
    Args:
        model: timm Swin model
        num_stages: number of stages from the end to replace (default: 1)
        manifold: hyperbolic manifold (default: Lorentz())
        mlp_ratio: MLP expansion ratio
        dropout: dropout rate
    
    Returns:
        model: modified model
        hyp_stages: list of created hyperbolic stages
    """
    if manifold is None:
        manifold = Lorentz()
    
    stages = model.layers
    num_stages = min(num_stages, len(stages))  # Don't exceed available stages
    
    if num_stages <= 0:
        return model, []
    
    # Calculate the stage index where hyperbolic starts
    hyp_start_idx = len(stages) - num_stages
    
    # Get dimensions and properties for each stage to be replaced
    hyp_stages = []
    transition_layer = None
    current_hyp_dim = None  # Track current hyperbolic dimension
    
    for i in range(num_stages):
        stage_idx = hyp_start_idx + i
        euclid_stage = stages[stage_idx]
        
        # Get stage properties
        C_stage = _stage_width(euclid_stage)  # Euclidean width (e.g., 1024)
        depth = len(euclid_stage.blocks)
        heads = getattr(euclid_stage.blocks[0].attn, 'num_heads')
        window_size = _get_window_size_from_block(euclid_stage.blocks[-1])
        has_downsample = _has_downsample(euclid_stage)
        
        # Get spatial dimensions at this stage
        H, W = _hw_at_stage(model, stage_idx)
        window_size = min(window_size, H, W)
        
        if i == 0:  # First hyperbolic stage needs transition from Euclidean
            # Get input width from previous stage or patch embedding
            if stage_idx > 0:
                prev_stage = stages[stage_idx - 1]
                # Check if previous stage has downsample that changes channels
                if _has_downsample(prev_stage):
                    # Get the output channels after downsample
                    if hasattr(prev_stage.downsample, 'proj'):
                        C_in = prev_stage.downsample.proj.out_features
                    else:
                        C_in = _stage_width(prev_stage)
                else:
                    C_in = _stage_width(prev_stage)
            else:
                # First stage - get from patch embedding
                if hasattr(model.patch_embed, 'proj'):
                    C_in = model.patch_embed.proj.out_features
                else:
                    C_in = C_stage
            
            # Create transition layer - this lifts from C_stage to C_stage + 1
            transition_layer = EuclidToLorentzTransition(
                manifold=manifold,
                C_in=C_in,
                C_e=C_stage,  # Target Euclidean width
                H=H, W=W
            )
            
            # Set initial hyperbolic dimension
            current_hyp_dim = C_stage + 1  # +1 for Lorentz dimension
            
            # Create first hyperbolic stage 
            hyp_stage = HyperbolicSwinStage(
                manifold=manifold,
                C_in=current_hyp_dim,
                C_out=current_hyp_dim,  # For now, keep same dimension
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                downsample=has_downsample
            )
            
            # Update current dimension based on downsample
            if has_downsample:
                # In typical Swin, channels double when spatial dims halve
                current_hyp_dim = current_hyp_dim * 2 - 1  # Double spatial, keep 1 time coord
                
        else:
            # Subsequent hyperbolic stages
            # Input dimension is whatever the previous hyperbolic stage outputs
            C_in_hyp = current_hyp_dim
            C_out_hyp = C_stage + 1  # Target hyperbolic width for this stage
                
            hyp_stage = HyperbolicSwinStage(
                manifold=manifold,
                C_in=C_in_hyp,
                C_out=C_out_hyp,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                downsample=has_downsample
            )
            
            # Update current dimension
            current_hyp_dim = C_out_hyp
            if has_downsample:
                current_hyp_dim = current_hyp_dim * 2 - 1  # Double spatial, keep 1 time coord
        
        hyp_stages.append(hyp_stage)
    
    # Create combined hyperbolic module
    combined_hyp_module = HyperbolicStageSequence(
        transition_layer, hyp_stages, manifold
    )
    
    # Replace the stages with a proper sequential module
    new_layers = list(stages[:hyp_start_idx]) + [combined_hyp_module]
    model.layers = SequentialLayers(new_layers)
    
    return model, [combined_hyp_module]


# --------- Backward compatibility function ---------
def replace_last_layer_with_hyperbolic(model, manifold=None, mlp_ratio=4.0, dropout=0.0):
    """Backward compatibility: replace only the last stage."""
    return replace_stages_with_hyperbolic(
        model, num_stages=1, manifold=manifold, mlp_ratio=mlp_ratio, dropout=dropout
    )


# ----------------- examples -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example 1: Replace only last stage (most stable)
    print("Example 1: Replacing last stage only")
    model1 = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True).to(device).eval()
    model1, hyp_stages1 = replace_last_layer_with_hyperbolic(model1)
    model1.to(device).eval()
    
    # Disable post-processing
    if hasattr(model1, "norm"):
        model1.norm = nn.Identity()
    if hasattr(model1, "global_pool"):
        model1.global_pool = ""
    model1.head = nn.Identity()
    
    x = torch.randn(2, 3, 224, 224, device=device)
    z_L1 = model1.forward_features(x)
    print(f"Output shape (1 hyperbolic stage): {z_L1.shape}")

    # Example 2: Replace last 2 stages (more complex)
    print("\nExample 2: Replacing last 2 stages")
    model2 = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True).to(device).eval()
    model2, hyp_stages2 = replace_stages_with_hyperbolic(model2, num_stages=2)
    model2.to(device).eval()
    
    # Disable post-processing
    if hasattr(model2, "norm"):
        model2.norm = nn.Identity()
    if hasattr(model2, "global_pool"):
        model2.global_pool = ""
    model2.head = nn.Identity()
    
    z_L2 = model2.forward_features(x)
    print(f"Output shape (2 hyperbolic stages): {z_L2.shape}")

    # Example 3: Replace last 3 stages
    print("\nExample 3: Replacing last 3 stages")
    model3 = timm.create_model('hf-hub:BVRA/MegaDescriptor-B-224', num_classes=0, pretrained=True).to(device).eval()
    model3, hyp_stages3 = replace_stages_with_hyperbolic(model3, num_stages=3)
    model3.to(device).eval()
    
    # Disable post-processing
    if hasattr(model3, "norm"):
        model3.norm = nn.Identity()
    if hasattr(model3, "global_pool"):
        model3.global_pool = ""
    model3.head = nn.Identity()
    
    z_L3 = model3.forward_features(x)
    print(f"Output shape (3 hyperbolic stages): {z_L3.shape}")