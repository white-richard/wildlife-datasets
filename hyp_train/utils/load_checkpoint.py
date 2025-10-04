import torch

def load_checkpoint(checkpoint_path, backbone, head=None, optimizer=None, scheduler=None, map_location="cpu"):
    """
    Loads model, optimizer, and scheduler weights from a checkpoint.

    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
        backbone (torch.nn.Module): The backbone model to load weights into.
        head (torch.nn.Module, optional): The head module to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to restore state.
        map_location (str): Device mapping for loading ('cpu' or 'cuda').

    Returns:
        dict: Dictionary containing metadata like epoch, best_mAP, learning_rate, and seed.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model weights
    backbone.load_state_dict(checkpoint["model"])
    if head is not None and "head" in checkpoint and checkpoint["head"] is not None:
        head.load_state_dict(checkpoint["head"])
    
    # Restore optimizer and scheduler if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    print(f"Loaded checkpoint from '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    
    # Return metadata
    return {
        "epoch": checkpoint.get("epoch"),
        "best_mAP": checkpoint.get("best_mAP"),
        "learning_rate": checkpoint.get("learning_rate"),
        "seed": checkpoint.get("seed"),
    }
