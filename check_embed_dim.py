import torch

def get_backbone_embedding_dim(backbone: torch.nn.Module, img_size: int) -> int:
    """
    Returns the embedding dimension of a backbone model by running a dummy forward pass.

    Args:
        backbone (torch.nn.Module): The backbone model to inspect.
        img_size (int): The input image size (assumes square input).

    Returns:
        int: The embedding dimension of the backbone's output.
    """
    # Save current training state
    was_training = backbone.training

    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        out = backbone(dummy)
    
    # Restore training state
    if was_training:
        backbone.train()

    return out.shape[1]
