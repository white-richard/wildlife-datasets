import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import copy
from .wr10k_dataset import WR10kDataset


def calculate_normalization_stats(df_train_fit, root, image_size=224):
    """Calculate mean and std for dataset"""
   

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()  # Converts to [0,1]
    ])

    dataset = WR10kDataset(df_train_fit, root, transform)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Print collage of some training images to see if the augmentations look ok
    import copy
    import os
    import torchvision
    from torchvision import transforms as T
    tmp_loader = copy.deepcopy(loader)
    iterr = iter(tmp_loader)
    # grab the second batch
    batch = next(iterr)
    # batch = next(iterr)
    images = batch[0]
    grid = T.ToPILImage()(torchvision.utils.make_grid(images[:16], nrow=4))
    # save to file
    grid.save(os.path.join(f"meow_train_augmentations.png"))
    del tmp_loader
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    # Calculate mean
    for batch in loader:
        images = batch[0]
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_size
    
    mean /= total_images
    
    # Calculate std
    for batch in loader:
        images = batch[0]
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0,2])
        
    std = torch.sqrt(std / (total_images * image_size * image_size))
    
    return mean.tolist(), std.tolist()