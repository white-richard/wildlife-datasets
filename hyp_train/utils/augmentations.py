from dataclasses import dataclass
import torchvision.transforms as T

@dataclass
class AugCfg:
    policy: str = "baseline" # baseline | weak | strong | randaug | augmix
    randaug_n: int = 2
    randaug_m: int = 9
    erasing_p: float = 0.25
    crop_min_scale: float = 0.7

def build_train_tfms(img_size: int, a: AugCfg, mean: list[float], std: list[float]) -> T.Compose:
    d = img_size
    if a.policy == "weak":
        raise
        tfms = [
            T.RandomResizedCrop(d, scale=(max(0.85, a.crop_min_scale), 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), # need to test if this is ok for wildlife
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ]
    elif a.policy == "baseline":
        raise
        tfms = [
            T.RandomResizedCrop(d, scale=(a.crop_min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.25,0.25,0.25,0.05),
            T.RandomGrayscale(p=0.10),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.2),
            T.RandomAdjustSharpness(1.5, p=0.2),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
            T.RandomErasing(p=a.erasing_p, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=True),
        ]
    elif a.policy == "strong":
        raise
        tfms = [
            T.RandomResizedCrop(d, scale=(max(0.5, a.crop_min_scale), 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
            T.RandomPerspective(distortion_scale=0.3, p=0.2),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
            T.RandomErasing(p=min(0.4, a.erasing_p+0.1), scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random', inplace=True),
        ]
    elif a.policy == "randaug":
        # tfms = [
        #     T.RandomResizedCrop(d, scale=(a.crop_min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        #     T.RandomHorizontalFlip(),
        #     T.RandAugment(num_ops=a.randaug_n, magnitude=a.randaug_m),
        #     T.ToTensor(),
        #     T.Normalize([0.5]*3, [0.5]*3),
        #     T.RandomErasing(p=a.erasing_p, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=True),
        # ]
        tfms = [
            T.Resize(256),  # Resize shorter edge to 256, better preserves aspect ratio
            T.RandomCrop(224),  # Random 224x224 crop
            T.RandAugment(num_ops=2, magnitude=20),
            T.ToTensor(),
        ]
        if mean is not None and std is not None:
            tfms.append(T.Normalize(mean, std))
    elif a.policy == "augmix":
        raise
        tfms = [
            T.RandomResizedCrop(d, scale=(a.crop_min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.AugMix(
                severity=5, 
                mixture_width=3, 
                chain_depth=-1, 
                alpha=1.0, 
                all_ops=True, 
                interpolation=T.InterpolationMode.BILINEAR, 
                fill=[128, 128, 128]
            ),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
            T.RandomErasing(p=a.erasing_p, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random', inplace=True),
            
        ]
    else:
        raise ValueError(f"Unknown policy {a.policy}")
    return T.Compose(tfms)
