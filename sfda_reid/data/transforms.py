from torchvision import transforms
from typing import Callable

def get_train_transforms(height: int = 256, width: int = 128) -> Callable:
    """
    Returns a torchvision transform pipeline for training images.
    Args:
        height: Output image height
        width: Output image width
    Returns:
        transforms.Compose
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((height, width)),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    ])

def get_test_transforms(height: int = 256, width: int = 128) -> Callable:
    """
    Returns a torchvision transform pipeline for test images.
    Args:
        height: Output image height
        width: Output image width
    Returns:
        transforms.Compose
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
