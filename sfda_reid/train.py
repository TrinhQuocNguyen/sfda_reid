import argparse
import sys
from typing import Any
from pathlib import Path

import torch
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfda_reid.data.dataloader import build_test_loader, build_train_loader
from sfda_reid.data.datasets.dukemtmc import DukeMTMC
from sfda_reid.data.datasets.market1501 import Market1501
from sfda_reid.data.datasets.msmt17 import MSMT17
from sfda_reid.data.transforms import get_test_transforms, get_train_transforms
from sfda_reid.engine.source_trainer import SourceTrainer
from sfda_reid.models.backbone.resnet import ResNet50Backbone
from sfda_reid.models.backbone.vit import ViTBackbone
from sfda_reid.utils.logger import setup_logger
from sfda_reid.utils.seed import set_seed

def get_dataset(name: str, root: str, mode: str, transform: Any):
    if name == 'market1501':
        return Market1501(root, mode=mode, transform=transform)
    if name == 'dukemtmc':
        return DukeMTMC(root, mode=mode, transform=transform)
    if name == 'msmt17':
        return MSMT17(root, mode=mode, transform=transform)
    raise ValueError(f"Unknown dataset: {name}")

def main():
    parser = argparse.ArgumentParser(description='Source Model Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--phase', type=str, choices=['source'], required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    logger = setup_logger('train', save_dir=cfg.source.output_dir)
    device = torch.device(cfg.device)
    
    logger.info("=" * 80)
    logger.info("TRAINING SCRIPT STARTED")
    logger.info("=" * 80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Training phase: {args.phase}")
    logger.info(f"Random seed: {cfg.seed}")
    logger.info(f"Device: {device}")
    
    # Build datasets
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING DATASETS")
    logger.info("=" * 80)
    logger.info(f"Loading {cfg.source.dataset} training set from {cfg.source.data_root}")
    train_set = get_dataset(cfg.source.dataset, cfg.source.data_root, 'train', get_train_transforms())
    logger.info(f"Training set loaded: {len(train_set)} samples")
    
    logger.info(f"Loading {cfg.source.dataset} validation set from {cfg.source.data_root}")
    val_set = get_dataset(cfg.source.dataset, cfg.source.data_root, 'gallery', get_test_transforms())
    logger.info(f"Validation set loaded: {len(val_set)} samples")
    
    # Build loaders
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING DATA LOADERS")
    logger.info("=" * 80)
    train_loader = build_train_loader(train_set, cfg)
    val_loader = build_test_loader(val_set, cfg)
    logger.info(f"Train loader: {len(train_loader)} batches of size {cfg.source.batch_size}")
    logger.info(f"Val loader: {len(val_loader)} batches of size 256")
    
    # Build model
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING MODEL")
    logger.info("=" * 80)
    logger.info(f"Model backbone: {cfg.source.backbone}")
    logger.info(f"Pretrained: {cfg.source.pretrained}")
    logger.info(f"Number of classes: {cfg.source.num_classes}")
    
    if cfg.source.backbone == 'resnet50':
        model = ResNet50Backbone(num_classes=cfg.source.num_classes, pretrained=cfg.source.pretrained).to(device)
        logger.info("ResNet50 backbone created")
    else:
        model = ViTBackbone(num_classes=cfg.source.num_classes, pretrained=cfg.source.pretrained).to(device)
        logger.info("ViT backbone created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING TRAINER")
    logger.info("=" * 80)
    trainer = SourceTrainer(model, train_loader, val_loader, cfg)
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info('Training complete.')

if __name__ == '__main__':
    main()
