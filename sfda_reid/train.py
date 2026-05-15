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
    set_seed(cfg.seed)
    logger = setup_logger('train', save_dir=cfg.source.output_dir)
    device = torch.device(cfg.device)
    # Build datasets
    train_set = get_dataset(cfg.source.dataset, cfg.source.data_root, 'train', get_train_transforms())
    val_set = get_dataset(cfg.source.dataset, cfg.source.data_root, 'gallery', get_test_transforms())
    # Build loaders
    train_loader = build_train_loader(train_set, cfg)
    val_loader = build_test_loader(val_set, cfg)
    # Build model
    if cfg.source.backbone == 'resnet50':
        model = ResNet50Backbone(num_classes=cfg.source.num_classes, pretrained=cfg.source.pretrained).to(device)
    else:
        model = ViTBackbone(num_classes=cfg.source.num_classes, pretrained=cfg.source.pretrained).to(device)
    # Trainer
    trainer = SourceTrainer(model, train_loader, val_loader, cfg)
    trainer.train()
    logger.info('Training complete.')

if __name__ == '__main__':
    main()
