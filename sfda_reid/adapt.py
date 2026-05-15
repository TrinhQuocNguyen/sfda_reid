import argparse
import sys
from typing import Any
from pathlib import Path

import torch
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sfda_reid.data.dataloader import build_target_loader
from sfda_reid.data.datasets.dukemtmc import DukeMTMC
from sfda_reid.data.datasets.market1501 import Market1501
from sfda_reid.data.datasets.msmt17 import MSMT17
from sfda_reid.data.transforms import get_test_transforms
from sfda_reid.engine.trainer import SFDATrainer
from sfda_reid.models.backbone.resnet import ResNet50Backbone
from sfda_reid.models.backbone.vit import ViTBackbone
from sfda_reid.models.memory_bank import ContrastiveMemoryBank
from sfda_reid.models.sfda_reid import SFDAReID
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
    parser = argparse.ArgumentParser(description='Source-Free Domain Adaptation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source_checkpoint', type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    logger = setup_logger('adapt', save_dir=cfg.target.output_dir)
    device = torch.device(cfg.device)
    # Build datasets
    target_train = get_dataset(cfg.target.dataset, cfg.target.data_root, 'train', get_test_transforms())
    query = get_dataset(cfg.target.dataset, cfg.target.data_root, 'query', get_test_transforms())
    gallery = get_dataset(cfg.target.dataset, cfg.target.data_root, 'gallery', get_test_transforms())
    # Build loaders
    target_loader = build_target_loader(target_train, cfg)
    query_loader = build_target_loader(query, cfg)
    gallery_loader = build_target_loader(gallery, cfg)
    # Build model
    if cfg.source.backbone == 'resnet50':
        backbone = ResNet50Backbone(num_classes=0, pretrained=cfg.source.pretrained)
        feat_dim = 2048
    else:
        backbone = ViTBackbone(num_classes=0, pretrained=cfg.source.pretrained)
        feat_dim = 384
    memory_bank = ContrastiveMemoryBank(num_features=feat_dim, bank_size=cfg.memory_bank.size, temperature=cfg.memory_bank.temperature, momentum=cfg.memory_bank.momentum)
    num_cameras = len(target_train.get_camera_ids())
    model = SFDAReID(backbone, memory_bank, num_cameras).to(device)
    # Load source checkpoint
    checkpoint = torch.load(args.source_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    # Trainer
    trainer = SFDATrainer(model, target_loader, query_loader, gallery_loader, cfg)
    trainer.run()
    logger.info('Adaptation complete.')

if __name__ == '__main__':
    main()
