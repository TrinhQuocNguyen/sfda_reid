#!/usr/bin/env python3
"""
Quick test to verify dataloader and evaluator fixes work correctly.
"""
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from omegaconf import OmegaConf
    from sfda_reid.data.dataloader import build_train_loader, build_test_loader
    from sfda_reid.data.datasets.market1501 import Market1501
    from sfda_reid.data.transforms import get_test_transforms, get_train_transforms

    # Load config
    cfg = OmegaConf.load('sfda_reid/configs/market2duke.yaml')
    cfg = OmegaConf.merge(OmegaConf.load('sfda_reid/configs/base.yaml'), cfg)
    
    # Reduce num_workers for testing
    cfg.num_workers = 0

    print("=" * 80)
    print("Testing Dataloader Fixes")
    print("=" * 80)

    # Test train loader
    print("\n[1] Testing build_train_loader...")
    try:
        train_set = Market1501(cfg.source.data_root, mode='train', transform=get_train_transforms())
        train_loader = build_train_loader(train_set, cfg)
        
        # Get first batch
        batch = next(iter(train_loader))
        print(f"✓ First batch retrieved successfully")
        print(f"  - images shape: {batch['image'].shape}")
        print(f"  - pids shape: {batch['pid'].shape}, type: {batch['pid'].dtype}")
        print(f"  - camids shape: {batch['camid'].shape}, type: {batch['camid'].dtype}")
        print(f"  - num_img_paths: {len(batch['img_path'])}")
        
        assert isinstance(batch, dict), "Batch should be a dict"
        assert batch['image'].dim() == 4, "Images should be 4D tensor"
        assert batch['pid'].dim() == 1, "PIDs should be 1D tensor"
        assert batch['camid'].dim() == 1, "CAMIDs should be 1D tensor"
        print("✓ Train loader format is correct\n")
    except Exception as e:
        print(f"✗ Train loader test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test test loader
    print("[2] Testing build_test_loader...")
    try:
        val_set = Market1501(cfg.source.data_root, mode='gallery', transform=get_test_transforms())
        val_loader = build_test_loader(val_set, cfg)
        
        # Get first batch
        batch = next(iter(val_loader))
        print(f"✓ First batch retrieved successfully")
        print(f"  - images shape: {batch['image'].shape}")
        print(f"  - pids shape: {batch['pid'].shape}, type: {batch['pid'].dtype}")
        print(f"  - camids shape: {batch['camid'].shape}, type: {batch['camid'].dtype}")
        print(f"  - num_img_paths: {len(batch['img_path'])}")
        
        assert isinstance(batch, dict), "Batch should be a dict"
        assert batch['image'].dim() == 4, "Images should be 4D tensor"
        assert batch['pid'].dim() == 1, "PIDs should be 1D tensor"
        assert batch['camid'].dim() == 1, "CAMIDs should be 1D tensor"
        print("✓ Test loader format is correct\n")
    except Exception as e:
        print(f"✗ Test loader test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test evaluator with mock model
    print("[3] Testing evaluator device handling...")
    try:
        from sfda_reid.engine.evaluator import ReIDEvaluator
        
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self, device='cuda'):
                super().__init__()
                self.fc = torch.nn.Linear(2048, 100).to(device)
                self._device = device
                
            def forward_inference(self, images):
                # Just return random features for testing
                batch_size = images.shape[0]
                return torch.randn(batch_size, 2048, device=self._device)
        
        device = torch.device(cfg.device)
        model = MockModel(device=device)
        
        # Get small loader for testing
        query_set = Market1501(cfg.source.data_root, mode='query', transform=get_test_transforms())
        query_loader = build_test_loader(query_set, cfg)
        
        # Create evaluator
        evaluator = ReIDEvaluator()
        
        print(f"  - Model device: {device}")
        print(f"  - Query loader batch size: 256")
        
        # Test extracting features
        features, pids, camids = evaluator._extract_features(model, query_loader)
        print(f"✓ Features extracted successfully")
        print(f"  - features shape: {features.shape}")
        print(f"  - num_pids: {len(pids)}")
        print(f"  - num_camids: {len(camids)}")
        
        assert features.device.type == 'cpu', "Features should be on CPU"
        assert isinstance(pids, list), "PIDs should be a list"
        assert isinstance(camids, list), "CAMIDs should be a list"
        print("✓ Evaluator device handling is correct\n")
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nYou can now run the full training:")
    print("  python train.py --config configs/market2duke.yaml --phase source")

