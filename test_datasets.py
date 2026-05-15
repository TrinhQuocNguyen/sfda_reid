#!/usr/bin/env python3
"""Test script to verify dataset implementations"""
import sys
sys.path.insert(0, '/mnt/AIProjects/trinh/Projects/sfda_reid')

from sfda_reid.data.datasets.market1501 import Market1501
from sfda_reid.data.datasets.dukemtmc import DukeMTMC
from sfda_reid.data.datasets.msmt17 import MSMT17
from sfda_reid.data.transforms import get_train_transforms, get_test_transforms
import torch

def test_dataset(dataset_class, root, name):
    """Test a dataset implementation"""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Test training split
    print(f"\n--- Training Split ---")
    train_set = dataset_class(root, mode='train', transform=get_train_transforms())
    print(f"Total samples: {len(train_set)}")
    print(f"Unique pids: {len(train_set.pid_set)}")
    print(f"Min raw pid: {min(train_set.pid_set)}")
    print(f"Max raw pid: {max(train_set.pid_set)}")
    print(f"Unique cameras: {len(train_set.camid_set)}")
    
    # Test pid2label mapping
    print(f"\nPID-to-Label Mapping Check:")
    print(f"  Mapping size: {len(train_set.pid2label)}")
    min_label = min(train_set.pid2label.values())
    max_label = max(train_set.pid2label.values())
    print(f"  Min label: {min_label}")
    print(f"  Max label: {max_label}")
    print(f"  Labels are consecutive: {max_label == len(train_set.pid2label) - 1}")
    
    # Sample a few items
    print(f"\nSampling 5 training items:")
    for i in range(min(5, len(train_set))):
        sample = train_set[i]
        assert 'image' in sample, "Missing 'image' key"
        assert 'pid' in sample, "Missing 'pid' key"
        assert 'camid' in sample, "Missing 'camid' key"
        assert 'img_path' in sample, "Missing 'img_path' key"
        
        pid = sample['pid']
        assert isinstance(pid, (int, torch.Tensor)), f"PID must be int or tensor, got {type(pid)}"
        
        # Convert to int if tensor
        if isinstance(pid, torch.Tensor):
            pid = pid.item()
        
        assert 0 <= pid < len(train_set.pid_set), f"Label {pid} out of range [0, {len(train_set.pid_set)-1}]"
        assert isinstance(sample['image'], torch.Tensor), "Image must be a tensor"
        assert sample['image'].shape[0] == 3, f"Image must have 3 channels, got {sample['image'].shape[0]}"
        
        print(f"  Sample {i}: label={pid}, camid={sample['camid']}, img_shape={sample['image'].shape}")
    
    # Test gallery/query splits
    print(f"\n--- Gallery Split ---")
    gallery_set = dataset_class(root, mode='gallery', transform=get_test_transforms())
    print(f"Total samples: {len(gallery_set)}")
    print(f"Unique pids: {len(gallery_set.pid_set)}")
    
    print(f"\n--- Query Split ---")
    query_set = dataset_class(root, mode='query', transform=get_test_transforms())
    print(f"Total samples: {len(query_set)}")
    print(f"Unique pids: {len(query_set.pid_set)}")
    
    # Verify labels are valid
    print(f"\nVerifying all labels in gallery split...")
    invalid_count = 0
    for i in range(len(gallery_set)):
        sample = gallery_set[i]
        pid = sample['pid']
        if isinstance(pid, torch.Tensor):
            pid = pid.item()
        if pid < 0 or pid >= len(gallery_set.pid_set):
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  Invalid label at index {i}: {pid}")
    
    if invalid_count == 0:
        print(f"✓ All labels are valid in gallery split")
    else:
        print(f"✗ Found {invalid_count} invalid labels in gallery split")
    
    print(f"\n✓ {name} dataset test PASSED")


def main():
    """Run all dataset tests"""
    print("Dataset Testing Suite")
    print("=" * 60)
    
    # Market1501
    try:
        test_dataset(Market1501, '/old/home/ccvn/Workspace/trinh/data/reidMarket1501', 'Market1501')
    except Exception as e:
        print(f"✗ Market1501 test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # DukeMTMC
    try:
        test_dataset(DukeMTMC, '/old/home/ccvn/Workspace/trinh/data/reidDukeMTMC-reID', 'DukeMTMC')
    except Exception as e:
        print(f"✗ DukeMTMC test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # MSMT17
    try:
        test_dataset(MSMT17, '/old/home/ccvn/Workspace/trinh/data/reidMSMT17', 'MSMT17')
    except Exception as e:
        print(f"✗ MSMT17 test FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All dataset tests completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
