import torch
from torch.utils.data import DataLoader, Sampler
from typing import Any, List, Dict
import random
import numpy as np

def _train_collate_fn(batch):
    """Collate function for training data that returns dict format."""
    images = torch.stack([item['image'] for item in batch])
    pids = torch.tensor([item['pid'] for item in batch], dtype=torch.long)
    camids = torch.tensor([item['camid'] for item in batch], dtype=torch.long)
    img_paths = [item.get('img_path', '') for item in batch]
    return {
        'image': images,
        'pid': pids,
        'camid': camids,
        'img_path': img_paths
    }

def _test_collate_fn(batch):
    """Collate function for test/query/gallery data that returns dict format."""
    images = torch.stack([item['image'] for item in batch])
    pids = torch.tensor([item['pid'] for item in batch], dtype=torch.long)
    camids = torch.tensor([item['camid'] for item in batch], dtype=torch.long)
    img_paths = [item.get('img_path', '') for item in batch]
    return {
        'image': images,
        'pid': pids,
        'camid': camids,
        'img_path': img_paths
    }

def _target_collate_fn(batch):
    """Collate function for unlabeled target data."""
    images = torch.stack([item['image'] for item in batch])
    camids = torch.tensor([item['camid'] for item in batch], dtype=torch.long)
    indices = torch.arange(len(batch), dtype=torch.long)
    return images, camids, indices

class RandomIdentitySampler(Sampler):
    """
    Randomly samples P identities, then K instances for each identity.
    """
    def __init__(self, data_source, num_pids: int = 16, num_instances: int = 4):
        self.data_source = data_source
        self.num_pids = num_pids
        self.num_instances = num_instances
        self.index_dic = self._build_index_dic()
        self.pids = list(self.index_dic.keys())
        self.length = self._calculate_length()

    def _build_index_dic(self) -> Dict[int, List[int]]:
        index_dic = {}
        for idx, sample in enumerate(self.data_source.samples):
            pid = sample['pid']
            # Skip invalid pids (junk=-1, distractor=0)
            if pid < 1:
                continue
            if pid not in index_dic:
                index_dic[pid] = []
            index_dic[pid].append(idx)
        return index_dic

    def _calculate_length(self) -> int:
        return sum([
            len(indexes) - len(indexes) % self.num_instances
            for indexes in self.index_dic.values()
        ])

    def __iter__(self):
        batch_idxs = []
        pids = self.pids.copy()
        random.shuffle(pids)
        for pid in pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.num_instances:
                continue
            idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
            batch_idxs.extend(idxs)
        return iter(batch_idxs)

    def __len__(self):
        return self.length

def build_train_loader(dataset, cfg) -> DataLoader:
    """
    Build DataLoader for training with RandomIdentitySampler.
    """
    sampler = RandomIdentitySampler(dataset, num_pids=16, num_instances=4)
    mp_context = 'spawn' if cfg.num_workers > 0 else None
    return DataLoader(dataset, batch_size=cfg.source.batch_size, sampler=sampler, num_workers=cfg.num_workers, 
                      drop_last=True, collate_fn=_train_collate_fn, multiprocessing_context=mp_context)

def build_test_loader(dataset, cfg) -> DataLoader:
    """
    Build DataLoader for test/query/gallery with sequential sampling.
    """
    mp_context = 'spawn' if cfg.num_workers > 0 else None
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers, 
                      collate_fn=_test_collate_fn, multiprocessing_context=mp_context)

def build_target_loader(dataset, cfg) -> DataLoader:
    """
    Build DataLoader for unlabeled target training images.
    Returns image, camera ID, image index.
    """
    mp_context = 'spawn' if cfg.num_workers > 0 else None
    return DataLoader(dataset, batch_size=cfg.target.batch_size, shuffle=True, num_workers=cfg.num_workers, 
                      collate_fn=_target_collate_fn, multiprocessing_context=mp_context)
