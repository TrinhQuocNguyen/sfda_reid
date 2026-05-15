import os
import glob
from typing import List, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import torch

class DukeMTMC(Dataset):
    """
    DukeMTMC-ReID Dataset Loader
    Expected folder structure:
        root/
            bounding_box_train/
            bounding_box_test/
            query/
    Filenames: '0001_c1s1_000001_00.jpg' (pid, camid)
    Modes: 'train', 'query', 'gallery'
    Returns dict: {"image": Tensor, "pid": int, "camid": int, "img_path": str}
    """
    def __init__(self, root: str, mode: str = 'train', transform=None):
        assert mode in ['train', 'query', 'gallery']
        self.root = root
        self.mode = mode
        self.transform = transform
        self.img_paths = self._get_img_paths()
        self.samples = self._parse_samples()
        self.pid_set = set([s['pid'] for s in self.samples if s['pid'] >= 0])
        self.camid_set = set([s['camid'] for s in self.samples])
        
        # Build pid to label mapping (remap to [0, num_classes))
        unique_pids = sorted(self.pid_set)
        self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}

    def _get_img_paths(self) -> List[str]:
        if self.mode == 'train':
            folder = 'bounding_box_train'
        elif self.mode == 'gallery':
            folder = 'bounding_box_test'
        else:
            folder = 'query'
        return sorted(glob.glob(os.path.join(self.root, folder, '*.jpg')))

    def _parse_samples(self) -> List[Dict[str, Any]]:
        samples = []
        for img_path in self.img_paths:
            fname = os.path.basename(img_path)
            pid, camid = self._parse_filename(fname)
            samples.append({
                'img_path': img_path,
                'pid': pid,
                'camid': camid
            })
        return samples

    @staticmethod
    def _parse_filename(fname: str) -> (int, int):
        splits = fname.split('_')
        pid = int(splits[0])
        camid = int(splits[1][1:]) - 1  # c1s1 -> camid=0
        if pid == -1:
            pid = -1  # junk images
        elif pid == 0:
            pid = 0   # distractor images
        return pid, camid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        pid = sample['pid']
        # Remap pid to label for training
        label = self.pid2label.get(pid, pid) if pid >= 0 else pid
        return {
            'image': img,
            'pid': label,
            'camid': sample['camid'],
            'img_path': sample['img_path']
        }

    def get_camera_ids(self) -> List[int]:
        """Return list of unique camera IDs."""
        return sorted(list(self.camid_set))

    def get_pid_count(self) -> int:
        """Return number of unique person identities."""
        return len(self.pid_set)
