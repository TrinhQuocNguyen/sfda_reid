import os
import glob
from typing import List, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import torch

class MSMT17(Dataset):
    """
    MSMT17 Dataset Loader
    Expected folder structure:
        root/
            train/
            test/
            list_train.txt
            list_val.txt
            list_query.txt
            list_gallery.txt
    Modes: 'train', 'query', 'gallery'
    Returns dict: {"image": Tensor, "pid": int, "camid": int, "img_path": str}
    """
    def __init__(self, root: str, mode: str = 'train', transform=None):
        assert mode in ['train', 'query', 'gallery']
        self.root = root
        self.mode = mode
        self.transform = transform
        self.img_paths, self.pids, self.camids = self._parse_list()
        self.pid_set = set([pid for pid in self.pids if pid >= 0])
        self.camid_set = set(self.camids)
        
        # Build pid to label mapping (remap to [0, num_classes))
        unique_pids = sorted(self.pid_set)
        self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}

    def _parse_list(self):
        if self.mode == 'train':
            list_file = os.path.join(self.root, 'list_train.txt')
            img_dir = os.path.join(self.root, 'train')
        elif self.mode == 'gallery':
            list_file = os.path.join(self.root, 'list_gallery.txt')
            img_dir = os.path.join(self.root, 'test')
        else:
            list_file = os.path.join(self.root, 'list_query.txt')
            img_dir = os.path.join(self.root, 'test')
        img_paths, pids, camids = [], [], []
        with open(list_file, 'r') as f:
            for line in f:
                img_name, pid, camid = line.strip().split(' ')
                img_paths.append(os.path.join(img_dir, img_name))
                pids.append(int(pid))
                camids.append(int(camid))
        return img_paths, pids, camids

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        pid = self.pids[idx]
        # Remap pid to label for training
        label = self.pid2label.get(pid, pid) if pid >= 0 else pid
        return {
            'image': img,
            'pid': label,
            'camid': self.camids[idx],
            'img_path': self.img_paths[idx]
        }

    def get_camera_ids(self) -> List[int]:
        """Return list of unique camera IDs."""
        return sorted(list(self.camid_set))

    def get_pid_count(self) -> int:
        """Return number of unique person identities."""
        return len(self.pid_set)
