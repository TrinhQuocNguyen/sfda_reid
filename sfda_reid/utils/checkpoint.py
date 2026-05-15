import torch
import os
from typing import Any, Dict

def save_checkpoint(state: Dict[str, Any], is_best: bool, output_dir: str, filename: str = 'checkpoint.pth'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(output_dir, 'best.pth')
        torch.save(state, best_path)

def load_checkpoint(filepath: str, map_location: str = 'cpu') -> Dict[str, Any]:
    return torch.load(filepath, map_location=map_location)
