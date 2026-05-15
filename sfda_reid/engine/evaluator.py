import torch
import numpy as np
from typing import Dict, Tuple, List

class ReIDEvaluator:
    def evaluate(self, model: torch.nn.Module, query_loader, gallery_loader, use_reranking: bool = False) -> Dict[str, float]:
        """
        Standard ReID evaluation protocol.
        1. Extract query and gallery features
        2. Compute pairwise Euclidean distance matrix
        3. For each query, rank gallery by distance
        4. Compute mAP and CMC at ranks [1, 5, 10, 20]
        5. Optionally apply k-reciprocal re-ranking (Zhong et al.)
        Returns evaluation metrics dict.
        """
        qf, q_pids, q_camids = self._extract_features(model, query_loader)
        gf, g_pids, g_camids = self._extract_features(model, gallery_loader)
        distmat = self._compute_distance_matrix(qf, gf)
        if use_reranking:
            # Placeholder: reranking not implemented
            pass
        cmc, mAP = self._eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20)
        return {'mAP': mAP, 'rank1': cmc[0], 'rank5': cmc[4], 'rank10': cmc[9]}

    def _extract_features(self, model: torch.nn.Module, loader) -> Tuple[torch.Tensor, List, List]:
        features, pids, camids = [], [], []
        model.eval()
        
        # Get model device
        model_device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                # Move images to model's device
                images = images.to(model_device)
                feats = model.forward_inference(images)
                features.append(feats.cpu())
                
                # Handle pids and camids - convert tensors to list if needed
                pids_batch = batch['pid'] if isinstance(batch, dict) else batch[1]
                camids_batch = batch['camid'] if isinstance(batch, dict) else batch[2]
                
                if isinstance(pids_batch, torch.Tensor):
                    pids.extend(pids_batch.cpu().numpy().tolist())
                else:
                    pids.extend(pids_batch if isinstance(pids_batch, list) else [pids_batch])
                
                if isinstance(camids_batch, torch.Tensor):
                    camids.extend(camids_batch.cpu().numpy().tolist())
                else:
                    camids.extend(camids_batch if isinstance(camids_batch, list) else [camids_batch])
        
        features = torch.cat(features, dim=0)
        return features, pids, camids

    def _compute_distance_matrix(self, qf: torch.Tensor, gf: torch.Tensor) -> torch.Tensor:
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        return distmat.cpu().numpy()

    def _eval_func(self, distmat: np.ndarray, q_pids: List, g_pids: List, q_camids: List, g_camids: List, max_rank: int = 20) -> Tuple[np.ndarray, float]:
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (np.array(g_pids)[indices] == np.array(q_pids)[:, np.newaxis])
        cmc = np.zeros(max_rank)
        all_AP = []
        for i in range(num_q):
            q_pid = q_pids[i]
            q_camid = q_camids[i]
            order = indices[i]
            remove = (np.array(g_pids)[order] == q_pid) & (np.array(g_camids)[order] == q_camid)
            keep = np.invert(remove)
            orig_cmc = matches[i, keep]
            if not np.any(orig_cmc):
                continue
            cmc_idx = np.where(orig_cmc)[0][0]
            if cmc_idx < max_rank:
                cmc[cmc_idx:] += 1
            # mAP
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
            AP = (precision * orig_cmc).sum() / num_rel
            all_AP.append(AP)
        cmc = cmc / num_q
        mAP = np.mean(all_AP) if all_AP else 0.0
        return cmc, mAP
