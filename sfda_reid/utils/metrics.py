import numpy as np
from typing import List

def compute_map(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (np.array(g_pids)[indices] == np.array(q_pids)[:, np.newaxis])
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
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * orig_cmc).sum() / num_rel
        all_AP.append(AP)
    return np.mean(all_AP) if all_AP else 0.0

def compute_cmc(distmat, q_pids, g_pids, q_camids, g_camids, ranks):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (np.array(g_pids)[indices] == np.array(q_pids)[:, np.newaxis])
    cmc = np.zeros(max(ranks))
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
        if cmc_idx < max(ranks):
            cmc[cmc_idx:] += 1
    cmc = cmc / num_q
    return cmc[ranks[0]-1:ranks[-1]]

def compute_pseudo_label_accuracy(pred_labels, gt_labels):
    return (np.array(pred_labels) == np.array(gt_labels)).mean()

def compute_noise_rate(pseudo_labels, true_labels):
    return (np.array(pseudo_labels) != np.array(true_labels)).mean()
