#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/AIProjects/trinh/Projects/sfda_reid')

from sfda_reid.data.datasets.market1501 import Market1501
import numpy as np

# Load training dataset
dataset = Market1501('/old/home/ccvn/Workspace/trinh/data/reidMarket1501', mode='train')
pids = [s['pid'] for s in dataset.samples]

print(f"Total samples: {len(pids)}")
print(f"Unique pids: {len(set(pids))}")
print(f"Min pid: {min(pids)}")
print(f"Max pid: {max(pids)}")
print(f"Pid range: {min(pids)} to {max(pids)}")

# Check pid distribution
unique_pids = sorted(set(pids))
print(f"\nFirst 10 unique pids: {unique_pids[:10]}")
print(f"Last 10 unique pids: {unique_pids[-10:]}")

# Check if pids are consecutive
pids_array = np.array(unique_pids)
gaps = np.diff(pids_array)
if max(gaps) > 1:
    print(f"\nWarning: Non-consecutive pids detected! Max gap: {max(gaps)}")
    print(f"Pids need to be remapped to [0, {len(unique_pids)-1}]")
else:
    print(f"\nPids are consecutive")
