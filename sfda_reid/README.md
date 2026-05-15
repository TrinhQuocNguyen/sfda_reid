# Source-Free Domain Adaptive Person Re-Identification with Formal Learnability Guarantees

## Abstract
This repository provides a complete, modular, and reproducible implementation of the method described in the paper:

**"Source-Free Domain Adaptive Person Re-Identification with Formal Learnability Guarantees"** (to appear in JAIR).

We present a source-free domain adaptation framework for person re-identification (ReID) that leverages formal generalization bounds. Our approach combines memory bank-based contrastive learning, camera-aware clustering, and theoretical analysis to achieve state-of-the-art performance on cross-domain ReID benchmarks, without access to source data during adaptation.

---

## Installation

```bash
pip install -e .
```

---

## Dataset Preparation

Download and extract the datasets:
- [Market-1501](https://github.com/zhunzhong07/Market-1501)
- [DukeMTMC-ReID](https://github.com/layumi/DukeMTMC-reID_devkit)
- [MSMT17](https://github.com/zhunzhong07/MSMT17)

Expected folder structure:
```
data/
  raw/
    market1501/
      bounding_box_train/
      bounding_box_test/
      query/
    dukemtmc/
      bounding_box_train/
      bounding_box_test/
      query/
    msmt17/
      train/
      test/
      list_train.txt
      list_val.txt
      list_query.txt
      list_gallery.txt
```

---

## Training the Source Model

```bash
python train.py --config configs/market2duke.yaml --phase source
```

Checkpoints are saved to `outputs/source_model/`.

---

## Source-Free Adaptation

```bash
python adapt.py --config configs/market2duke.yaml --source_checkpoint outputs/source_model/best.pth
```

Adapted model checkpoints are saved to `outputs/adapted_model/`.

---

## Reproducing Cross-Dataset Benchmarks

Run all four benchmarks:

```bash
bash scripts/run_all_benchmarks.sh
```

Or individually:

- Market1501 → DukeMTMC:
  ```bash
  python train.py --config configs/market2duke.yaml --phase source
  python adapt.py --config configs/market2duke.yaml --source_checkpoint outputs/source_model/best.pth
  ```
- DukeMTMC → Market1501:
  ```bash
  python train.py --config configs/duke2market.yaml --phase source
  python adapt.py --config configs/duke2market.yaml --source_checkpoint outputs/source_model/best.pth
  ```
- Market1501 → MSMT17:
  ```bash
  python train.py --config configs/market2msmt.yaml --phase source
  python adapt.py --config configs/market2msmt.yaml --source_checkpoint outputs/source_model/best.pth
  ```
- DukeMTMC → MSMT17:
  ```bash
  python train.py --config configs/duke2msmt.yaml --phase source
  python adapt.py --config configs/duke2msmt.yaml --source_checkpoint outputs/source_model/best.pth
  ```

---

## Theoretical Bound Validation Experiment

```bash
python experiments/bound_validation.py --config configs/market2duke.yaml --checkpoint outputs/adapted_model/best.pth --experiment a  # or b
```

Results and plots are saved to `outputs/bound_validation/`.

---

## Ablation Study

```bash
python experiments/ablation_study.py --config configs/market2duke.yaml
```

Results are saved as CSV and PDF in `outputs/ablation/`.

---

## Expected Results (Table 1)

| Benchmark           | mAP (%) | Rank-1 (%) |
|---------------------|---------|------------|
| Market→Duke         | 68.2    | 83.7       |
| Duke→Market         | 74.5    | 89.1       |
| Market→MSMT         | 32.8    | 58.4       |
| Duke→MSMT           | 30.2    | 54.7       |

---

## Citation

```bibtex
@article{sfda_reid2026,
  title={Source-Free Domain Adaptive Person Re-Identification with Formal Learnability Guarantees},
  author={Author, A. and Collaborator, B.},
  journal={Journal of Artificial Intelligence Research},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
