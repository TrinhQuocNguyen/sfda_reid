#!/bin/bash
python train.py --config configs/market2duke.yaml --phase source
python adapt.py --config configs/market2duke.yaml --source_checkpoint outputs/source_model/best.pth
python train.py --config configs/duke2market.yaml --phase source
python adapt.py --config configs/duke2market.yaml --source_checkpoint outputs/source_model/best.pth
python train.py --config configs/market2msmt.yaml --phase source
python adapt.py --config configs/market2msmt.yaml --source_checkpoint outputs/source_model/best.pth
python train.py --config configs/duke2msmt.yaml --phase source
python adapt.py --config configs/duke2msmt.yaml --source_checkpoint outputs/source_model/best.pth
