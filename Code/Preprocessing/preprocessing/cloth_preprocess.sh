#!/bin/bash

BASE_DIR="/home/user/stableVTON"
DATA_DIR="$BASE_DIR/StableVITON/DATA/zalando-hd-resized/inference"
python3 cloth_mask.py --data_path "$DATA_DIR/"
