#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1, python trainer_aaai.py --data-root /p300/datasets/Gaze/ShanghaiTechGaze+/data \
--batch-size-train 64 --batch-size-val 128 --num-workers 24 \
 - train_base --epochs 10 --lr 1e-2 --use-refined-depth False --fine-tune-headpose True\
 - train_base --epochs 15 --lr 1e-3 --use-refined-depth False --fine-tune-headpose True\
 - train_base --epochs 17 --lr 1e-4 --use-refined-depth False --fine-tune-headpose True\
 - end

