#!/bin/bash

# 设置参数
LR=1e-8
WD=0.001
BATCH_SIZE=4
ITERS=12800

python3 clip_ft.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS