#!/bin/bash

# 设置参数
LR=0.001
WD=0.0001
BATCH_SIZE=4
ITERS=12800
# FT_CLIP="all_ft"
FT_CLIP="fronze_text"
# FT_CLIP="fronze_all"

# 运行 Python 脚本并传递参数
python3 train.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
# python3 train_res.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
# python3 train_temp.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
# python3 train_no_text.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
# python3 train_only_text.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
# python3 train_two_stream.py --lr $LR --wd $WD --batch_size $BATCH_SIZE --iters $ITERS --ft_clip $FT_CLIP
