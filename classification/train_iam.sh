#!/bin/sh
EXP=exp2

CUDA_VISIBLE_DEVICES=0 python3 ./scripts/train_iam.py \
    --img_dir=/root/autodl-tmp/nsrom11/data/JPEGImages/ \
    --train_list=./data/train_set.txt \
    --test_list=./data/test_set.txt \
    --epoch=15 \
    --lr=0.001 \
    --batch_size=5 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --disp_interval=100 \
	  --num_classes=17 \
	  --num_workers=8 \
	  --snapshot_dir=./runs/${EXP}/model/  \
    --att_dir=./runs/exp1/accu_att/ \
    --decay_points='5,10'
