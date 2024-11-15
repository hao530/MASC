#!/bin/sh
EXP=exp1

CUDA_VISIBLE_DEVICES=0  python3 ./scripts/test.py \
    --img_dir=/root/autodl-tmp/nsrom11/data/JPEGImages/ \
    --test_list=./data/train_set.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --num_classes=17 \
    --restore_from=./runs/${EXP}/model/pascal_voc_epoch_14.pth \
    --save_dir=./runs/${EXP}/attention/ \
    --save_clustering_dir=./runs/${EXP}/clustering/ \
 
