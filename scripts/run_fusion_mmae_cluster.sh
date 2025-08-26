#! /bin/bash

surfix="train.epochs=30 train.batch_size=512 train.data_root=/local/wding/Dataset/coco/images/" # General settings

python main_fusion_mmae.py train.lr=1e-3 $surfix

python main_fusion_mmae.py train.lr=1e-4 $surfix

python main_fusion_mmae.py train.lr=1e-5 $surfix