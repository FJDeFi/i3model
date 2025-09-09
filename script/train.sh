#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python ../clinfonce.py --epochs 200 --save_folder cl_infonce_imagenet --warmup_epoch 5 --batch-size-pergpu 32
