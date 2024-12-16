#!/bin/bash
#SBATCH -J haifangong 
#SBATCH -A P00120220004 
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
# python train_3d.py --model_name resunet --dg_method baseline --seed 1
# python train_3d.py --model_name resunet --dg_method baseline --seed 2
# python train_3d.py --model_name resunet --dg_method baseline --seed 3
# python train_3d.py --model_name resunet --dg_method bio --seed 1
# python train_3d.py --model_name resunet --dg_method bio --seed 2
# python train_3d.py --model_name resunet --dg_method bio --seed 3
# python train_3d.py --model_name resunet --dg_method segdg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_segdg_255/best_pretrain.pth --seed 1
# python train_3d.py --model_name resunet --dg_method segdg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_segdg_255/best_pretrain.pth --seed 2
# python train_3d.py --model_name resunet --dg_method segdg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_segdg_255/best_pretrain.pth --seed 3
# python train_3d.py --model_name resunet --dg_method edgedg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_edgedg_255/best_pretrain.pth --seed 1
# python train_3d.py --model_name resunet --dg_method edgedg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_edgedg_255/best_pretrain.pth --seed 2
# python train_3d.py --model_name resunet --dg_method edgedg --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_edgedg_255/best_pretrain.pth --seed 3
# python train_3d.py --model_name resunet --dg_method logcosh --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_robust_255/best_pretrain.pth --seed 1
# python train_3d.py --model_name resunet --dg_method logcosh --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_robust_255/best_pretrain.pth --seed 2
# python train_3d.py --model_name resunet --dg_method logcosh --model_weights /mntnfs/med_data5/gonghaifan/code4pretrain/runs/resunet_robust_255/best_pretrain.pth --seed 3
