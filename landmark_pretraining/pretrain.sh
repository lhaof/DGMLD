#!/bin/bash
#SBATCH -J haifangong 
#SBATCH -A P00120220004 
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
python pretrain.py --model_name resunet --max_dist 0
python pretrain.py --model_name resunet --max_dist -1
python pretrain.py --model_name resunet --max_dist 5 --task 'robust'

python pretrain.py --model_name vitpose --max_dist 0
python pretrain.py --model_name vitpose --max_dist -1
python pretrain.py --model_name vitpose --max_dist 5 --task 'robust'