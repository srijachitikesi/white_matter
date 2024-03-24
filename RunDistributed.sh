#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=40g
#SBATCH --gres=gpu:V100:1
#SBATCH -p qTRDGPUH
#SBATCH -J ConvLr00001
#SBATCH -e error%A.err
#SBATCH -o gpu4_%A.txt
#SBATCH -A trends396s109
#SBATCH -t 1-05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijachitikesi01@gmail.com
#SBATCH --oversubscribe

/home/users/schitikesi1/miniconda3/bin/python3 crossval_pred_test.py