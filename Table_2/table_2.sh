#!/bin/bash

# SD experiments
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_sd_seed_0 --seed 0 --sp 0.1 --adam --mode spw3
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_sd_seed_1 --seed 1 --sp 0.1 --adam --mode spw3
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_sd_seed_2 --seed 2 --sp 0.1 --adam --mode spw3
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_sd_seed_3 --seed 3 --sp 0.1 --adam --mode spw3
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_sd_seed_4 --seed 4 --sp 0.1 --adam --mode spw3

# Vanilla Cross-Entropy experiments
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_ce_seed_0 --seed 0
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_ce_seed_1 --seed 1
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_ce_seed_2 --seed 2
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_ce_seed_3 --seed 3
~/miniconda/bin/python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --show_progress --weight_decay 0.0 --data_dir $SLURM_TMPDIR --lr 1e-05 --log_dir ./logs_ce_seed_4 --seed 4