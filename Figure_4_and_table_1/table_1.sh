#!/bin/bash

python -u cmnist.py \
  --exp erm \
  --hidden_dim=300 \
  --l2_regularizer_weight=0.001 \
  --lr=0.0001 \
  --penalty_anneal_iters=0 \
  --sd_coef 0.0 \
  --penalty_weight=0 \
  --steps=1001 \
  --n_restarts 10

python -u cmnist.py \
    --exp sd \
    --hidden_dim=300 \
    --l2_regularizer_weight=0.0 \
    --lr=0.0001 \
    --penalty_anneal_iters=450 \
    --sd_coef 0.00002 \
    --penalty_weight=0 \
    --steps=2001 \
    --n_restarts 10

# hyperparameters from IRM paper
python -u cmnist.py \
  --exp irm \
  --hidden_dim=390 \
  --l2_regularizer_weight=0.00110794568 \
  --lr=0.0004898536566546834 \
  --penalty_anneal_iters=190 \
  --penalty_weight=91257.18613115903 \
  --steps=501 \
  --n_restarts 10

# ERM:     
# Average train accuracy over 10 runs: 91.08 % (pm 0.43)
# Average test accuracy over 10 runs: 23.70 % (pm 0.76)

# SD:
# Average train accuracy over 10 runs: 70.04 % (pm 0.91)
# Average test accuracy over 10 runs: 68.40 % (pm 1.24)

# IRM:
# Average train accuracy over 10 runs: 70.54 % (pm 0.67)
# Average test accuracy over 10 runs: 67.08 % (pm 1.47)