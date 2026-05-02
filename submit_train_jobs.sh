#!/bin/bash
sbatch slurm_train_splicevi.sh "concatenate"
sbatch slurm_train_splicevi.sh "sum"
sbatch slurm_train_splicevi.sh "product"
sbatch slurm_train_splicevi.sh "cross_attention"
sbatch slurm_train_splicevi.sh "cross_attention_reverse"
sbatch slurm_train_splicevi.sh "gating"
sbatch slurm_train_splicevi.sh "mlp"