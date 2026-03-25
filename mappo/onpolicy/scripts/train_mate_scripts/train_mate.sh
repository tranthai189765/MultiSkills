#!/bin/sh

export OMP_NUM_THREADS=1

env="MATE"
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"

    CUDA_VISIBLE_DEVICES=0 python ../train/train_mate.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 96 \
    --num_mini_batch 1 \
    --episode_length 900 \
    --levels 11 \
    --num_env_steps 100000000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --use_eval \
    --eval_episodes 32 \
    --diayn_alpha 0.5 \
    --num_skills 8 \
    --use_wandb False
done