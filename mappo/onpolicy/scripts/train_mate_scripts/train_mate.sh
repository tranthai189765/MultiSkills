#!/bin/sh
env="mate"
algo="rmappo"
exp="check"

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mate.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    ---seed ${seed} --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 --diayn_alpha 0.5 --num_skills 8
done
