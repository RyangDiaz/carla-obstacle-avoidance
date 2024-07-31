#!/bin/bash

# # NOTE: Comment the below function out if you are training with DAgger (i.e. not training iteration 0)
# # ============= Training visuomotor agent with imitation learning (training iteration 0) =============
# # You will need to set the following parameters to their appropriate values before starting the training process: 
# #     dagger_datasets = [[ABSOLUTE PATH TO EXPERT DATA FOLDER]]
# #     agent.clirs_bbox.rl_run_path = [WANDB RUN ID OF TRAINED RL MODEL]
# #     agent.cilrs_bbox.rl_ckpt_step = [SPECIFIC SAVED CKPT STEP FROM RL TRAINING]
# # NOTE: To use the pretrained RL model from the original Roach paper:
# # agent.cilrs_bbox.rl_run_path=iccv21-roach/trained-models/1929isj0
# train_il () {
# python -u train_il.py reset_step=true \
# agent.cilrs_bbox.wb_run_path=null agent.cilrs_bbox.wb_ckpt_step=null \
# wb_project="simple_env_obs_av" wb_group="train0" 'wb_name="L1_action_loss"' \
# dagger_datasets=["/home/diaz0329/REU/carla-roach/dataset/bc_bboxes_lanenet/expert"] \
# 'agent.cilrs_bbox.env_wrapper.kwargs.input_states=[speed]' \
# agent.cilrs_bbox.policy.kwargs.number_of_branches=8 \
# agent.cilrs_bbox.training.kwargs.branch_weights=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] \
# agent.cilrs_bbox.env_wrapper.kwargs.action_distribution="beta_shared" \
# agent.cilrs_bbox.rl_run_path=diaz0329/train_rl_route_follow/lkh5t85f agent.cilrs_bbox.rl_ckpt_step=11943936 \
# agent.cilrs_bbox.training.kwargs.action_kl=true \
# agent.cilrs_bbox.env_wrapper.kwargs.value_as_supervision=true \
# agent.cilrs_bbox.training.kwargs.value_weight=0.001 \
# agent.cilrs_bbox.env_wrapper.kwargs.dim_features_supervision=256 \
# agent.cilrs_bbox.training.kwargs.features_weight=0.05 \
# agent.cilrs_bbox.training.kwargs.batch_size=64 \
# cache_dir=${CACHE_DIR}
# }

# NOTE: Comment the below function out if you are on training iteration 0 (behavior cloning)
# ============= Training visuomotor agent further with imitation learning (DAgger -> training iteration 1+) =============
# You will need to set the following parameters to their appropriate values before starting the data collection process: 
#     dagger_datasets = [[ABSOLUTE PATH TO DAGGER DATA FOLDER(S)], [ABSOLUTE PATH TO EXPERT DATA FOLDER]] # NOTE: Make sure to put the expert dataset LAST in the list
#     agent.clirs_bbox.wb_run_path = [WANDB RUN ID OF TRAINED IL MODEL]
# NOTE: Be sure to update the datasets to the new collected dataset and wb_run_path to the new IL model for every iteration of DAgger training
train_il () {
python -u train_il.py reset_step=true \
agent.cilrs_bbox.wb_run_path=diaz0329/simple_env_obs_av/149b7elx agent.cilrs_bbox.wb_ckpt_step=24 \
wb_project="simple_env_obs_av" wb_group="train1" wb_name='"L_all_dagger"' \
dagger_datasets=["/home/diaz0329/REU/carla-roach/dataset/bc_bboxes_lanenet/simple_env_obs_av/149b7elx","/home/diaz0329/REU/carla-roach/dataset/bc_bboxes_lanenet/expert"] \
cache_dir=${CACHE_DIR}
}

NODE_ROOT=/home/diaz0329/REU/carla-roach/dataset/tmp_data
mkdir -p "${NODE_ROOT}"
CACHE_DIR=$(mktemp -d --tmpdir="${NODE_ROOT}")

echo "CACHE_DIR: ${CACHE_DIR}"

train_il

echo "Python finished!!"
rm -rf "${CACHE_DIR}"
echo "Bash script done!!"
echo finished at: `date`
exit 0;
