#!/bin/bash

# ============= Collecting additional demonstrations for DAgger training =============
# You will need to set the following parameters to their appropriate values before starting the data collection process: 
#     dataset_root = [ABSOLUTE PATH TO DATA FOLDER] (should be the same as the folder where the expert demos are)
#     agent.cilrs_bbox.wb_run_path = [WANDB RUN ID OF TRAINED IL MODEL]
#     agent.ppo.wb_run_path = [WANDB RUN ID OF TRAINED RL MODEL] (should be the same model you collected expert demos with)
# NOTE: To use the pretrained RL model from the original Roach paper:
# agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0
data_collect_dagger () {
python -u data_collect.py resume=true log_video=true save_to_wandb=false \
agent.cilrs_bbox.wb_run_path=diaz0329/simple_env_obs_av/149b7elx \
wb_group=dagger5 \
test_suites=static_data \
dataset_root=/home/diaz0329/REU/carla-roach/dataset/bc_bboxes_lanenet/ \
actors.hero.coach=ppo \
agent.ppo.wb_run_path=diaz0329/train_rl_route_follow/lkh5t85f \
agent.ppo.wb_ckpt_step=11943936 \
actors.hero.driver=cilrs_bbox \
n_episodes=6 inject_noise=false \
dagger_thresholds.acc=0.2 \
remove_final_steps=false \
actors.hero.terminal.kwargs.max_time=300 \
actors.hero.terminal.kwargs.no_collision=true \
actors.hero.terminal.kwargs.no_run_rl=false \
actors.hero.terminal.kwargs.no_run_stop=false \
carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}

# remove checkpoint files
rm outputs/checkpoint.txt
rm outputs/wb_run_id.txt
rm outputs/ep_stat_buffer_*.json


# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  data_collect_dagger
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done
