#!/bin/bash

# ============= Collecting expert demonstrations =============
# You will need to set the following parameters to their appropriate values before starting the data collection process: 
#     dataset_root = [ABSOLUTE PATH TO DATA FOLDER]
#     agent.ppo.wb_run_path = [WANDB RUN ID OF TRAINED RL MODEL]
# NOTE: To use the pretrained RL model from the original Roach paper:
# agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0
data_collect () {
python -u data_collect.py resume=true log_video=true save_to_wandb=false \
wb_project=carla_obs_av_data \
wb_group=bc_data \
test_suites=static_data \
n_episodes=30 \
dataset_root=/home/diaz0329/REU/carla-roach/dataset/bc_bboxes_lanenet/ \
actors.hero.driver=ppo \
agent.ppo.wb_run_path=diaz0329/train_rl_route_follow/lkh5t85f \
agent.ppo.wb_ckpt_step=null \
inject_noise=true \
actors.hero.terminal.kwargs.max_time=100 \
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
  data_collect
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."
