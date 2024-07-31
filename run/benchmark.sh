#!/bin/bash

# NOTE: Make sure to comment out the benchmark function of the agent that you are NOT benchmarking

# # ============= Evaluating trained reinforcement learning agent =============
# # You will need to set the following parameters to their appropriate values before starting the evaluation process: 
# #     agent.ppo.wb_run_path = [WANDB RUN ID OF TRAINED RL MODEL]
# # NOTE: To use the pretrained RL model from the original Roach paper:
# # agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0
# agent="ppo"
# benchmark () {
#   python -u benchmark.py resume=true log_video=true \
#   wb_project=obs_av_benchmark \
#   agent=$agent actors.hero.agent=$agent \
#   agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0 \
#   'wb_group="Roach"' \
#   'wb_notes="Benchmark Roach on NoCrash-dense."' \
#   test_suites=static_test \
#   seed=2021 \
#   +wb_sub_group=nocrash_dense \
#   no_rendering=true \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }

# ============= Evaluating trained imitation learning agent =============
# You will need to set the following parameters to their appropriate values before starting the evaluation process: 
#     agent.cilrs_bbox.wb_run_path = [WANDB RUN ID OF TRAINED IL MODEL]
agent="cilrs_bbox"
benchmark () {
  python -u benchmark.py resume=true log_video=true \
  wb_project=obs_av_benchmark \
  agent=$agent actors.hero.agent=$agent \
  agent.cilrs_bbox.wb_run_path=diaz0329/simple_env_obs_av/149b7elx \
  'wb_group="L_K+L_F(c)"' \
  'wb_notes="Benchmark L_K+L_F(c) on NoCrash-dense."' \
  test_suites=simple_route \
  seed=2021 \
  +wb_sub_group=nocrash_dense-2021 \
  no_rendering=true \
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
  benchmark
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."
