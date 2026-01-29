#!/bin/bash
set -x
export LLM_AS_A_JUDGE_BASE="http://127.0.0.1:18901/v1"
python -m verl.trainer.main_ppo \
    --config-path /path/to/DeepEyes/config \
    --config-name deepeyes_coz  2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
