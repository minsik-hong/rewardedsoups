#!/bin/bash

# Run PPO training: ./run_ppo.sh 

TASK=review         # 첫 번째 인자가 없으면 기본값 review
DATASET=imdb        # 두 번째 인자가 없으면 기본값 imdb

echo "Starting PPO training for task: $TASK with dataset: $DATASET"

# 태스크별 리워드 모델 및 포맷 설정
if [ "$TASK" == "review" ]; then
    REWARD_MODELS="OpenAssistant/reward-model-deberta-v3-base OpenAssistant/reward-model-electra-large-discriminator"
    REWARD_FORMATS="0 0"
elif [ "$TASK" == "summary" ]; then
    REWARD_MODELS="Tristan/gpt2_reward_summarization CogComp/bart-faithful-summary-detector"
    REWARD_FORMATS="0 1-0"
elif [ "$TASK" == "assistant" ]; then
    REWARD_MODELS="OpenAssistant/reward-model-deberta-v3-large-v2 OpenAssistant/reward-model-electra-large-discriminator theblackcat102/reward-model-deberta-v3-base-v2 OpenAssistant/reward-model-deberta-v3-base"
    REWARD_FORMATS="0 0 0 0"
else
    # 기본값 (단일 모델)
    REWARD_MODELS="OpenAssistant/reward-model-deberta-v3-base"
    REWARD_FORMATS="0"
fi

echo "Using Reward Models: $REWARD_MODELS"

python llama/train_ppo.py \
    --task "$TASK" \
    --dataset_name "$DATASET" \
    --reward_models $REWARD_MODELS \
    --reward_formats $REWARD_FORMATS \
    --base_model_name PKU-Alignment/alpaca-8b-reproduced-llama-3 \
    --peft_name None \
    --learning_rate 1.41e-5 \
    --batch_size 1 \
    --mini_batch_size 1 \
    --log_with wandb
