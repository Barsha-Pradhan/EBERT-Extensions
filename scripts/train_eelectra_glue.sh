#!/bin/bash
export WANDB_PROJECT="EBERT-lab"
export WANDB_DISABLED="true"

TASK_NAME=SST-2
RUN_NAME=electra_sst_2_s0.8

BATCH_SIZE=16
EPOCHS=5.0
WARMUP_STEPS=$(python scripts/compute_warmup_steps.py --task $TASK_NAME --batch_size $BATCH_SIZE --epochs $EPOCHS --warmup_ratio 0.1)

CUDA_VISIBLE_DEVICES=0 python run_dy_glue.py \
    --model_name_or_path ./logs/$TASK_NAME/electra_sst_2_base \
    --task_name $TASK_NAME \
    --data_dir ./data/GLUE/$TASK_NAME \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --max_seq_length 128 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps $WARMUP_STEPS \
    --num_train_epochs $EPOCHS \
    --seed 42 \
    --target_flops_ratio 0.8 \
    --predictor_lr 0.02 \
    --loss_lambda 4 \
    --head_mask_mode gumbel \
    --ffn_mask_mode gumbel \
    --fill_mode zero \
    --load_best_model_at_end \
    --output_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_dir ./logs/$TASK_NAME/$RUN_NAME \
    --logging_steps 50 \
    --run_name $RUN_NAME \
    --disable_tqdm
