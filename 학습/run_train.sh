# DDP
export OMP_NUM_THREADS=32
num_epochs=5
num_gpu=8
run_name=eff4_v01_discard:1_epoch:$num_epochs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun --nproc_per_node $num_gpu run_train.py \
    --num_device $num_gpu \
    --run_name $run_name \
    --output_dir ./ckpt/$run_name \
    --model_name_or_path google/efficientnet-b4 \
    --dataset_name /mnt/raid6/dltmddbs100/miricanbus/train/train_v01_discard_text_1 \
    --image_column image_path \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --dataloader_num_workers 32 \
    --lr_scheduler_type constant_with_warmup \
    --num_train_epochs $num_epochs \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --eval_ratio 0.1 \
    --weight_decay 0.1 \
    --eval_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --gradient_accumulation_steps 4 > logs/$run_name.out &


# test_ddp
# export OMP_NUM_THREADS=32
# num_epochs=5
# num_gpu=8
# run_name=eval_test

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun --nproc_per_node $num_gpu run_train.py \
#     --num_device $num_gpu \
#     --run_name $run_name \
#     --output_dir ./ckpt/$run_name \
#     --model_name_or_path google/efficientnet-b4 \
#     --dataset_name /mnt/raid6/dltmddbs100/miricanbus/train/train_v01_discard_text_1 \
#     --image_column image_path \
#     --remove_unused_columns False \
#     --do_eval \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --dataloader_num_workers 32 \
#     --lr_scheduler_type constant_with_warmup \
#     --num_train_epochs $num_epochs \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.1 \
#     --eval_strategy steps \
#     --eval_steps 470 \
#     --save_steps 470 \
#     --logging_steps 1 \
#     --save_strategy steps \
#     --save_total_limit 2 \
#     --load_best_model_at_end \
#     --gradient_accumulation_steps 2 > logs/eval_test.out &