srun --gres=gpu:4 -c 24  --mem 320G -p general accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes 4 experimental/ddpo_finetune.py \
    --reward_model_name "clipscore_v2" \
    --prompt_file "finetune_datasets/biased_finetune_prompt_filtered_list.txt" \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=64 \
    --train_batch_size=64 \
    --sample_num_batches_per_epoch=8 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb" \
    --use_lora=True