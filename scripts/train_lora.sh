export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/data_5/data/matt/datasets/VGENOME"
export OUTPUT_DIR="logs/VG_run_regularizer_{lg}_steps_{10k}_lr_{5e-4}_lambda_1_cosine_lora"
# export OUTPUT_DIR="outputs/test"

CUDA_VISIBLE_DEVICES=6 python main.py --exp_name=train \
    --train.pretrained_model_name_or_path=$MODEL_NAME  \
    --train.instance_data_dir=$INSTANCE_DIR \
    --train.output_dir=$OUTPUT_DIR \
    --train.use_lora=True \
    --train.train_text_encoder=False \
    --train.instance_prompt="a photo of sks dog" \
    --train.resolution=512 \
    --train.train_batch_size=1 \
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=5e-4 \
    --train.lr_scheduler="constant" \
    --train.lr_warmup_steps=0 \
    --train.max_train_steps=10000 \
    --train.checkpointing_steps=2000 \
    --train.regularizer="lg" \
    --train.regularizer_weight=1.0 \
    # --debugme=True


# CUDA_VISIBLE_DEVICES=6 python main.py --exp_name=train \
#     --train.pretrained_model_name_or_path=$MODEL_NAME  \
#     --train.instance_data_dir=$INSTANCE_DIR \
#     --train.output_dir=$OUTPUT_DIR \
#     # --train.use_lora=True \
#     --train.train_text_encoder=False \
#     --train.instance_prompt="a photo of sks dog" \
#     --train.resolution=512 \
#     --train.train_batch_size=1 \
#     --train.gradient_accumulation_steps=1 \
#     --train.learning_rate=5e-4 \
#     --train.lr_scheduler="constant" \
#     --train.lr_warmup_steps=0 \
#     --train.max_train_steps=1000 \
#     --train.checkpointing_steps=200 \
#     --train.regularizer="lg" \
#     --debugme=True
