export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export PKL_PATH="/data_5/data/matt/LSDGen/data/vg_owlvit_regions_v3.pkl"
export INSTANCE_DIR="/data_5/data/matt/datasets/VGENOME/images/"
export OUTPUT_DIR="logs/VG_run_regularizer_{lg}_steps_{1k}_lr_{5e-6}_lambda_5_cosine_higher_cross"

CUDA_VISIBLE_DEVICES=6 python main.py --exp_name="train" \
    --train.pretrained_model_name_or_path=$MODEL_NAME  \
    --train.instance_pkl_path=$PKL_PATH \
    --train.instance_data_dir=$INSTANCE_DIR \
    --train.output_dir=$OUTPUT_DIR \
    --train.train_text_encoder=False \
    --train.resolution=512 \
    --train.train_batch_size=1 \
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=5e-6 \
    --train.lr_scheduler="constant" \
    --train.lr_warmup_steps=0 \
    --train.max_train_steps=1000 \
    --train.checkpointing_steps=2000 \
    --train.regularizer="lg" \
    --train.regularizer_weight=5.0 \
    # --debugme=True
