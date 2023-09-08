# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="logs/MSCOCO_run_regularizer_{lg}_steps_{1k}_lr_{5e-6}_lambda_5_cosine_higher_cross_and_self_all_attention_L1_norm"
export RUN_NAME="MSCOCO_run_regularizer_{lg}_steps_{1k}_lr_{5e-6}_lambda_5_cosine_higher_cross_and_self_all_attention_L1_norm" #"VG_run_regularizer_{lg}_steps_{10k}_lr_{5e-6}_lambda_10"
export OUTPUT_DIR="outputs"

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple to the right of the dog" &

CUDA_VISIBLE_DEVICES=1 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple on top of the dog" &

CUDA_VISIBLE_DEVICES=2 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple on top of the frog" &

CUDA_VISIBLE_DEVICES=4 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a red vase besides the yellow book" &

wait