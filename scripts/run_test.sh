export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_NAME="logs/VG_run_regularizer_{lg}_steps_{10k}_lr_{5e-6}_lambda_10"
export RUN_NAME="stable_diffusion" #"VG_run_regularizer_{lg}_steps_{10k}_lr_{5e-6}_lambda_10"
export OUTPUT_DIR="outputs"

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a cat and a dog" &

CUDA_VISIBLE_DEVICES=1 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple and a frog" &

CUDA_VISIBLE_DEVICES=2 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a car and a sheep" &

CUDA_VISIBLE_DEVICES=3 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a bench and a boat" &

CUDA_VISIBLE_DEVICES=4 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a bird and a clock" &

CUDA_VISIBLE_DEVICES=5 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a cup and a vase" &

CUDA_VISIBLE_DEVICES=6 python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a book and a vase" &

wait