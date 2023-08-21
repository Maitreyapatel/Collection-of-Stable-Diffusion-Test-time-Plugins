# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="logs/VG_run_regularizer_{lg}_steps_{1k}_lr_{5e-6}_lambda_5_cosine"
export RUN_NAME="VG_run_regularizer_{lg}_steps_{1k}_lr_{5e-6}_lambda_5_cosine" #"VG_run_regularizer_{lg}_steps_{10k}_lr_{5e-6}_lambda_10"
export OUTPUT_DIR="outputs"

python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple to the right of the dog"

python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple on top of the dog"

python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="an apple on top of the frog"

python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR/$RUN_NAME \
    --test.prompt="a red vase besides the yellow book"
