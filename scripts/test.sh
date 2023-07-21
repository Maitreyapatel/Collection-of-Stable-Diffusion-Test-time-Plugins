export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_NAME="outputs/test"
export OUTPUT_DIR="outputs/test_images_org"

python main.py --exp_name=test \
    --test.pretrained_model_name_or_path=$MODEL_NAME  \
    --test.inference_outdir=$OUTPUT_DIR \
    --test.prompt="an apple to the right of the dog"
