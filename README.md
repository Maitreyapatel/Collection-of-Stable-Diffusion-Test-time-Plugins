# Collection of Stable Diffusion Test-time Plugins
Stable Diffusion cannot perform well on various compositions. And it observes the attribute leakage, missing object, and issues with spatial understanding. Several works have focused on introducing test-time plugins (e.g., Attend-And-Excite, Layout-Guidance, etc.) to effectively control the image generation. While other set of work focuses on introducing new layers to control the Stable Diffusion (e.g., ControlNet, GLIGEN). 

However, question still remains how to improve the Stable Diffusion without any additional information. Therefore, this repository focuses on to first understand the limitations of current pre-trianing method and then introducing the new pre-trianing strategy. Currently, repository contains the several test-time baselines methodologies alongwith object-proposal based LoRA fine-tuning strategy. 

**Contributions are welcome!**
If interested, reach out to Maitreya via [mpatel57@asu.edu](mailto:mpatel57@asu.edu).

Note: HuggingFace diffusers is a great library with many of the presented pipelines. However, it is difficult for researchers to modify the existing pipelines in the backend to understand the what's going on behind the scene. This respository wants to bridge this gap.

# Supported Features

## Baselines and repository setup related features:
- [x] Setup initial attention store
- [x] Add Attend-and-Excite
- [x] Add Composable Diffusion Models
- [x] Add training-free layout guided inference with attention aggregation methods - `<aggregate_attention, all_attention, aggregate_layer_attention>`
- [x] Add CAR+SAR based layout guided inference
- [ ] Add support to LLM-based layout generation

## Additional trianing-time features:
- [x] Add biased sampling -- COSINE
- [x] Fine-tune whole UNet
- [x] LoRA based fine-tuning
- [ ] Orthogonal fine-tuning

# How to run the experiments?

## Installation

```bash
conda create LSDGen python=3.8
conda activate LSDGen

pip install -r requirements.txt
```

## Run baseline experiment:
For more details on "Attend & Excite", "Layout Guidance" config requirements, visit: [Config](utils/configs.py)
```bash
# for attend-and-excite
python main.py --exp_name=aae --aae.prompt="a dog and a cat" --aae.token_indices [2,5] --aae.seeds [42]

# for composable-diffusion-models
python main.py --exp_name=cdm --cdm.prompt="a dog and a cat" --cdm.prompt_a="a dog" --cdm.prompt_b="a cat" --cdm.seeds [42]

# for layout-guidance
python main.py --exp_name=lg --lg.seeds=[42] --lg.prompt="an apple to the right of the dog." --lg.phrases="dog;apple" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="aggregate_attention"

# for attention refocus
python main.py --exp_name=af --af.seeds=[42] --af.prompt="an apple to the right of the dog." --af.phrases="dog;apple" --af.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]"
```

## Custom trainer

To fine-tune the stable diffusion model run following command (under-development):
```bash
# bash script defining all parameters
bash ./scripts/train.sh
# bash script for LoRA based fine-tuning
bash ./scripts/train_lora.sh

# alternatively define the parameters manually
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export PKL_PATH="data/coco_data.pkl" # a pre-processed sample pickle file (reach out for the access)
export INSTANCE_DIR="/data/data/matt/datasets/VGENOME"
export OUTPUT_DIR="logs/mask_train_10k"

# change cuda device as needed
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name="train" \
    --train.pretrained_model_name_or_path=$MODEL_NAME  \
    --train.instance_pkl_path=$PKL_PATH \
    --train.instance_data_dir=$INSTANCE_DIR \
    --train.output_dir=$OUTPUT_DIR \
    --train.train_text_encoder=False \
    --train.resolution=512 \
    --train.train_batch_size=1 \ # !!!! current version only supports single batch size
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=5e-6 \
    --train.lr_scheduler="constant" \
    --train.lr_warmup_steps=0 \
    --train.max_train_steps=10000 \
    --train.checkpointing_steps=5000 \
    --train.regularizer="lg" \
    --train.regularizer_weight=5.0 \
    --debugme=True # only pass if you want to perform debugging
```

## Currently supported tasks:
* Attend-and-Excite ("aae")
* Layout Guided inference ('lg") -- attention aggregation methods - `<aggregate_attention, all_attention, aggregate_layer_attention>`
* Attention Refocus ("af")


# Acknowledgement
This repository is build after [diffusers](https://github.com/huggingface/diffusers), [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite), and [Training-Free Layout Control with Cross-Attention Guidance](https://github.com/silent-chen/layout-guidance).
