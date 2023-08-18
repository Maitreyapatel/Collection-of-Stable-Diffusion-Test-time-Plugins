# LSDGen (Divide-And-Conquer)
Performing Layout-Free Spatial Compositions for Text-to-Image Diffusion Models

# TO-DO list

## Baselines and repository setup related tasksk:
- [x] Setup initial attention store
- [x] Add Attend-and-Excite
- [x] Add Composable Diffusion Models
- [x] Add training-free layout guided inference
- [x] Add CAR+SAR based layout guided inference
- [ ] Add support to LLM-based layout generation

## Proposed work
- [ ] (new!) Add object detector based spatial fine-tuning
- [ ] (new!) Add support to all previous layout guided inference to augment both fine-tuning and inference
- [ ] (new!) Add Spatial Attend-and-Excite

## Ablations & Additional feature support
- [ ] (must!) Add biased sampling -- COSINE
- [ ] (ablation) Fine-tune whole UNet
- [ ] (ablation) LoRA based fine-tuning
- [ ] (ablation) Fine-tune QKV weight metrices
- [ ] (ablation) Orthogonal fine-tuning

# How to run the experiments?

## Installation

```bash
conda create LSDGen python=3.8
conda activate LSDGen

pip install -r requirements.txt
```

## Run experiment:
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

# for Divide-And-Conquer (i.e., DAC)
python main.py --exp_name=dac --dac.prompt="Superman on the street" --dac.prompt_a="the street" --dac.token="street" --dac.seeds=[42]
```

To fine-tune the stable diffusion model run following command (under-development):
```bash
# bash script defining all parameters
bash ./scripts/train.sh

# alternatively define the parameters manually
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/data/data/matt/datasets/VGENOME"
export OUTPUT_DIR="logs/mask_train_10k"

# change cuda device as needed
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name=train \
    --train.pretrained_model_name_or_path=$MODEL_NAME  \
    --train.instance_data_dir=$INSTANCE_DIR \
    --train.output_dir=$OUTPUT_DIR \
    --train.instance_prompt="a photo of sks dog" \ # does not matter
    --train.resolution=512 \
    --train.train_batch_size=1 \ # !!!! current version only supports single batch size
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=5e-6 \
    --train.lr_scheduler="constant" \
    --train.lr_warmup_steps=0 \
    --train.max_train_steps=10000 \
    --debugme=True # only pass if you want to perform debugging

```

## Currently supported tasks:
* Attend-and-Excite ("aae") -- only inference
* Layout Guided inference ('lg") --only inference, attention aggregation methods - <aggregate_attention, all_attention, aggregate_layer_attention>
* Attention Refocus ("af") -- only inference


# Acknowledgement
This repository is build after [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite), [Training-Free Layout Control with Cross-Attention Guidance](https://github.com/silent-chen/layout-guidance).
