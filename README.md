# LSDGen
Performing Layout-Free Spatial Compositions for Text-to-Image Diffusion Models

# TO-DO list

## Baselines and repository setup related tasksk:
- [x] Setup initial attention store
- [x] Add Attend-and-Excite
- [ ] Add training-free layout guided inference
- [ ] Add CAR+SAR based layout guided inference
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
For more details on "Attend & Excite" config requirements, visit: [AttendExciteConfig](utils/configs.py)
```bash
python main.py --exp_name=aae --aae.prompt="a dog and a cat" --aae.token_indices [2,5] --aae.seeds [42]
```

## Currently supported tasks:
* Attend-and-Excite ("aae") -- only inference


# Acknowledgement
This repository is build after [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite).