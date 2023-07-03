# LSDGen
Performing Layout-Free Spatial Compositions for Text-to-Image Diffusion Models

# TO-DO list

## Baselines and repository setup related tasksk:
- [x] Setup initial attention store
- [x] Add Attend-and-Excite
- [x] Add training-free layout guided inference
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
For more details on "Attend & Excite", "Layout Guidance" config requirements, visit: [Config](utils/configs.py)
```bash
# for attend-and-excite
python main.py --exp_name=aae --aae.prompt="a dog and a cat" --aae.token_indices [2,5] --aae.seeds [42]

# for layout-guidance
<<<<<<< HEAD
python main.py --exp_name=lg --lg.seeds=[20,25,30,25,40,42,52,55,60,80,90,101,300] --lg.prompt="an apple to the right of the dog at a beach." --lg.phrases="dog;apple" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention"
=======
python main.py --exp_name=lg --lg.seeds=[20,25,30,25,40,42,52,55,60,80,90,101,300] --lg.prompt="an apple to the right of the dog." --lg.phrases="dog;apple" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="aggregate_attention"

# for attention refocus
 python main.py --exp_name=af --af.seeds=[20] --af.prompt="an apple to the right of the dog." --af.phrases="dog;apple" --af.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]"
>>>>>>> 20be28c (Attention Refocus added)
```


## Currently supported tasks:
* Attend-and-Excite ("aae") -- only inference
* Layout Guided inference ('lg") --only inference, attention aggregation methods - <aggregate_attention, all_attention, aggregate_layer_attention>
* Attention Refocus ("af") -- only inference


# Acknowledgement
This repository is build after [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite), [Training-Free Layout Control with Cross-Attention Guidance](https://github.com/silent-chen/layout-guidance).