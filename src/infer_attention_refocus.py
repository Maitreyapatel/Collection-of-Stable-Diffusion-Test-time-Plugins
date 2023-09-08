import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from src.pipeline_attention_refocus import AttentionRefocusPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = AttentionRefocusPipeline.from_pretrained(stable_diffusion_version, torch_dtype=torch.float16).to(device)
    # stable.enable_attention_slicing()
    return stable

def run_on_prompt(prompt: List[str],
                  model: AttentionRefocusPipeline,
                  controller: AttentionStore,
                  seed: torch.Generator,
                  config) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    bbox=config.bounding_box,
                    object_positions=config.object_positions,
                    max_iter_to_backward=config.max_iter_to_backward,
                    loss_threshold=config.loss_threshold,
                    max_iter_per_step=config.max_iter_per_step,
                    loss_scale=config.loss_scale,
                    attention_store=controller,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    run_standard_sd=config.run_standard_sd,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    attention_aggregation_method=config.attention_aggregation_method)
    image = outputs.images[0]
    return image


def RunAttentionRefocus(config):
    stable = load_model(config)
    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              seed=g,
                              config=config)
        prompt_output_path = config.output_path / str(config.attention_aggregation_method) /config.prompt / str(config.scale_factor)
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')