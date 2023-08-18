import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from src.pipeline_composable_diffusion import ComposableDiffusionPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = ComposableDiffusionPipeline.from_pretrained(stable_diffusion_version).to(
        device
    )
    return stable


def run_on_prompt(
    prompt: List[str],
    model: ComposableDiffusionPipeline,
    seed: torch.Generator,
    config,
) -> Image.Image:
    outputs = model(
        prompt=prompt,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        run_standard_sd=config.run_standard_sd,
        sd_2_1=config.sd_2_1,
        cfg=config,
    )
    image = outputs.images[0]
    return image


def RunComposableDiffusion(config):
    stable = load_model(config)

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator("cuda").manual_seed(seed)
        image = run_on_prompt(
            prompt=config.prompt,
            model=stable,
            seed=g,
            config=config,
        )
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f"{seed}.png")
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f"{config.prompt}.png")
