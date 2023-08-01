import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from src.pipeline_divide_and_conquer import DivideAndConquerPipeline
from utils import ptp_utils, ptp_retrieval_utils, vis_utils
from utils.ptp_utils import AttentionStore
from utils.ptp_retrieval_utils import AttentionRetrievalStore

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
    stable = DivideAndConquerPipeline.from_pretrained(stable_diffusion_version).to(
        device
    )
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
        if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
    }
    pprint.pprint(token_idx_to_word)
    token_indices = input(
        "Please enter the a comma-separated list indices of the tokens you wish to "
        "alter (e.g., 2,5): "
    )
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(
    prompt: List[str],
    model: DivideAndConquerPipeline,
    controller: AttentionStore,
    retrieval_controller: AttentionRetrievalStore,
    token_indices: List[int],
    seed: torch.Generator,
    config,
) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
        ptp_retrieval_utils.register_attention_control_retrieval(
            model, retrieval_controller
        )
    outputs = model(
        prompt=prompt,
        attention_store=controller,
        retrieval_attention_store=retrieval_controller,
        indices_to_alter=token_indices,
        attention_res=config.attention_res,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        max_iter_to_alter=config.max_iter_to_alter,
        run_standard_sd=config.run_standard_sd,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        smooth_attentions=config.smooth_attentions,
        sigma=config.sigma,
        kernel_size=config.kernel_size,
        sd_2_1=config.sd_2_1,
        cfg=config,
    )
    image = outputs.images[0]
    return image


def RunDivideAndConquer(config):
    stable = load_model(config)
    token_indices = (
        get_indices_to_alter(stable, config.prompt)
        if config.token_indices is None
        else config.token_indices
    )

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator("cuda").manual_seed(seed)
        controller = AttentionStore(save_global_store=True)
        retrieval_controller = AttentionRetrievalStore(
            reference_attentionstore=controller, save_global_store=True
        )
        image = run_on_prompt(
            prompt=config.prompt,
            model=stable,
            controller=controller,
            retrieval_controller=retrieval_controller,
            token_indices=token_indices,
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
